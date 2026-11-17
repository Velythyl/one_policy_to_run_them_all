"""Functional, jit/vmap-able port of URMA's per-robot MuJoCo env (a1 template) to MJX.

One generic class parameterised by RobotSpec (robot_spec.py). Every function is pure: state in,
state out; a whole batch of envs of ONE robot type is a vmap over these functions (different
robots have different models, so each robot type gets its own vmapped/compiled step).

Ported semantics (CPU file -> here):
  control_functions/rudin2022.py            _torques
  domain_randomization/*                     _sample_dr (seen + unseen + mujoco_model + action delay),
                                             resampled with prob 0.002 per step (never at reset, as on CPU)
  perturbation_functions/default.py          _perturb (prob 0.002 per step)
  command_functions/random.py + sampling     goal resampled with prob 0.002 per step and at an env's first step
  initial_state_functions/random.py          _reset_physics
  reward_functions/rudin_own_var.py          _reward (incl. curriculum and upstream quirks, see notes)
  termination_functions/*                    _terminated
  observation_noise/dropout                  _observation
  context/limb_context.py                    _descriptions (D/G recomputed from the per-env DR values)
Contacts: python loops over data.contact -> fixed-size masks over MJX's contact buffer
(active contact = dist < includemargin, like mj_collision's inclusion rule).
"""
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import json
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from one_policy_to_run_them_all.context import limb_context as lc


def configure_warp_ffi():
    """MJX-Warp wraps every mjwarp call in a Warp-captured CUDA graph (GraphMode.WARP), cached per
    FFI callable and per set of buffer addresses (Warp default: up to 32 graphs per callable). With
    16 robot types (64 callables: step + forward per robot) XLA hands the step/forward calls new
    buffer addresses every PPO iteration, so ~32 new graphs are captured per iteration and never
    evicted; each captured graph keeps its own graph-allocated temporaries (collision/solver scratch),
    and after ~4 iterations (~150 graphs) the next in-capture allocation fails ("Failed to allocate
    72351744 bytes" in convex_narrowphase) although cudaMemGetInfo still reports >30 GB free.
    Fix (default): keep at most 1 graph per callable -> stale graphs are destroyed on re-capture
    (re-capture costs ~nothing: 44k sps either way). Measured on 16 robots x 1024 x 32 (L40S):
      warp, cache 32 -> OOM at iteration 5;  warp, cache 1 -> 44k sps, flat memory;
      warp_staged_ex -> 34k sps, flat memory;  none -> very slow;  jax -> CUDA_ERROR_NOT_SUPPORTED.
    Env knobs (read at import, before the first trace):
      URMA_WARP_GRAPH_MODE = warp | warp_staged_ex | warp_staged | jax | none   (default warp)
      URMA_WARP_GRAPH_CACHE = max cached graphs per callable                     (default 1)"""
    import os
    import warp as wp
    from mujoco.mjx.warp import ffi as mffi
    from warp._src.jax import ffi as wffi
    mode = os.environ.get("URMA_WARP_GRAPH_MODE", "warp").upper()
    getattr(wp.JaxCallableGraphMode, mode)  # validate
    # NB: MJX passes graph_mode=m.opt._impl.graph_mode explicitly to every FFI wrapper, so the mode
    # must be set through mjx.put_model(graph_mode=...) (see MJXRobotEnv), not via wrapper defaults.
    cache = int(os.environ.get("URMA_WARP_GRAPH_CACHE", 1))
    import inspect
    names = [p for p in inspect.signature(wffi.jax_callable).parameters.values() if p.default is not inspect.Parameter.empty]
    dd = list(wffi.jax_callable.__defaults__)
    for i, prm in enumerate(names):
        if prm.name == "graph_cache_max":
            dd[i] = cache
    wffi.jax_callable.__defaults__ = tuple(dd)
    return mode, cache


def patch_warp_external_streams():
    """Warp's FFI callback wraps XLA's stream in a fresh non-owning wp.Stream on every call; the
    wrapper's __del__ unregisters the stream handle from Warp's global stream table. A stale wrapper
    for the same handle that is freed late (cyclic GC, evicted capture) therefore unregisters the
    stream in the middle of a later call -> "Warp error: unknown stream" in capture_begin (seen in the
    first jit_evaluate of a 16-robot run). Reference-count external registrations per handle, so the
    handle is unregistered only when its last wrapper dies."""
    import threading
    from warp._src import context as wctx
    S = wctx.Stream
    if getattr(S, "_urma_refcount_patch", False):
        return
    refs, lock = {}, threading.Lock()
    orig_init, orig_del = S.__init__, S.__del__

    def __init__(self, device=None, priority=0, **kwargs):
        if "cuda_stream" not in kwargs or not kwargs["cuda_stream"]:
            return orig_init(self, device, priority, **kwargs)
        self._cached_event = None
        device = wctx.runtime.get_device(device)
        if not device.is_cuda:
            raise RuntimeError(f"Device {device} is not a CUDA device")
        self.device, self.cuda_stream, self._is_blocking = device, kwargs["cuda_stream"], None
        key = (device.ordinal, int(self.cuda_stream))
        with lock:
            if refs.get(key, 0) == 0:
                device.runtime.core.wp_cuda_stream_register(device.context, self.cuda_stream)
            refs[key] = refs.get(key, 0) + 1
        self._urma_key = key

    def __del__(self):
        key = getattr(self, "_urma_key", None)
        if key is None:
            return orig_del(self)
        try:
            with lock:
                refs[key] -= 1
                last = refs[key] == 0
                if last:
                    del refs[key]
                    wctx.runtime.core.wp_cuda_stream_unregister(self.device.context, self.cuda_stream)
        except (TypeError, AttributeError, KeyError):
            pass

    S.__init__, S.__del__, S._urma_refcount_patch = __init__, __del__, True


WARP_FFI_CONFIG = configure_warp_ffi()
if __import__("os").environ.get("URMA_STREAM_PATCH", "1") == "1":
    patch_warp_external_streams()
from one_policy_to_run_them_all.mjx.robot_spec import RobotSpec


@struct.dataclass
class DR:
    """Per-env domain-randomisation values (seen = shown to the policy, *_noise = hidden)."""
    seen_trunk_mass: jax.Array
    seen_nominal: jax.Array
    seen_torque: jax.Array
    seen_vel: jax.Array
    seen_range: jax.Array
    seen_damping: jax.Array
    seen_armature: jax.Array
    seen_stiffness: jax.Array
    seen_friction: jax.Array
    add_p: jax.Array
    add_d: jax.Array
    add_scale: jax.Array
    p_noise: jax.Array
    d_noise: jax.Array
    motor_noise: jax.Array
    pos_offset: jax.Array
    # model fields (what MuJoCo actually simulates)
    body_mass_trunk: jax.Array
    body_inertia_trunk: jax.Array
    body_ipos_trunk: jax.Array
    foot_size: jax.Array
    damping: jax.Array
    armature: jax.Array
    stiffness: jax.Array
    frictionloss: jax.Array
    friction_ground: jax.Array
    friction_feet: jax.Array
    solref: jax.Array
    gravity_z: jax.Array
    delay_mixed: jax.Array


@struct.dataclass
class EnvState:
    data: Any
    dr: DR
    key: jax.Array
    goal: jax.Array
    action_hist: jax.Array
    current_action: jax.Array
    last_action: jax.Array
    prev_joint_vel: jax.Array
    timers: jax.Array
    episode_step: jax.Array
    steps_done: jax.Array
    sum_track: jax.Array
    episode_return: jax.Array
    obs: Any


def quat_to_mat(q):  # mujoco (w, x, y, z)
    w, x, y, z = q
    return jnp.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_xyz_to_quat(roll, pitch, yaw):
    """scipy Rotation.from_euler('xyz', [r, p, y]) (extrinsic) as mujoco (w, x, y, z)."""
    cr, sr = jnp.cos(roll / 2), jnp.sin(roll / 2)
    cp, sp = jnp.cos(pitch / 2), jnp.sin(pitch / 2)
    cy, sy = jnp.cos(yaw / 2), jnp.sin(yaw / 2)
    return jnp.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


def merge_data(done, reset_d, step_d, impl):
    """Per-env select of batched Data (done -> reset_d). For warp, fields without a world batch dim
    (the shared contact/constraint buffers) are taken from step_d, like mjx Data.where."""
    from mujoco.mjx._src.types import tree_path_to_attr_str
    from mujoco.mjx.warp import types as mjxw_types

    def leaf(path, r, s_):
        if impl == "warp" and not mjxw_types._BATCH_DIM["Data"].get(tree_path_to_attr_str(path), True):
            return s_
        if not hasattr(r, "ndim") or r.ndim == 0 or r.shape[0] != done.shape[0]:
            return s_
        return jnp.where(done.reshape(done.shape + (1,) * (r.ndim - 1)), r, s_)

    return jax.tree_util.tree_map_with_path(leaf, reset_d, step_d)


def primitive_collision_model(xml_path, m0):
    """Replace every colliding mesh geom by a box fitted to its compiled AABB (same geom ids, so
    all index tables still hold). Body inertials are copied back from the original model so only
    the collision shapes change."""
    spec = mujoco.MjSpec.from_file(xml_path)
    for i, g in enumerate(spec.geoms):
        if m0.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH or not (m0.geom_contype[i] or m0.geom_conaffinity[i]):
            continue
        center, half = m0.geom_aabb[i, :3], m0.geom_aabb[i, 3:]
        rot = np.zeros(9); mujoco.mju_quat2Mat(rot, m0.geom_quat[i])
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.meshname = ""
        g.size = np.maximum(half, 1e-4)
        g.pos = m0.geom_pos[i] + rot.reshape(3, 3) @ center
        g.quat = m0.geom_quat[i]
    m = spec.compile()
    assert m.ngeom == m0.ngeom and m.nbody == m0.nbody
    for f in ("body_mass", "body_inertia", "body_ipos", "body_iquat"):
        getattr(m, f)[:] = getattr(m0, f)
    return m


@dataclass
class MJXRobotEnv:
    spec: RobotSpec
    context_arm: str = "urma"
    latent_dir: str = ""
    impl: str = "jax"
    nr_envs_per_type: int = 1  # for the curriculum (total type steps = env steps * nr_envs_per_type)
    model_variant: str = "native"  # native | feet_only (contacts only floor vs feet/trunk/hip)
    eval_mode: bool = False  # CPU env.eval: curriculum coefficient 1.0
    contacts_per_env: int = 64
    solimp_dmax_clamp: float = 0.0  # >0: cap geom solimp dmax (see __post_init__)
    friction_tangential_floor: float = 0.0  # >0: DR tangential friction sampled in [max(min, floor), max]
    # JSON dict of GPU-only physics overrides (experiments; "" = none): cone (elliptic|pyramidal),
    # impratio, iterations, ls_iterations, tolerance, ls_tolerance, noslip_iterations, solver (cg|newton),
    # substep_mult k (timestep/k, k x substeps), robots [only apply to these robot names]
    physics_overrides: str = ""
    impratio_adapt_mu0: float = 0.0  # >0: per-world impratio lowered for DR friction mu < mu0 (see _model_for_all)

    def __post_init__(self):
        sp, c = self.spec, self.spec.cfg
        m = mujoco.MjModel.from_xml_path(str(sp.xml_path))
        if self.model_variant == "primitives":
            m = primitive_collision_model(str(sp.xml_path), m)
        m.opt.timestep = c["timestep"]
        if self.model_variant == "feet_only":
            keep = set(c["collision_groups"]["floor"])
            for k, v in c["collision_groups"].items():
                if k != "floor":
                    keep |= set(v)
            for g in range(m.ngeom):
                if g not in keep:
                    m.geom_contype[g] = 0
                    m.geom_conaffinity[g] = 0
        if self.solimp_dmax_clamp > 0:
            # float32 Warp blows up on contacts whose impedance reaches 1 (solimp dmax = 1, e.g. the
            # go1 / anymal feet "0.015 1 0.031", badger "1.2 1.1" -> clamped to 0.9999 by MuJoCo):
            # zero constraint regularisation. Cap dmax (and dmin) at solimp_dmax_clamp.
            hit = m.geom_solimp[:, 1] > self.solimp_dmax_clamp
            m.geom_solimp[hit, 1] = self.solimp_dmax_clamp
            m.geom_solimp[:, 0] = np.minimum(m.geom_solimp[:, 0], self.solimp_dmax_clamp)
            self.solimp_clamped_geoms = np.nonzero(hit)[0].tolist()
        self.substep_mult = 1
        ov = json.loads(self.physics_overrides) if self.physics_overrides else {}
        if ov and (not ov.get("robots") or self.spec.name in ov["robots"]):
            for k, v in ov.items():
                if k == "robots":
                    continue
                if k == "impratio_adapt_mu0":
                    self.impratio_adapt_mu0 = float(v)
                    continue
                if k == "cone":
                    m.opt.cone = {"elliptic": mujoco.mjtCone.mjCONE_ELLIPTIC, "pyramidal": mujoco.mjtCone.mjCONE_PYRAMIDAL}[v]
                elif k == "solver":
                    m.opt.solver = {"cg": mujoco.mjtSolver.mjSOL_CG, "newton": mujoco.mjtSolver.mjSOL_NEWTON}[v]
                elif k == "substep_mult":
                    self.substep_mult = int(v)
                    m.opt.timestep = m.opt.timestep / int(v)
                else:
                    setattr(m.opt, k, type(getattr(m.opt, k))(v))
            self.physics_overrides_applied = ov
        self.mj_model = m
        if self.impl == "warp":
            from mujoco.mjx.warp import types as mjxw_types
            self.mx = mjx.put_model(m, impl=self.impl, graph_mode=getattr(mjxw_types.GraphMode, WARP_FFI_CONFIG[0]))
        else:
            self.mx = mjx.put_model(m, impl=self.impl)
        self.n = len(c["joint_names"])
        self.nf = len(c["foot_names"])
        self.nu = m.nu
        self.qpos_j = np.asarray(c["qpos_joint_idx"], dtype=int)
        self.qvel_j = np.asarray(c["qvel_joint_idx"], dtype=int)
        self.dof_idx = np.asarray(c["context"]["dof_idx"], dtype=int)  # into dof_*[6:]
        self.jnt_idx = np.asarray(c["context"]["jnt_idx"], dtype=int)  # into jnt_*[1:]
        P = c["params"]
        self.P = P
        self.dt = c["dt"]
        self.nsub = c["nr_substeps"] * self.substep_mult
        self.horizon = c["horizon"]
        d = c["defaults"]
        f64 = lambda x: np.asarray(x, dtype=np.float64)
        self.def_nominal = f64(c["nominal_joint_positions"])
        self.def_vel = f64(c["max_joint_velocities"])
        self.def_torque = f64(d["actuator_ctrlrange"])[:, 1]
        self.def_range = f64(d["jnt_range"])[1:]
        self.def_damping = f64(d["dof_damping"])[6:]
        self.def_armature = f64(d["dof_armature"])[6:]
        self.def_stiffness = f64(d["jnt_stiffness"])[1:]
        self.def_friction = f64(d["dof_frictionloss"])[6:]
        self.def_body_mass = f64(d["body_mass"])
        self.def_trunk_inertia = f64(d["body_inertia"])[1]
        self.def_trunk_com = f64(d["body_ipos"])[1]
        self.def_geom_friction = f64(d["geom_friction"])
        self.def_geom_size = f64(d["geom_size"])
        self.def_solref = f64(d["geom_solref"])
        st = sp.struct
        self.seen_struct, self.model_struct = st["seen_dr"], st["model_dr"]
        self.feet_dr = np.asarray(st["seen_dr"]["foot_size_geoms"], dtype=int)  # geom_size DR (seen fn)
        self.feet_fric = np.asarray(st["model_dr"]["friction_feet_geoms"], dtype=int)  # geom_friction DR (model fn)
        self.foot_noise_power = int(st["seen_dr"]["foot_size_noise_power"])
        # contact group masks over geoms
        ng = m.ngeom
        def mask(ids):
            mk = np.zeros(ng, dtype=bool); mk[list(ids)] = True; return mk
        self.floor_mask = mask(c["collision_groups"]["floor"])
        groups = c["collision_groups"]
        self.foot_masks = np.stack([mask(groups.get(f, [])) for f in c["foot_names"]])
        self.rc_geoms = np.asarray(sp.group_geoms(st["reward_collision_geoms"]), dtype=int)
        self.rc_trunk_mask = mask(sp.group_geoms(st["reward_trunk_groups"])) if st["reward_trunk_groups"] else np.zeros(ng, bool)
        term = st["termination"]
        self.term_kind = term["kind"]
        if self.term_kind == "contact":
            self.term_mask = mask(sp.group_geoms(term["groups"]))
        elif self.term_kind in ("height", "geom_height"):
            self.term_height = float(term["threshold"])
            self.term_geom = int(term.get("geom", -1))
        self.sym_pairs = np.asarray(st["symmetry_pairs"], dtype=int).reshape(-1, 2)
        self.air_pairs = np.asarray(st["air_pairs"], dtype=int).reshape(-1, 2)
        self.nominal_diff_joints = np.asarray(st["nominal_diff_joints"], dtype=int)
        self.foot_obs_mode = st["foot_obs_mode"]
        self.obs_timer_feet = np.asarray(st["obs_timer_feet"], dtype=int)
        cn = c.get("complete_nominal_joint_position") or []
        self.qpos_base = np.asarray(cn, dtype=np.float64) if len(cn) else np.asarray(m.qpos0, dtype=np.float64)
        # context: static parts at the nominal pose (see extract_robot_config.py)
        ctx = c["context"]
        self.ctx_fields = {k: f64(v) for k, v in ctx["fields"].items()}
        self.ctx_feet = f64([ctx["feet"][fn] for fn in c["foot_names"]])
        self.robot_dims = f64(ctx["robot_dimensions"])
        self.robot_height = float(self.robot_dims[2])
        self.limb_bodies = np.asarray(ctx["limb_bodies"], dtype=int)
        self.latents = None
        if lc.is_latent_arm(self.context_arm):
            lat = lc.load_latents(self.latent_dir, self.context_arm, sp.name, m)
            self.latents = f64([lat[b] for b in ctx["body_names"]])
        lat_dim = 0 if self.latents is None else self.latents.shape[1]
        self.layout = lc.joint_layout(self.context_arm, lat_dim)
        self.on = lc.ARM_BLOCKS_ON[self.context_arm]
        self.joint_desc_size = self.layout.width
        self.foot_desc_size = 10
        # reward params
        R = P["reward"]
        self.R = R
        self.init_dr = self._default_dr()

    # ------------------------------------------------------------------ domain randomisation
    def _default_dr(self):
        j = lambda x: jnp.asarray(x, dtype=jnp.float32)
        return DR(
            seen_trunk_mass=j(self.def_body_mass[1]), seen_nominal=j(self.def_nominal), seen_torque=j(self.def_torque),
            seen_vel=j(self.def_vel), seen_range=j(self.def_range), seen_damping=j(self.def_damping),
            seen_armature=j(self.def_armature), seen_stiffness=j(self.def_stiffness), seen_friction=j(self.def_friction),
            add_p=j(0.0), add_d=j(0.0), add_scale=j(0.0),
            p_noise=j(1.0), d_noise=j(1.0), motor_noise=j(1.0), pos_offset=j(np.zeros(self.nu)),
            body_mass_trunk=j(self.def_body_mass[1]), body_inertia_trunk=j(self.def_trunk_inertia),
            body_ipos_trunk=j(self.def_trunk_com), foot_size=j(self.def_geom_size[self.feet_dr, 0]) if len(self.feet_dr) else j(np.zeros(0)),
            damping=j(self.def_damping), armature=j(self.def_armature), stiffness=j(self.def_stiffness),
            frictionloss=j(self.def_friction), friction_ground=j(self.def_geom_friction[0]),
            friction_feet=j(self._fric_target_default()), solref=j(self.def_solref[:, :2]),
            gravity_z=j(self.mj_model.opt.gravity[2]), delay_mixed=jnp.array(False),
        )

    def _fric_target_default(self):
        """friction_feet holds the per-geom friction rows the model DR writes: the foot geoms
        (ground_feet mode) or every geom (sea_snake 'all' mode)."""
        if self.model_struct["friction_mode"] == "ground_feet":
            return self.def_geom_friction[self.feet_fric]
        return self.def_geom_friction

    def _sample_dr(self, key, old: DR):
        S, U, M, A = self.P["seen"], self.P["unseen"], self.P["model_dr"], self.P["action_delay"]
        ks = iter(jax.random.split(key, 40))
        u = lambda lo, hi, shape=(): jax.random.uniform(next(ks), shape, minval=lo, maxval=hi)
        n, nu = self.n, self.nu
        # unseen_robot_functions/default.py
        trunk_mass_nf = u(1 - U["trunk_mass_factor"], 1 + U["trunk_mass_factor"])
        com_nf = u(1 - U["com_displacement_factor"], 1 + U["com_displacement_factor"])
        foot_nf = u(1 - U.get("foot_size_factor", 0.0), 1 + U.get("foot_size_factor", 0.0))
        damp_nf = u(1 - U["joint_damping_factor"], 1 + U["joint_damping_factor"])
        arm_nf = u(1 - U["joint_armature_factor"], 1 + U["joint_armature_factor"])
        stiff_nf = u(1 - U["joint_stiffness_factor"], 1 + U["joint_stiffness_factor"])
        fric_nf = u(1 - U["joint_friction_loss_factor"], 1 + U["joint_friction_loss_factor"])
        p_noise = u(1 - U["p_gain_factor"], 1 + U["p_gain_factor"])
        d_noise = u(1 - U["d_gain_factor"], 1 + U["d_gain_factor"])
        motor = u(1 - U["motor_strength_factor"], 1 + U["motor_strength_factor"])
        offset = u(-U["position_offset"], U["position_offset"], (nu,))
        # seen_robot_functions/default.py
        stay = u(0.0, 1.0) < S["stay_at_default_for_asymmetric_percentage"]
        seen_trunk = self.def_body_mass[1] + u(S["add_trunk_mass_min"], S["add_trunk_mass_max"])
        mass_trunk = seen_trunk * trunk_mass_nf
        inertia_trunk = self.def_trunk_inertia * (mass_trunk / self.def_body_mass[1])
        ipos = (self.def_trunk_com + u(S["add_com_displacement_min"], S["add_com_displacement_max"])) * com_nf
        if len(self.feet_dr):
            foot_size = jnp.full(len(self.feet_dr), u(S["foot_size_min"], S["foot_size_max"]) * foot_nf ** self.foot_noise_power)
        else:
            foot_size = jnp.zeros(0)
        nominal = self.def_nominal + u(S["add_joint_nominal_position_min"], S["add_joint_nominal_position_max"], (n,))
        torque = self.def_torque * (1 + u(-S["torque_limit_factor"], S["torque_limit_factor"], self.def_torque.shape))
        vel = self.def_vel * (1 + u(-S["joint_velocity_factor"], S["joint_velocity_factor"], self.def_vel.shape))
        rng = self.def_range + u(S["add_joint_range_min"], S["add_joint_range_max"], self.def_range.shape)
        sd = self.seen_struct
        shp = self.def_damping.shape
        if sd["damping_mode"] == "uniform":
            damp_s = u(S["joint_damping_min"], S["joint_damping_max"], shp)
        elif sd["damping_mode"] == "add_clamp":  # talos
            damp_s = jnp.maximum(0.0, self.def_damping + u(S["add_joint_damping_min"], S["add_joint_damping_max"], shp))
        else:  # factor (cassie)
            damp_s = self.def_damping * (1 + u(-S["joint_damping_factor"], S["joint_damping_factor"], shp))
        if sd["frictionloss_mode"] == "uniform":
            fric_s = u(S["joint_friction_loss_min"], S["joint_friction_loss_max"], self.def_friction.shape)
        elif sd["frictionloss_mode"] == "add_clamp":
            fric_s = jnp.maximum(0.0, self.def_friction + u(S["add_joint_friction_loss_min"], S["add_joint_friction_loss_max"], self.def_friction.shape))
        else:
            raise NotImplementedError(sd["frictionloss_mode"])
        arm_s = u(S["joint_armature_min"], S["joint_armature_max"], self.def_armature.shape)
        stiff_s = u(S["joint_stiffness_min"], S["joint_stiffness_max"], self.def_stiffness.shape)
        keep = sd["keep_default"]  # indices that keep their default even when randomised (cassie, atlas)
        if "damping" in keep:
            damp_s = damp_s.at[jnp.asarray(keep["damping"])].set(self.def_damping[keep["damping"]])
        if "stiffness" in keep:
            stiff_s = stiff_s.at[jnp.asarray(keep["stiffness"])].set(self.def_stiffness[keep["stiffness"]])
        assert set(keep) <= {"damping", "stiffness"}, keep
        damping = jnp.where(stay, self.def_damping, damp_s)
        # upstream typo: the stay-at-default branch sets `seen_armature`, so armature keeps its previous sample
        assert sd["armature_stay_typo"]
        armature = jnp.where(stay, old.seen_armature, arm_s)
        stiffness = jnp.where(stay, self.def_stiffness, stiff_s)
        friction = jnp.where(stay, self.def_friction, fric_s)
        add_p = u(S["add_p_gain_min"], S["add_p_gain_max"])
        add_d = u(S["add_d_gain_min"], S["add_d_gain_max"])
        add_s = u(S["add_scaling_factor_min"], S["add_scaling_factor_max"])
        # mujoco_model_functions/default.py
        it = u(0.0, 1.0)
        lerp = lambda lo, hi, t: lo + (hi - lo) * t
        # friction_tangential_floor > 0 (GPU deviation, off by default): sample the tangential friction in
        # [max(min, floor), max]. Warp (float32, pyramidal cone, impratio 100) diverges for mu < ~0.05.
        tang = lerp(max(M["friction_tangential_min"], self.friction_tangential_floor), M["friction_tangential_max"], it)
        if self.model_struct["friction_mode"] == "ground_feet":
            fg = jnp.stack([tang, lerp(M["friction_torsional_ground_min"], M["friction_torsional_ground_max"], it),
                            lerp(M["friction_rolling_ground_min"], M["friction_rolling_ground_max"], it)])
            ff = jnp.stack([tang, lerp(M["friction_torsional_feet_min"], M["friction_torsional_feet_max"], it),
                            lerp(M["friction_rolling_feet_min"], M["friction_rolling_feet_max"], it)])
        else:  # sea_snake: one friction triple for every geom
            fg = jnp.stack([tang, lerp(M["friction_torsional_min"], M["friction_torsional_max"], it),
                            lerp(M["friction_rolling_min"], M["friction_rolling_max"], it)])
            ff = fg
        ff = jnp.broadcast_to(ff, self._fric_target_default().shape)
        it2 = u(0.0, 1.0)
        damp_c = lerp(M["damping_min"], M["damping_max"], it2)
        stiff_c = lerp(M["stiffness_min"], M["stiffness_max"], it2)
        solref = jnp.broadcast_to(jnp.stack([-stiff_c, -damp_c]), self.def_solref[:, :2].shape)
        grav = -u(M["gravity_min"], M["gravity_max"])
        mixed = u(0.0, 1.0) < A["mixed_chance"]
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        return DR(
            seen_trunk_mass=f32(seen_trunk), seen_nominal=f32(nominal), seen_torque=f32(torque), seen_vel=f32(vel),
            seen_range=f32(rng), seen_damping=f32(damping), seen_armature=f32(armature), seen_stiffness=f32(stiffness),
            seen_friction=f32(friction), add_p=f32(add_p), add_d=f32(add_d), add_scale=f32(add_s),
            p_noise=f32(p_noise), d_noise=f32(d_noise), motor_noise=f32(motor), pos_offset=f32(offset),
            body_mass_trunk=f32(mass_trunk), body_inertia_trunk=f32(inertia_trunk), body_ipos_trunk=f32(ipos),
            foot_size=f32(foot_size), damping=f32(damping * damp_nf), armature=f32(armature * arm_nf),
            stiffness=f32(stiffness * stiff_nf), frictionloss=f32(friction * fric_nf),
            friction_ground=f32(fg), friction_feet=f32(ff), solref=f32(solref), gravity_z=f32(grav), delay_mixed=mixed,
        )

    DR_MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_friction", "geom_solref",
                       "actuator_ctrlrange", "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss",
                       "jnt_stiffness", "opt")

    def model_for(self, dr: DR):
        """MJX model with this env's DR values written in (used inside vmap -> batched model)."""
        full = self._model_for_all(dr)
        keep = getattr(self, "dr_model_fields", self.DR_MODEL_FIELDS)
        return self.mx.replace(**{k: getattr(full, k) for k in keep})

    def _model_for_all(self, dr: DR):
        mx = self.mx
        body_mass = mx.body_mass.at[1].set(dr.body_mass_trunk)
        body_inertia = mx.body_inertia.at[1].set(dr.body_inertia_trunk)
        body_ipos = mx.body_ipos.at[1].set(dr.body_ipos_trunk)
        geom_size = mx.geom_size.at[self.feet_dr, 0].set(dr.foot_size) if len(self.feet_dr) else mx.geom_size
        if self.model_struct["friction_mode"] == "ground_feet":
            geom_friction = mx.geom_friction.at[0].set(dr.friction_ground).at[self.feet_fric].set(dr.friction_feet)
        else:
            geom_friction = dr.friction_feet
        geom_solref = mx.geom_solref.at[:, :2].set(dr.solref)
        ctrlrange = jnp.stack([-dr.seen_torque, dr.seen_torque], axis=-1)
        jnt_range = mx.jnt_range.at[1:].set(dr.seen_range)
        dof_damping = mx.dof_damping.at[6:].set(dr.damping)
        dof_armature = mx.dof_armature.at[6:].set(dr.armature)
        dof_frictionloss = mx.dof_frictionloss.at[6:].set(dr.frictionloss)
        jnt_stiffness = mx.jnt_stiffness.at[1:].set(dr.stiffness)
        opt = mx.opt.replace(gravity=mx.opt.gravity.at[2].set(dr.gravity_z))
        if self.impratio_adapt_mu0 > 0 and self.impl == "warp":
            # Per-world impratio so that the pyramidal friction regulariser R ~ 2 mu^2 (1 + mu^2) / impratio never
            # drops below its value at mu0 with the model impratio: worlds with mu >= mu0 keep the exact model
            # physics; lower-mu worlds get proportionally softer (lower impratio) friction rows.
            mu = dr.friction_ground[0] if self.model_struct["friction_mode"] == "ground_feet" else jnp.max(dr.friction_feet[..., 0])
            g = lambda u: u * u * (1.0 + u * u)
            imp = self.mj_model.opt.impratio * jnp.minimum(1.0, g(mu) / g(self.impratio_adapt_mu0))
            isq = opt._impl.impratio_invsqrt
            opt = opt.replace(_impl=opt._impl.replace(impratio_invsqrt=jnp.broadcast_to(1.0 / jnp.sqrt(imp), isq.shape).astype(isq.dtype)))
        return mx.replace(body_mass=body_mass, body_inertia=body_inertia, body_ipos=body_ipos, geom_size=geom_size,
                          geom_friction=geom_friction, geom_solref=geom_solref, actuator_ctrlrange=ctrlrange,
                          jnt_range=jnt_range, dof_damping=dof_damping, dof_armature=dof_armature,
                          dof_frictionloss=dof_frictionloss, jnt_stiffness=jnt_stiffness, opt=opt)

    # ------------------------------------------------------------------ context
    def descriptions(self, dr: DR):
        """limb_context.compute_name_to_description_vector, with K-geometry at the nominal pose and
        the DR-dependent fields (seen nominal/range, D, gains, mass) from this env's DR state."""
        f = {k: jnp.asarray(v, jnp.float32) for k, v in self.ctx_fields.items()}
        col = lambda x: x.reshape(self.n, 1)
        # URMA stock fields (urma arm), URMA normalisation
        f["urma_joint_range"] = dr.seen_range[self.jnt_idx] / 4.6
        f["urma_nominal_pos"] = col(dr.seen_nominal / 4.6)
        f["urma_torque_limit"] = col(dr.seen_torque / (1000.0 / 2) - 1.0)
        f["urma_max_joint_vel"] = col(dr.seen_vel / (35.0 / 2) - 1.0)
        f["urma_damping"] = col(dr.seen_damping[self.dof_idx] / (10.0 / 2) - 1.0)
        f["urma_armature"] = col(dr.seen_armature[self.dof_idx] / (0.2 / 2) - 1.0)
        f["urma_stiffness"] = col(dr.seen_stiffness[self.jnt_idx] / (30.0 / 2) - 1.0)
        f["urma_frictionloss"] = col(dr.seen_friction[self.dof_idx] / (1.2 / 2) - 1.0)
        # ModuMorph D fields (seen DR values), ModuMorph min-max with the refit bounds (limb_context.mm_normalise)
        B = lc.modumorph_bounds()

        def mm(name, x):
            lo, hi = (jnp.asarray(v, jnp.float32) for v in B[name])
            return -1.0 * (lo != hi) + 2.0 * (x - lo) / (hi - lo + 1e-8)
        f["joint_range"] = mm("joint_range", dr.seen_range[self.jnt_idx])
        f["nominal_pos"] = mm("nominal_pos", col(dr.seen_nominal))
        f["gear"] = mm("gear", col(dr.seen_torque))
        f["max_joint_vel"] = mm("max_joint_vel", col(dr.seen_vel))
        f["damping"] = mm("damping", col(dr.seen_damping[self.dof_idx]))
        f["armature"] = mm("armature", col(dr.seen_armature[self.dof_idx]))
        f["stiffness"] = mm("stiffness", col(dr.seen_stiffness[self.jnt_idx]))
        f["frictionloss"] = mm("frictionloss", col(dr.seen_friction[self.dof_idx]))
        # body_mass (limbs): only the trunk mass is randomised and the trunk is never a limb -> static field from the config
        gains = jnp.stack([self.P["control"]["p_gain"] + dr.add_p, self.P["control"]["d_gain"] + dr.add_d,
                           self.P["control"]["scaling_factor"] + dr.add_scale])
        mass = jnp.sum(jnp.asarray(self.def_body_mass, jnp.float32).at[1].set(dr.seen_trunk_mass))
        g = {
            "p_d_gain_scaling": gains / jnp.array([100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0,
            "robot_mass": jnp.reshape(mass / (170.0 / 2) - 1.0, (1,)),
            "robot_dims": jnp.asarray(self.robot_dims / (2.0 / 2) - 1.0, jnp.float32),
        }
        cols = []
        for block, name, s, e in self.layout.columns:
            if block not in self.on:
                cols.append(jnp.zeros((self.n, e - s)))
            elif block == "G":
                cols.append(jnp.broadcast_to(g[name], (self.n, e - s)))
            elif block == "LAT":
                cols.append(jnp.asarray(self.latents, jnp.float32))
            else:
                cols.append(f[name])
        jd = jnp.concatenate(cols, axis=-1)
        gvec = jnp.concatenate([g["p_d_gain_scaling"], g["robot_mass"], g["robot_dims"]])
        feet_pos = jnp.asarray(self.ctx_feet, jnp.float32)
        if "K" not in self.on:
            feet_pos = jnp.zeros_like(feet_pos)
        if "G" not in self.on:
            fg = jnp.zeros((self.nf, 7))
        else:
            fg = jnp.broadcast_to(gvec, (self.nf, 7))
        fd = jnp.concatenate([feet_pos, fg], axis=-1)
        return jd, fd, gvec

    # ------------------------------------------------------------------ contacts (batch level)
    def contact_masks(self, d, nr_envs):
        """Per-env contact facts from a batched Data. MJX-jax keeps one contact buffer per env;
        MJX-warp keeps ONE buffer for all worlds (naconmax slots, `worldid`, first `nacon` valid).
        Returns foot_contact (N, nf) bool, nr_collisions (N,) float, contact-termination (N,) bool."""
        if self.impl == "warp":
            imp = d._impl
            geom = imp.contact__geom.reshape(-1, 2)
            g1, g2 = geom[:, 0], geom[:, 1]
            wid = imp.contact__worldid.reshape(-1)
            dist = imp.contact__dist.reshape(-1)
            marg = imp.contact__includemargin.reshape(-1)
            nacon = jnp.asarray(imp.nacon).reshape(-1)[0]
            valid = (jnp.arange(g1.shape[0]) < nacon) & (dist < marg)
            seg = jnp.where(valid, wid, nr_envs)  # invalid slots -> dummy segment
            anyw = lambda b: jax.ops.segment_max(b.astype(jnp.int32), seg, num_segments=nr_envs + 1)[:nr_envs] > 0
        else:
            c = d.contact
            g1, g2 = c.geom1, c.geom2  # (N, ncon)
            valid = c.dist < c.includemargin
            anyw = lambda b: jnp.any(b, axis=-1)
        floor = jnp.asarray(self.floor_mask)
        fm = jnp.asarray(self.foot_masks)
        f1, f2 = floor[g1], floor[g2]
        foot = jnp.stack([anyw(valid & ((f1 & fm[i][g2]) | (f2 & fm[i][g1]))) for i in range(self.nf)], axis=-1)
        nr = jnp.zeros(nr_envs)
        for gid in self.rc_geoms:
            nr = nr + anyw(valid & ((g1 == gid) | (g2 == gid))).astype(jnp.float32)
        trm = jnp.asarray(self.rc_trunk_mask)
        if self.rc_trunk_mask.any():
            nr = nr + anyw(valid & (trm[g1] | trm[g2])).astype(jnp.float32)
        if self.term_kind == "contact":
            tm = jnp.asarray(self.term_mask)
            term = anyw(valid & (tm[g1] | tm[g2]))
        elif self.term_kind == "height":
            term = d.qpos[:, 2] < self.term_height
        elif self.term_kind == "geom_height":  # atlas: data.geom("middle_chest").xpos[2]
            term = d.geom_xpos[:, self.term_geom, 2] < self.term_height
        else:
            term = jnp.zeros(nr_envs, bool)
        return foot, nr, term

    # ------------------------------------------------------------------ physics
    def _torques(self, d, dr, action):
        C = self.P["control"]
        p, dg, s = C["p_gain"] + dr.add_p, C["d_gain"] + dr.add_d, C["scaling_factor"] + dr.add_scale
        target = dr.seen_nominal + action * s
        tau = p * dr.p_noise * (target - d.qpos[self.qpos_j] + dr.pos_offset) - dg * dr.d_noise * d.qvel[self.qvel_j]
        return tau * dr.motor_noise

    def _initial_qpos_qvel(self, key, dr):
        """initial_state_functions/random.py"""
        I = self.P["init"]
        k = jax.random.split(key, 8)
        u = lambda kk, lo, hi, shape=(): jax.random.uniform(kk, shape, minval=lo, maxval=hi)
        roll = u(k[0], -jnp.pi * I["roll_angle_factor"], jnp.pi * I["roll_angle_factor"])
        pitch = u(k[1], -jnp.pi * I["pitch_angle_factor"], jnp.pi * I["pitch_angle_factor"])
        yaw = u(k[2], -jnp.pi * I["yaw_angle_factor"], jnp.pi * I["yaw_angle_factor"])
        quat = euler_xyz_to_quat(roll, pitch, yaw)
        jpos = dr.seen_nominal * u(k[3], I["nominal_joint_position_factor"], 1 + I["nominal_joint_position_factor"], (self.n,))
        jvel = dr.seen_vel * u(k[4], -I["joint_velocity_factor"], I["joint_velocity_factor"])
        lin = u(k[5], -I["max_linear_velocity"], I["max_linear_velocity"], (3,))
        ang = u(k[6], -I["max_angular_velocity"], I["max_angular_velocity"], (3,))
        qpos = jnp.asarray(self.qpos_base, jnp.float32)  # masked robots: complete nominal pose
        qpos = qpos.at[0:3].set(jnp.array([0.0, 0.0, self.spec.cfg["initial_drop_height"]])).at[3:7].set(quat)
        qpos = qpos.at[self.qpos_j].set(jpos)
        qvel = jnp.zeros(self.mj_model.nv).at[0:3].set(lin).at[3:6].set(ang).at[self.qvel_j].set(jvel)
        return qpos, qvel

    def make_batch_data(self, nr_envs, contacts_per_env=None, constraints_per_env=None):
        contacts_per_env = contacts_per_env or self.contacts_per_env
        njmax = constraints_per_env or contacts_per_env * 8
        if self.impl == "warp":
            mk = lambda _: mjx.make_data(self.mj_model, impl="warp", naconmax=nr_envs * contacts_per_env, njmax=njmax)
        else:
            mk = lambda _: mjx.make_data(self.mj_model, impl="jax")
        return jax.vmap(mk)(jnp.arange(nr_envs))

    # ------------------------------------------------------------------ observation
    def _observation(self, key, d, dr, goal, current_action, timers, foot_contact):
        N, D = self.P["noise"], self.P["dropout"]
        k = jax.random.split(key, 6)
        R = quat_to_mat(d.qpos[3:7])
        jd, fd, gvec = self.descriptions(dr)
        jp = d.qpos[self.qpos_j] - dr.seen_nominal + jax.random.uniform(k[0], (self.n,), minval=-N["joint_position_noise"], maxval=N["joint_position_noise"])
        ang = d.qvel[3:6] + jax.random.uniform(k[1], (3,), minval=-N["trunk_angular_velocity_noise"], maxval=N["trunk_angular_velocity_noise"])
        jv = d.qvel[self.qvel_j] + jax.random.uniform(k[2], (self.n,), minval=-N["joint_velocity_noise"], maxval=N["joint_velocity_noise"])
        grav = R.T @ jnp.array([0.0, 0.0, -1.0]) + jax.random.uniform(k[3], (3,), minval=-N["gravity_vector_noise"], maxval=N["gravity_vector_noise"])
        mv = 0.0  # missing_value
        jmask = jax.random.uniform(k[4], (self.n,)) < D["dynamic_dropout_chance"]
        jp, jv, pa = (jnp.where(jmask, mv, x) for x in (jp, jv, current_action))
        if self.foot_obs_mode == "live":
            fc = foot_contact.astype(jnp.float32)
            ft = timers[self.obs_timer_feet]
            fmask = jax.random.uniform(k[5], (self.nf,)) < D["dynamic_dropout_chance"]
            fc, ft = (jnp.where(fmask, mv, x) for x in (fc, ft))
        else:  # badger family / sea_snake: foot contact + timer obs are never written (missing_value)
            fc = ft = jnp.full((self.nf,), mv)
        jstate = jnp.stack([jp / 4.6, jv / 35.0, pa / 10.0], axis=-1)
        fstate = jnp.stack([fc / 0.5 - 1.0, jnp.clip(ft / (5.0 / 2) - 1.0, -1.0, 1.0)], axis=-1)
        lin = jnp.clip((R.T @ d.qvel[0:3]) / 10.0, -1.0, 1.0)
        angn = jnp.clip(ang / 50.0, -1.0, 1.0)
        height = jnp.clip(d.qpos[2:3] / (2 * self.robot_height / 2) - 1.0, -1.0, 1.0)
        g_policy = jnp.concatenate([angn, goal, grav, gvec])
        g_critic = jnp.concatenate([lin, angn, goal, grav, height, gvec])
        return {"jd": jd, "js": jstate, "fd": fd, "fs": fstate, "gp": g_policy, "gc": g_critic}

    # ------------------------------------------------------------------ reward
    def _reward(self, d, dr, goal, action, last_action, prev_vel, timers, foot_contact, nr_coll, type_steps):
        R = self.R
        cur = 1.0 if self.eval_mode else jnp.minimum(type_steps / R["curriculum_steps"], 1.0)
        Rm = quat_to_mat(d.qpos[3:7])
        local_lin = Rm.T @ d.qvel[0:3]
        ang = d.qvel[3:6]
        track_xy = R["tracking_xy_velocity_command_coeff"] * jnp.exp(-jnp.sum((goal[:2] - local_lin[:2]) ** 2) / R["xy_tracking_temperature"])
        track_yaw = R["tracking_yaw_velocity_command_coeff"] * jnp.exp(-((ang[2] - goal[2]) ** 2) / R["yaw_tracking_temperature"])
        r_z = cur * R["z_velocity_coeff"] * -(local_lin[2] ** 2)
        r_ang = cur * R["pitch_roll_vel_coeff"] * -jnp.sum(ang[:2] ** 2)
        roll = jnp.arctan2(Rm[2, 1], Rm[2, 2])
        pitch = jnp.arcsin(jnp.clip(-Rm[2, 0], -1.0, 1.0))
        r_pos = cur * R["pitch_roll_pos_coeff"] * -(roll ** 2 + pitch ** 2)
        qj = d.qpos[self.qpos_j]
        if len(self.nominal_diff_joints):
            nd = jnp.sum((qj[self.nominal_diff_joints] - dr.seen_nominal[self.nominal_diff_joints]) ** 2)
        else:
            nd = 0.0
        r_nom = cur * R["joint_nominal_diff_coeff"] * -nd
        rng = dr.seen_range[self.jnt_idx]
        mid, half = (rng[:, 0] + rng[:, 1]) / 2, (rng[:, 1] - rng[:, 0]) / 2 * R["soft_joint_position_limit"]
        lo, hi = mid - half, mid + half
        lim = -jnp.minimum(qj - lo, 0.0).sum() + jnp.maximum(qj - hi, 0.0).sum()
        r_lim = cur * R["joint_position_limit_coeff"] * -lim
        vj = d.qvel[self.qvel_j]
        r_vel = cur * R["joint_velocity_coeff"] * -jnp.sum(vj ** 2)
        # upstream precedence quirk kept: (prev - vel / dt)
        r_acc = cur * R["joint_acceleration_coeff"] * -jnp.sum((prev_vel - d.qvel[self.qvel_j] / self.dt) ** 2)
        r_tq = cur * R["joint_torque_coeff"] * -jnp.sum(d.qfrc_actuator[self.qvel_j] ** 2)
        r_ar = cur * R["action_rate_coeff"] * -jnp.sum((action - last_action) ** 2)
        r_col = cur * R["collision_coeff"] * -nr_coll
        r_h = cur * R["base_height_coeff"] * -((d.qpos[2] - R["nominal_trunk_z"]) ** 2)
        if len(self.air_pairs):
            air = jnp.sum(jnp.where(foot_contact[self.air_pairs[:, 0]], timers[self.air_pairs[:, 1]] - R["air_time_max"], 0.0))
        else:
            air = 0.0
        r_air = cur * R["air_time_coeff"] * air
        viol = jnp.sum(~foot_contact[self.sym_pairs[:, 0]] & ~foot_contact[self.sym_pairs[:, 1]]) if len(self.sym_pairs) else 0.0
        r_sym = cur * R["symmetry_air_coeff"] * -viol
        reward = track_xy + track_yaw + r_z + r_ang + r_pos + r_nom + r_lim + r_vel + r_acc + r_tq + r_ar + r_col + r_h + r_air + r_sym
        reward = jnp.maximum(reward, 0.0)
        curv = jnp.array([local_lin[0], local_lin[1], ang[2]])
        perc = jnp.mean(1 - jnp.abs(curv - goal) / jnp.abs(goal))
        perc = jnp.where(jnp.isfinite(perc), jnp.maximum(perc, 0.0), 0.0)
        # per-term contributions (pre-clip), logging only; order = wandb_logger.REWARD_TERMS
        terms = jnp.stack([jnp.asarray(x, jnp.float32) for x in (track_xy, track_yaw, r_z, r_ang, r_pos, r_nom, r_lim, r_vel,
                                                                  r_acc, r_tq, r_ar, r_col, r_h, r_air, r_sym)])
        return reward, perc, jnp.sum((curv - goal) ** 2), terms


    # ------------------------------------------------------------------ batched API
    def _fresh_state(self, key, dr, goal, steps_done):
        k1, k2 = jax.random.split(key)
        qpos, qvel = self._initial_qpos_qvel(k1, dr)
        A = int(self.P["action_delay"]["max_nr_delay_steps"])
        zu = jnp.zeros(self.nu)
        return dict(qpos=qpos, qvel=qvel, goal=goal, action_hist=jnp.zeros((A + 1, self.nu)), current_action=zu,
                    last_action=zu, prev_joint_vel=jnp.zeros(self.n), timers=jnp.zeros(self.nf),
                    episode_step=jnp.zeros((), jnp.int32), steps_done=steps_done, sum_track=jnp.zeros((), jnp.float32),
                    episode_return=jnp.zeros((), jnp.float32), key=k2)

    def _set_state(self, data, qpos, qvel):
        z = jnp.zeros_like
        return data.replace(qpos=qpos, qvel=qvel, ctrl=z(data.ctrl), act=z(data.act), qacc_warmstart=z(data.qacc_warmstart))

    def batch_reset(self, key, nr_envs, data=None):
        keys = jax.random.split(key, nr_envs)
        dr = jax.vmap(lambda _: self.init_dr)(jnp.arange(nr_envs))
        fs = jax.vmap(lambda k, d: self._fresh_state(k, d, jnp.zeros(3), jnp.zeros((), jnp.int32)))(keys, dr)
        data = self.make_batch_data(nr_envs) if data is None else data
        data = data.replace(qpos=fs["qpos"], qvel=fs["qvel"])
        models = jax.vmap(self.model_for)(dr)
        data = jax.vmap(mjx.forward, in_axes=(0, 0))(models, data)
        foot, _, _ = self.contact_masks(data, nr_envs)
        okeys = jax.vmap(lambda k: jax.random.split(k)[0])(fs["key"])
        obs = jax.vmap(self._observation)(okeys, data, dr, fs["goal"], fs["current_action"], fs["timers"], foot)
        fs.pop("qpos"); fs.pop("qvel")
        return EnvState(data=data, dr=dr, obs=obs, **fs)

    def batch_step(self, s: EnvState, raw_action):
        """One control step for all N envs of this robot type (gymnasium autoreset semantics)."""
        P = self.P
        N = raw_action.shape[0]

        def pre(s1, a):
            key, kc, kcs, kdl = jax.random.split(s1.key, 4)
            cmin = jnp.array([P["command"]["min_x_velocity"], P["command"]["min_y_velocity"], P["command"]["min_yaw_velocity"]])
            cmax = jnp.array([P["command"]["max_x_velocity"], P["command"]["max_y_velocity"], P["command"]["max_yaw_velocity"]])
            new_cmd = jax.random.uniform(kc, (3,), minval=cmin, maxval=cmax)
            resample = (jax.random.uniform(kcs) < P["command_sampling_prob"]) | (s1.steps_done == 0)
            goal = jnp.where(resample, new_cmd, s1.goal)
            A = s1.action_hist.shape[0] - 1
            hist = jnp.roll(s1.action_hist, -1, axis=0).at[-1].set(a[: self.nu])
            nd = jnp.where(s1.dr.delay_mixed, jax.random.randint(kdl, (), 0, A + 1), 0)
            action = hist[-1 - nd]
            tau = self._torques(s1.data, s1.dr, action)
            return key, goal, hist, action, tau

        key, goal, hist, action, tau = jax.vmap(pre)(s, raw_action)
        models = jax.vmap(self.model_for)(s.dr)
        d = s.data.replace(ctrl=tau)
        stepper = jax.vmap(mjx.step, in_axes=(0, 0))
        d = jax.lax.fori_loop(0, self.nsub, lambda i, dd: stepper(models, dd), d)
        # Diverged worlds: MuJoCo C (CPU env) auto-resets such data (mj_checkPos/Vel/Acc warnings);
        # MJX/Warp does not. Here: roll the world back to its pre-step state, end the episode
        # (terminated) with zero reward, and count it in info["sim_unstable"].
        unstable = ~(jnp.all(jnp.isfinite(d.qpos), axis=-1) & jnp.all(jnp.isfinite(d.qvel), axis=-1)
                     & jnp.all(jnp.abs(d.qvel) < 1e4, axis=-1))
        d = merge_data(unstable, s.data, d, self.impl)

        def post_physics(key1, d_qvel, dr):
            key1, kdr, kdrs, kp, kps, ko, kr = jax.random.split(key1, 7)
            qvel = d_qvel.at[self.qvel_j].set(jnp.clip(d_qvel[self.qvel_j], -dr.seen_vel, dr.seen_vel))
            do_dr = jax.random.uniform(kdrs) < P["dr_sampling_prob"]
            new_dr = self._sample_dr(kdr, dr)
            dr = jax.tree_util.tree_map(lambda a, b: jnp.where(do_dr, a, b), new_dr, dr)
            T = P["perturbation"]
            kp1, kp2 = jax.random.split(kp)
            push = jax.random.uniform(kp1, (3,), minval=jnp.array([T["push_velocity_x_min"], T["push_velocity_y_min"], T["push_velocity_z_min"]]),
                                      maxval=jnp.array([T["push_velocity_x_max"], T["push_velocity_y_max"], T["push_velocity_z_max"]]))
            set_push = jax.random.uniform(kp2) < T["set_chance"]
            lin = jnp.where(set_push, push, qvel[0:3] + push * T["additive_multiplier"])
            do_push = jax.random.uniform(kps) < P["perturb_sampling_prob"]
            qvel = jnp.where(do_push, qvel.at[0:3].set(lin), qvel)
            return key1, qvel, dr, ko, kr

        key, qvel, dr, ko, kr = jax.vmap(post_physics)(key, d.qvel, s.dr)
        d = d.replace(qvel=qvel)
        # contacts of the post-step state (velocity edits do not change contacts)
        foot, nr_coll, term_c = self.contact_masks(d, N)

        def post(dd, dr1, goal1, action1, s1, ko1, foot1, nrc1, term1):
            obs = self._observation(ko1, dd, dr1, goal1, action1, s1.timers, foot1)
            trunc = s1.episode_step + 1 >= self.horizon
            type_steps = s1.steps_done.astype(jnp.float32) * self.nr_envs_per_type
            reward, perc, track_err, terms = self._reward(dd, dr1, goal1, action1, s1.last_action, s1.prev_joint_vel, s1.timers, foot1, nrc1, type_steps)
            timers = jnp.where(foot1, 0.0, s1.timers + self.dt)
            done = term1 | trunc
            sum_track = s1.sum_track + perc
            ep_ret = s1.episode_return + reward
            info = {"reward": reward, "terminated": term1, "truncated": trunc, "done": done,
                    "episode_return": ep_ret, "episode_length": s1.episode_step + 1,
                    "eps_track_perf_perc": sum_track / self.horizon, "track_perf_perc": perc, "track_err_sq": track_err,
                    "reward_terms": terms}
            return obs, timers, sum_track, ep_ret, info

        obs, timers, sum_track, ep_ret, info = jax.vmap(post)(d, dr, goal, action, s, ko, foot, nr_coll, term_c | unstable)
        info["reward"] = jnp.where(unstable, 0.0, info["reward"])
        info["episode_return"] = jnp.where(unstable, s.episode_return, info["episode_return"])
        ep_ret = info["episode_return"]
        info["sim_unstable"] = unstable
        info["reward_terms"] = jnp.where(unstable[:, None], 0.0, info["reward_terms"])
        info["ctrl"] = tau  # applied torques (probes only; DCE'd in training)
        if self.impl == "warp":  # buffer-pressure diagnostics (overflow bits are sticky per world)
            imp = d._impl
            ov = jnp.asarray(imp.overflow).reshape(N, -1).max(-1)
            for name, bit in (("nefc", 1), ("broad", 4), ("narrow", 8), ("ccd", 16), ("epa", 256), ("iters", 512)):
                info[f"ovf_{name}"] = (ov & bit) > 0
            info["nacon_frac"] = jnp.asarray(imp.nacon).reshape(-1)[0] / (N * self.contacts_per_env)
            info["nefc_max"] = jnp.asarray(imp.nefc).reshape(-1).max()
        info["qpos_next"], info["qvel_next"] = d.qpos, d.qvel  # pre-autoreset state (checks only; DCE'd in training)
        info["final_obs"] = obs
        done = info["done"]
        # ---- autoreset of done envs: keep DR, goal and the env's step counter (as on CPU)
        steps_done = s.steps_done + 1
        fs = jax.vmap(self._fresh_state)(kr, dr, goal, steps_done)
        rdata = self._set_state(d, fs["qpos"], fs["qvel"])
        models2 = jax.vmap(self.model_for)(dr)
        rdata = jax.vmap(mjx.forward, in_axes=(0, 0))(models2, rdata)
        rfoot, _, _ = self.contact_masks(rdata, N)
        okeys = jax.vmap(lambda k: jax.random.split(k)[0])(fs["key"])
        robs = jax.vmap(self._observation)(okeys, rdata, dr, goal, fs["current_action"], fs["timers"], rfoot)
        w = lambda a, b: jnp.where(done.reshape(done.shape + (1,) * (a.ndim - 1)), a, b)
        new_data = merge_data(done, rdata, d, self.impl)
        stepped = dict(goal=goal, action_hist=hist, current_action=action, last_action=action,
                       prev_joint_vel=d.qvel[:, self.qvel_j], timers=timers, episode_step=s.episode_step + 1,
                       steps_done=steps_done, sum_track=sum_track, episode_return=ep_ret, key=key)
        fs.pop("qpos"); fs.pop("qvel")
        merged = {k: w(fs[k], stepped[k]) for k in stepped}
        merged["key"] = key
        new_obs = jax.tree_util.tree_map(w, robs, obs)
        return EnvState(data=new_data, dr=dr, obs=new_obs, **merged), info
