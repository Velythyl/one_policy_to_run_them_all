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
import mujoco
import numpy as np
from flax import struct
from mujoco import mjx

from one_policy_to_run_them_all.context import limb_context as lc
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
        self.mj_model = m
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
        self.nsub = c["nr_substeps"]
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
        self.feet_dr = np.asarray(sp.struct["dr_foot_geoms"], dtype=int)
        # contact group masks over geoms
        ng = m.ngeom
        def mask(ids):
            mk = np.zeros(ng, dtype=bool); mk[list(ids)] = True; return mk
        self.floor_mask = mask(c["collision_groups"]["floor"])
        self.foot_masks = np.stack([mask(sp.group_geoms([g])) for g in sp.struct["foot_groups"]])
        self.rc_geoms = np.asarray(sp.group_geoms(sp.struct["reward_collision_geoms"]), dtype=int)
        self.rc_trunk_mask = mask(sp.group_geoms(sp.struct["reward_trunk_groups"])) if sp.struct["reward_trunk_groups"] else np.zeros(ng, bool)
        term = sp.struct["termination"]
        self.term_kind = term[0]
        if term[0] == "contact":
            self.term_mask = mask(sp.group_geoms(term[1]))
        else:
            self.term_height = float(term[1])
        self.sym_pairs = np.asarray(sp.struct["symmetry_pairs"], dtype=int)
        self.nominal_diff_joints = np.asarray(sp.struct["nominal_diff_joints"], dtype=int)
        # context: static parts at the nominal pose (see extract_robot_config.py)
        ctx = c["context"]
        self.ctx_fields = {k: f64(v) for k, v in ctx["fields"].items()}
        self.ctx_feet = f64([ctx["feet"][fn] for fn in c["foot_names"]])
        self.robot_dims = f64(ctx["robot_dimensions"])
        self.robot_height = float(self.robot_dims[2])
        self.limb_bodies = np.asarray(ctx["limb_bodies"], dtype=int)
        self.latents = None
        if self.context_arm in lc.LATENT_ARMS:
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
            body_ipos_trunk=j(self.def_trunk_com), foot_size=j(self.def_geom_size[self.feet_dr, 0]),
            damping=j(self.def_damping), armature=j(self.def_armature), stiffness=j(self.def_stiffness),
            frictionloss=j(self.def_friction), friction_ground=j(self.def_geom_friction[0]),
            friction_feet=j(self.def_geom_friction[self.feet_dr]), solref=j(self.def_solref[:, :2]),
            gravity_z=j(self.mj_model.opt.gravity[2]), delay_mixed=jnp.array(False),
        )

    def _sample_dr(self, key, old: DR):
        S, U, M, A = self.P["seen"], self.P["unseen"], self.P["model_dr"], self.P["action_delay"]
        ks = iter(jax.random.split(key, 40))
        u = lambda lo, hi, shape=(): jax.random.uniform(next(ks), shape, minval=lo, maxval=hi)
        n, nu = self.n, self.nu
        # unseen_robot_functions/default.py
        trunk_mass_nf = u(1 - U["trunk_mass_factor"], 1 + U["trunk_mass_factor"])
        com_nf = u(1 - U["com_displacement_factor"], 1 + U["com_displacement_factor"])
        foot_nf = u(1 - U["foot_size_factor"], 1 + U["foot_size_factor"])
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
        foot_size = jnp.full(len(self.feet_dr), u(S["foot_size_min"], S["foot_size_max"]) * foot_nf ** self.spec.struct["foot_size_noise_power"])
        nominal = self.def_nominal + u(S["add_joint_nominal_position_min"], S["add_joint_nominal_position_max"], (n,))
        torque = self.def_torque * (1 + u(-S["torque_limit_factor"], S["torque_limit_factor"], self.def_torque.shape))
        vel = self.def_vel * (1 + u(-S["joint_velocity_factor"], S["joint_velocity_factor"], self.def_vel.shape))
        rng = self.def_range + u(S["add_joint_range_min"], S["add_joint_range_max"], self.def_range.shape)
        damping = jnp.where(stay, self.def_damping, u(S["joint_damping_min"], S["joint_damping_max"], self.def_damping.shape))
        # upstream typo: the stay-at-default branch sets `seen_armature`, so armature keeps its previous sample
        armature = jnp.where(stay, old.seen_armature, u(S["joint_armature_min"], S["joint_armature_max"], self.def_armature.shape))
        stiffness = jnp.where(stay, self.def_stiffness, u(S["joint_stiffness_min"], S["joint_stiffness_max"], self.def_stiffness.shape))
        friction = jnp.where(stay, self.def_friction, u(S["joint_friction_loss_min"], S["joint_friction_loss_max"], self.def_friction.shape))
        add_p = u(S["add_p_gain_min"], S["add_p_gain_max"])
        add_d = u(S["add_d_gain_min"], S["add_d_gain_max"])
        add_s = u(S["add_scaling_factor_min"], S["add_scaling_factor_max"])
        # mujoco_model_functions/default.py
        it = u(0.0, 1.0)
        lerp = lambda lo, hi, t: lo + (hi - lo) * t
        tang = lerp(M["friction_tangential_min"], M["friction_tangential_max"], it)
        fg = jnp.stack([tang, lerp(M["friction_torsional_ground_min"], M["friction_torsional_ground_max"], it),
                        lerp(M["friction_rolling_ground_min"], M["friction_rolling_ground_max"], it)])
        ff = jnp.stack([tang, lerp(M["friction_torsional_feet_min"], M["friction_torsional_feet_max"], it),
                        lerp(M["friction_rolling_feet_min"], M["friction_rolling_feet_max"], it)])
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
        geom_size = mx.geom_size.at[self.feet_dr, 0].set(dr.foot_size)
        geom_friction = mx.geom_friction.at[0].set(dr.friction_ground).at[self.feet_dr].set(dr.friction_feet)
        geom_solref = mx.geom_solref.at[:, :2].set(dr.solref)
        ctrlrange = jnp.stack([-dr.seen_torque, dr.seen_torque], axis=-1)
        jnt_range = mx.jnt_range.at[1:].set(dr.seen_range)
        dof_damping = mx.dof_damping.at[6:].set(dr.damping)
        dof_armature = mx.dof_armature.at[6:].set(dr.armature)
        dof_frictionloss = mx.dof_frictionloss.at[6:].set(dr.frictionloss)
        jnt_stiffness = mx.jnt_stiffness.at[1:].set(dr.stiffness)
        opt = mx.opt.replace(gravity=mx.opt.gravity.at[2].set(dr.gravity_z))
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
        f["joint_range"] = dr.seen_range[self.jnt_idx] / 4.6
        f["nominal_pos"] = col(dr.seen_nominal / 4.6)
        f["torque_limit"] = col(dr.seen_torque / (1000.0 / 2) - 1.0)
        f["max_joint_vel"] = col(dr.seen_vel / (35.0 / 2) - 1.0)
        f["damping"] = col(dr.seen_damping[self.dof_idx] / (10.0 / 2) - 1.0)
        f["armature"] = col(dr.seen_armature[self.dof_idx] / (0.2 / 2) - 1.0)
        f["stiffness"] = col(dr.seen_stiffness[self.jnt_idx] / (30.0 / 2) - 1.0)
        f["frictionloss"] = col(dr.seen_friction[self.dof_idx] / (1.2 / 2) - 1.0)
        # link mass / inertia: only the trunk mass is randomised, and the trunk is never a limb
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
        else:
            term = d.qpos[:, 2] < self.term_height
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
        qpos = jnp.asarray(self.mj_model.qpos0, jnp.float32)
        qpos = qpos.at[0:3].set(jnp.array([0.0, 0.0, self.spec.cfg["initial_drop_height"]])).at[3:7].set(quat)
        qpos = qpos.at[self.qpos_j].set(jpos)
        qvel = jnp.zeros(self.mj_model.nv).at[0:3].set(lin).at[3:6].set(ang).at[self.qvel_j].set(jvel)
        return qpos, qvel

    def make_batch_data(self, nr_envs, contacts_per_env=64, constraints_per_env=None):
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
        fc = foot_contact.astype(jnp.float32)
        ft = timers
        jmask = jax.random.uniform(k[4], (self.n,)) < D["dynamic_dropout_chance"]
        fmask = jax.random.uniform(k[5], (self.nf,)) < D["dynamic_dropout_chance"]
        mv = 0.0  # missing_value
        jp, jv, pa = (jnp.where(jmask, mv, x) for x in (jp, jv, current_action))
        fc, ft = (jnp.where(fmask, mv, x) for x in (fc, ft))
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
        cur = jnp.minimum(type_steps / R["curriculum_steps"], 1.0)
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
        r_acc = cur * R["joint_acceleration_coeff"] * -jnp.sum((prev_vel - d.qvel[6:] / self.dt) ** 2)
        r_tq = cur * R["joint_torque_coeff"] * -jnp.sum(d.qfrc_actuator[6:] ** 2)
        r_ar = cur * R["action_rate_coeff"] * -jnp.sum((action - last_action) ** 2)
        r_col = cur * R["collision_coeff"] * -nr_coll
        r_h = cur * R["base_height_coeff"] * -((d.qpos[2] - R["nominal_trunk_z"]) ** 2)
        air = jnp.sum(jnp.where(foot_contact, timers - R["air_time_max"], 0.0))
        r_air = cur * R["air_time_coeff"] * air
        viol = jnp.sum(~foot_contact[self.sym_pairs[:, 0]] & ~foot_contact[self.sym_pairs[:, 1]])
        r_sym = cur * R["symmetry_air_coeff"] * -viol
        reward = track_xy + track_yaw + r_z + r_ang + r_pos + r_nom + r_lim + r_vel + r_acc + r_tq + r_ar + r_col + r_h + r_air + r_sym
        reward = jnp.maximum(reward, 0.0)
        curv = jnp.array([local_lin[0], local_lin[1], ang[2]])
        perc = jnp.mean(1 - jnp.abs(curv - goal) / jnp.abs(goal))
        perc = jnp.where(jnp.isfinite(perc), jnp.maximum(perc, 0.0), 0.0)
        return reward, perc, jnp.sum((curv - goal) ** 2)


    # ------------------------------------------------------------------ batched API
    def _fresh_state(self, key, dr, goal, steps_done):
        k1, k2 = jax.random.split(key)
        qpos, qvel = self._initial_qpos_qvel(k1, dr)
        A = int(self.P["action_delay"]["max_nr_delay_steps"])
        zu = jnp.zeros(self.nu)
        return dict(qpos=qpos, qvel=qvel, goal=goal, action_hist=jnp.zeros((A + 1, self.nu)), current_action=zu,
                    last_action=zu, prev_joint_vel=jnp.zeros(self.mj_model.nv - 6), timers=jnp.zeros(self.nf),
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
            reward, perc, track_err = self._reward(dd, dr1, goal1, action1, s1.last_action, s1.prev_joint_vel, s1.timers, foot1, nrc1, type_steps)
            timers = jnp.where(foot1, 0.0, s1.timers + self.dt)
            done = term1 | trunc
            sum_track = s1.sum_track + perc
            ep_ret = s1.episode_return + reward
            info = {"reward": reward, "terminated": term1, "truncated": trunc, "done": done,
                    "episode_return": ep_ret, "episode_length": s1.episode_step + 1,
                    "eps_track_perf_perc": sum_track / self.horizon, "track_perf_perc": perc, "track_err_sq": track_err}
            return obs, timers, sum_track, ep_ret, info

        obs, timers, sum_track, ep_ret, info = jax.vmap(post)(d, dr, goal, action, s, ko, foot, nr_coll, term_c | unstable)
        info["reward"] = jnp.where(unstable, 0.0, info["reward"])
        info["episode_return"] = jnp.where(unstable, s.episode_return, info["episode_return"])
        ep_ret = info["episode_return"]
        info["sim_unstable"] = unstable
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
                       prev_joint_vel=d.qvel[:, 6:], timers=timers, episode_step=s.episode_step + 1,
                       steps_done=steps_done, sum_track=sum_track, episode_return=ep_ret, key=key)
        fs.pop("qpos"); fs.pop("qvel")
        merged = {k: w(fs[k], stepped[k]) for k in stepped}
        merged["key"] = key
        new_obs = jax.tree_util.tree_map(w, robs, obs)
        return EnvState(data=new_data, dr=dr, obs=new_obs, **merged), info
