"""Why do Warp worlds blow up on the first step after an autoreset? Runs the training env with
Gaussian actions, takes the first K worlds flagged sim_unstable at episode_step 0 and re-steps each
pre-state in a fresh 1-world Warp Data in three ways:
  full   = every per-world Data field copied from the training batch (should reproduce),
  fresh  = only qpos/qvel/ctrl/qacc_warmstart copied, then mjx.forward (clean solver/contact state),
  bisect = full, but with field groups taken from `fresh` -> names the stale field(s).
    python debug_reset_blowup.py <robot> [N=1024] [S=300] [K=8]"""
import json
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx
from mujoco.mjx._src.types import tree_path_to_attr_str
from mujoco.mjx.warp import types as mjxw_types

from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.mjx.robot_spec import load_spec

robot = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
S = int(sys.argv[3]) if len(sys.argv) > 3 else 300
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8
env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=N, solimp_dmax_clamp=0.99)
step = jax.jit(env.batch_step)
s = jax.jit(partial(env.batch_reset, nr_envs=N))(jax.random.PRNGKey(0))
act_dim = s.action_hist.shape[-1]
BD = mjxw_types._BATCH_DIM["Data"]
B1 = 1
d1_template = env.make_batch_data(B1)


@jax.jit
def phys(dr1, d):
    models = jax.vmap(env.model_for)(dr1)
    stepper = jax.vmap(mjx.step, in_axes=(0, 0))
    d = jax.lax.fori_loop(0, env.nsub, lambda i, dd: stepper(models, dd), d)
    return jnp.max(jnp.abs(d.qvel)), jnp.all(jnp.isfinite(d.qvel))


@jax.jit
def fwd(dr1, d):
    models = jax.vmap(env.model_for)(dr1)
    return jax.vmap(mjx.forward, in_axes=(0, 0))(models, d)


@partial(jax.jit, static_argnums=2)
def phys_nominal(dr1, d, field):
    """phys() with one DR'd model field reset to the nominal (un-randomised) model value."""
    models = jax.vmap(env.model_for)(dr1)
    if field == "gravity":
        models = models.replace(opt=models.opt.replace(gravity=jnp.broadcast_to(env.mx.opt.gravity, models.opt.gravity.shape)))
    else:
        nom = getattr(env.mx, field)
        models = models.replace(**{field: jnp.broadcast_to(nom, getattr(models, field).shape)})
    stepper = jax.vmap(mjx.step, in_axes=(0, 0))
    d = jax.lax.fori_loop(0, env.nsub, lambda i, dd: stepper(models, dd), d)
    return jnp.max(jnp.abs(d.qvel)), jnp.all(jnp.isfinite(d.qvel))


@partial(jax.jit, static_argnums=(2, 3))
def phys_fric(dr1, d, col, floor):
    """phys() with geom_friction column `col` (0 tangential, 1 torsional, 2 rolling) at the nominal
    model value (floor=None) or floored at `floor`."""
    models = jax.vmap(env.model_for)(dr1)
    gf = models.geom_friction
    if floor is None:
        gf = gf.at[..., col].set(jnp.broadcast_to(env.mx.geom_friction[:, col], gf[..., col].shape))
    else:
        gf = gf.at[..., col].set(jnp.maximum(gf[..., col], floor))
    models = models.replace(geom_friction=gf)
    stepper = jax.vmap(mjx.step, in_axes=(0, 0))
    d = jax.lax.fori_loop(0, env.nsub, lambda i, dd: stepper(models, dd), d)
    return jnp.max(jnp.abs(d.qvel)), jnp.all(jnp.isfinite(d.qvel))


FRIC_TESTS = [(0, None), (1, None), (2, None), (0, 0.1), (0, 0.2), (0, 0.3), (1, 1e-3), (1, 3e-3), (2, 1e-4), (2, 1e-3)]


DR_FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_friction", "geom_solref", "actuator_ctrlrange",
             "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness", "gravity")


def per_world_paths(d):
    out = []
    for path, x in jax.tree_util.tree_leaves_with_path(d):
        name = tree_path_to_attr_str(path)
        if BD.get(name, True) and hasattr(x, "ndim") and x.ndim > 0 and x.shape[0] == N:
            out.append(name)
    return out


def build(pre_d, i, ctrl_i, take_from_pre):
    """1-world Data: fields in take_from_pre copied from batch world i, the rest from a fresh forward."""
    dr1 = jax.tree_util.tree_map(lambda x: x[i:i + 1], s_pre.dr)
    base = d1_template.replace(qpos=pre_d.qpos[i:i + 1], qvel=pre_d.qvel[i:i + 1],
                               qacc_warmstart=pre_d.qacc_warmstart[i:i + 1])
    base = fwd(dr1, base)

    pre_leaves = {tree_path_to_attr_str(pth): x for pth, x in jax.tree_util.tree_leaves_with_path(pre_d)}
    bl, tdef = jax.tree_util.tree_flatten_with_path(base)
    new = []
    for pth, b in bl:
        name = tree_path_to_attr_str(pth)
        p = pre_leaves.get(name)
        if name in take_from_pre and p is not None and p.ndim > 0 and p.shape[0] == N and p.shape[1:] == b.shape[1:]:
            new.append(p[i:i + 1])
        else:
            new.append(b)
    d = jax.tree_util.tree_unflatten(tdef, new)
    d = d.replace(ctrl=jnp.asarray(ctrl_i)[None])
    return dr1, d


def bad(dr1, d):
    mv, fin = phys(dr1, d)
    return bool((~fin) | (mv >= 1e4)), float(mv)


key = jax.random.PRNGKey(1)
results = []
fields = None
for t in range(S):
    key, ka = jax.random.split(key)
    s_pre = s
    s, info = step(s, jax.random.normal(ka, (N, act_dim)))
    u = np.asarray(info["sim_unstable"]) & (np.asarray(s_pre.episode_step) == 0)
    if not u.any():
        continue
    fields = fields or per_world_paths(s_pre.data)
    ctrl = np.asarray(info["ctrl"])
    for i in np.nonzero(u)[0][:2]:
        i = int(i)
        r = {"t": t, "world": i}
        r["full"] = bad(*build(s_pre.data, i, ctrl[i], set(fields)))
        r["fresh"] = bad(*build(s_pre.data, i, ctrl[i], set()))
        if r["fresh"][0]:
            dr1, dfr = build(s_pre.data, i, ctrl[i], set())
            fixes = {}
            for f in DR_FIELDS:
                mv, fin = phys_nominal(dr1, dfr, f)
                fixes[f] = bool(fin & (mv < 1e4))
            r["fixed_by_nominal"] = [f for f, ok in fixes.items() if ok]
            fr = {}
            for col, fl in FRIC_TESTS:
                mv, fin = phys_fric(dr1, dfr, col, fl)
                fr[f"col{col}_{'nominal' if fl is None else fl}"] = bool(fin & (mv < 1e4))
            r["friction_fix"] = fr
            r["geom_friction_used"] = np.round(np.asarray(jax.vmap(env.model_for)(dr1).geom_friction[0]), 5)[
                np.unique(np.r_[0, np.asarray(env.feet_fric).ravel()]) if hasattr(env, "feet_fric") else slice(0, 3)].tolist()
            r["dr_unused"] = {k: np.round(np.asarray(v[i]), 4).tolist() for k, v in vars(s_pre.dr).items()
                       if k in ("solref", "friction_feet", "friction_ground", "damping", "armature", "frictionloss",
                                "stiffness", "body_mass_trunk", "foot_size", "gravity_z")} if hasattr(s_pre.dr, "__dict__") else {}
        if r["full"][0] and not r["fresh"][0]:
            # one field at a time from the batch (on top of fresh) -> which stale field reproduces it
            r["culprits"] = [f for f in fields if f not in ("qpos", "qvel", "ctrl", "qacc_warmstart")
                             and bad(*build(s_pre.data, i, ctrl[i], {f}))[0]]
        r.pop("dr_unused", None); print(json.dumps(r), flush=True)
        results.append(r)
    if len(results) >= K:
        break
print("per-world fields:", fields, flush=True)
