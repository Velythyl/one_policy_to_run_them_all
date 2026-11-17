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

    def leaf(path, b, p):
        name = tree_path_to_attr_str(path)
        if name in take_from_pre and hasattr(p, "ndim") and p.ndim > 0 and p.shape[0] == N:
            return p[i:i + 1]
        return b
    d = jax.tree_util.tree_map_with_path(leaf, base, pre_d)
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
        if r["full"][0] and not r["fresh"][0]:
            # one field at a time from the batch (on top of fresh) -> which stale field reproduces it
            r["culprits"] = [f for f in fields if f not in ("qpos", "qvel", "ctrl", "qacc_warmstart")
                             and bad(*build(s_pre.data, i, ctrl[i], {f}))[0]]
        print(json.dumps(r), flush=True)
        results.append(r)
    if len(results) >= K:
        break
print("per-world fields:", fields, flush=True)
