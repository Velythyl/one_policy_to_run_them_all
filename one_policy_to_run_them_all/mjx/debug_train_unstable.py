"""Training-env divergence probe: run MJXRobotEnv.batch_step (full DR, pushes, delays, solimp cap) with
Gaussian actions (mean 0, std SIGMA) and, for every world flagged sim_unstable, replay the exact
pre-step state (qpos, qvel, qacc_warmstart, ctrl, per-env DR model fields) in MuJoCo C (float64).
    python debug_train_unstable.py <robot> [N=1024] [S=600] [SIGMA=1.0] [OUT_JSON]
Env: CLAMP (solimp dmax cap, default 0.99)."""
import copy
import json
import os
import sys
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.mjx.robot_spec import load_spec

robot = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
S = int(sys.argv[3]) if len(sys.argv) > 3 else 600
SIGMA = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0
OUT = sys.argv[5] if len(sys.argv) > 5 else None
CLAMP = float(os.environ.get("CLAMP", 0.99))
env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=N, solimp_dmax_clamp=CLAMP)
step = jax.jit(env.batch_step)
s = jax.jit(partial(env.batch_reset, nr_envs=N))(jax.random.PRNGKey(0))
model_all = jax.jit(env._model_for_all)
act_dim = s.action_hist.shape[-1]
FIELDS = ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_friction", "geom_solref", "actuator_ctrlrange",
          "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness")


def c_replay(i, pre, ctrl):
    m = copy.deepcopy(env.mj_model)
    dr_i = jax.tree_util.tree_map(lambda x: x[i], pre.dr)
    full = model_all(dr_i)
    for f in FIELDS:
        getattr(m, f)[:] = np.asarray(getattr(full, f), np.float64).reshape(getattr(m, f).shape)
    m.opt.gravity[:] = np.asarray(full.opt.gravity, np.float64)
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(pre.data.qpos[i], np.float64)
    d.qvel[:] = np.asarray(pre.data.qvel[i], np.float64)
    d.qacc_warmstart[:] = np.asarray(pre.data.qacc_warmstart[i], np.float64)
    d.ctrl[:] = np.asarray(ctrl[i], np.float64)
    mujoco.mj_forward(m, d)
    ncon0 = int(d.ncon)
    maxv = 0.0
    warn = 0
    for _ in range(env.nsub):
        mujoco.mj_step(m, d)
        maxv = max(maxv, float(np.abs(d.qvel).max()))
    warn = int(sum(d.warning[k].number for k in range(len(d.warning))))
    bad = (not np.all(np.isfinite(d.qvel))) or maxv >= 1e4 or warn > 0
    pairs = sorted({tuple(sorted((int(c.geom1), int(c.geom2)))) for c in d.contact[:ncon0]}) if ncon0 else []
    return {"c_bad": bool(bad), "c_max_qvel": maxv, "c_warnings": warn, "ncon_pre": ncon0, "pairs_pre": pairs[:12],
            "pre_max_qvel": float(np.abs(pre.data.qvel[i]).max()), "trunk_z": float(pre.data.qpos[i][2]),
            "max_ctrl": float(np.abs(ctrl[i]).max()),
            "solref": np.asarray(pre.dr.solref[i]).tolist(), "episode_step": int(pre.episode_step[i])}


key = jax.random.PRNGKey(1)
events, n_unst, n_steps = [], 0, 0
for t in range(S):
    key, ka = jax.random.split(key)
    a = SIGMA * jax.random.normal(ka, (N, act_dim))
    pre = s
    s, info = step(s, a)
    u = np.asarray(info["sim_unstable"])
    n_steps += N
    if u.any():
        idx = np.nonzero(u)[0]
        n_unst += len(idx)
        ctrl = np.asarray(info["ctrl"])
        for i in idx[:4]:
            if len(events) < 60:
                ev = c_replay(int(i), pre, ctrl); ev["t"] = t; events.append(ev)
res = {"robot": robot, "N": N, "S": S, "sigma": SIGMA, "clamp": CLAMP, "unstable": n_unst, "rate": n_unst / n_steps,
       "replayed": len(events), "c_also_bad": sum(e["c_bad"] for e in events), "events": events}
print(json.dumps({k: v for k, v in res.items() if k != "events"}), flush=True)
for e in events[:15]:
    print(json.dumps(e), flush=True)
if OUT:
    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)
