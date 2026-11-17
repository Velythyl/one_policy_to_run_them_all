"""Warp divergence rate vs tangential friction: run the training env (full DR, Gaussian actions,
std 1) with the ground + feet tangential friction forced to each value of a grid.
    python sweep_friction.py <out_json> <robot> [robot ...]     (env: N=1024 S=200 GRID=0.001,0.01,...)"""
import json
import os
import sys
from functools import partial
from pathlib import Path

import jax
import numpy as np

from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.mjx.robot_spec import load_spec

out = Path(sys.argv[1])
N, S = int(os.environ.get("N", 1024)), int(os.environ.get("S", 200))
GRID = [float(x) for x in os.environ.get("GRID", "0.001,0.01,0.03,0.05,0.07,0.1,0.15,0.3,1.0").split(",")]
res = json.loads(out.read_text()) if out.is_file() else {}
res_all = res
res = res_all.setdefault(os.environ.get("TAG", "base"), {})
for robot in sys.argv[2:]:
    env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=N, solimp_dmax_clamp=float(os.environ.get("CLAMP", 0.99)),
                      physics_overrides=os.environ.get("PHYS", ""))
    import time
    orig = env.model_for
    geoms = np.unique(np.r_[0, np.asarray(env.feet_fric).ravel()]) if env.model_struct["friction_mode"] == "ground_feet" else np.arange(env.mj_model.ngeom)
    res[robot] = {}

    def patched(mu):
        def model_for(dr):  # force the DR tangential friction (so DR-dependent options see it too)
            dr = dr.replace(friction_ground=dr.friction_ground.at[0].set(mu), friction_feet=dr.friction_feet.at[..., 0].set(mu))
            return orig(dr)
        return model_for

    def step_mu(s, a, mu):
        env.model_for = patched(mu)
        return env.batch_step(s, a)

    def reset_mu(k, mu):
        env.model_for = patched(mu)
        return env.batch_reset(k, nr_envs=N)
    step, reset = jax.jit(step_mu), jax.jit(reset_mu)  # mu traced: one compile per robot
    for mu in GRID:
        m = jax.numpy.float32(mu)
        s = reset(jax.random.PRNGKey(0), m)
        key = jax.random.PRNGKey(1)
        n = 0
        jax.block_until_ready(s)
        t0 = time.time()
        for t in range(S):
            key, ka = jax.random.split(key)
            s, info = step(s, jax.random.normal(ka, (N, s.action_hist.shape[-1])), m)
            n += int(info["sim_unstable"].sum())
        sps = N * S / (time.time() - t0)
        res[robot][str(mu)] = n / (N * S)
        print(os.environ.get("TAG", ""), robot, mu, n, round(n / (N * S), 5), "env_sps", round(sps), flush=True)
    env.model_for = orig
    out.write_text(json.dumps(res_all, indent=1))
