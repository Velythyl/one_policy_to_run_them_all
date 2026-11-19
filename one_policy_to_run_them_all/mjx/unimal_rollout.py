"""Roll out a trained checkpoint (deterministic policy mean) on one robot with a fixed command and save env 0's qpos
for rendering (unimal_render.py replay). Also reports the mean tracked velocity over N envs.
    python unimal_rollout.py <checkpoint.pkl> <robot> <out.npz> [vx vy wz] [T=300] [N=64]
"""
import pickle
import sys
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.train_mjx import Policy

ck, robot, out = sys.argv[1], sys.argv[2], sys.argv[3]
cmd = jnp.array([float(x) for x in sys.argv[4:7]]) if len(sys.argv) > 6 else jnp.array([0.5, 0.0, 0.0])
T = int(sys.argv[7]) if len(sys.argv) > 7 else 300
N = int(sys.argv[8]) if len(sys.argv) > 8 else 64
with open(ck, "rb") as f:
    c = pickle.load(f)
a = c["args"]
env = MJXRobotEnv(load_spec(robot), context_arm=a["context_arm"], latent_dir=a["latent_dir"], impl="warp", nr_envs_per_type=N,
                  eval_mode=True, solimp_dmax_clamp=a["solimp_dmax_clamp"], friction_tangential_floor=a["friction_tangential_floor"],
                  physics_overrides=a["physics_overrides"])
policy = Policy(1.0, 1.0, 0.015, 1e-8, 10.0, 1e-8, 2.0)
s = jax.jit(partial(env.batch_reset, nr_envs=N))(jax.random.PRNGKey(0))
s = s.replace(goal=jnp.broadcast_to(cmd, s.goal.shape), steps_done=jnp.full_like(s.steps_done, 10 ** 7))
# command resampling (prob 0.002 per step) is left on, as in training; env 0's command is re-pinned every step below


@jax.jit
def step(s):
    o = s.obs
    mean, _ = policy.apply(c["pparams"], o["jd"], o["js"], o["fd"], o["fs"], o["gp"])
    s2, info = env.batch_step(s, mean)
    return s2.replace(goal=jnp.broadcast_to(cmd, s2.goal.shape)), info


qs, vel, done = [], [], []
for t in range(T):
    s, info = step(s)
    qs.append(np.asarray(info["qpos_next"][0]))
    q = np.asarray(info["qpos_next"]); v = np.asarray(info["qvel_next"])
    w, x, y, z = q[:, 3], q[:, 4], q[:, 5], q[:, 6]
    R = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                  np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                  np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)
    loc = np.einsum("nji,nj->ni", R, v[:, 0:3])
    vel.append(np.concatenate([loc[:, :2], v[:, 5:6]], 1).mean(0))
    done.append(float(np.asarray(info["done"]).mean()))
np.savez(out, qpos=np.stack(qs))
vel = np.stack(vel)
print(robot, "cmd", np.asarray(cmd).tolist(), "mean local vel (vx, vy, wz) over last 200 steps:", vel[-200:].mean(0).round(3).tolist(),
      "done rate/step", round(float(np.mean(done)), 4), flush=True)
