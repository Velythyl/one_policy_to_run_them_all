"""NaN hunt part 2: random actions (python loop) vs lax.scan rollout with impl=warp."""
import sys
import numpy as np, jax, jax.numpy as jnp
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

robot, B = sys.argv[1], int(sys.argv[2])
env = MJXRobotEnv(load_spec(robot), impl=sys.argv[3] if len(sys.argv) > 3 else "warp", nr_envs_per_type=B)
env.horizon = 7  # force autoresets
reset = jax.jit(lambda k: env.batch_reset(k, B))
step = jax.jit(env.batch_step)
fin = lambda t: all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(t) if jnp.issubdtype(x.dtype, jnp.floating))
st = reset(jax.random.PRNGKey(0))
key = jax.random.PRNGKey(1)
for i in range(60):
    key, k = jax.random.split(key)
    st, info = step(st, jax.random.normal(k, (B, env.nu)))
    if not fin(st.data.qpos) or not fin(st.obs) or not fin(info["reward"]):
        print("random-action loop: first non-finite at step", i, "qpos", fin(st.data.qpos), "obs", {k_: fin(v) for k_, v in st.obs.items()}, "rew", fin(info["reward"]), flush=True)
        break
else:
    print("random-action loop: 60 steps finite; dones", int(info["done"].sum()), flush=True)

@jax.jit
def roll(st, key):
    def body(c, _):
        st, key = c
        key, k = jax.random.split(key)
        st, info = env.batch_step(st, jax.random.normal(k, (B, env.nu)))
        return (st, key), (st.data.qpos[:, 2], info["reward"], info["final_obs"]["gc"])
    (st, key), out = jax.lax.scan(body, (st, key), None, length=30)
    return st, out
st = reset(jax.random.PRNGKey(0))
st2, (z, r, gc) = roll(st, jax.random.PRNGKey(1))
print("scan: z finite per step", np.asarray(jnp.all(jnp.isfinite(z), axis=1)).astype(int), "rew finite", bool(jnp.all(jnp.isfinite(r))), "gc finite", bool(jnp.all(jnp.isfinite(gc))), flush=True)
print("z first 5 steps env0", np.asarray(z[:5, 0]))
