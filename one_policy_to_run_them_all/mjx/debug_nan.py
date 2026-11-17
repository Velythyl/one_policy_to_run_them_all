"""Find the NaN source with impl=warp: step B envs with zero actions under different sets of
DR-batched model fields.  python debug_nan.py <robot> <B>"""
import sys
import numpy as np, jax, jax.numpy as jnp
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

robot, B = sys.argv[1], int(sys.argv[2])
env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=B)
ALL = env.DR_MODEL_FIELDS
trials = [("none", ())] + [(f, (f,)) for f in ALL] + [("all", ALL)]
for name, fields in trials:
    env.dr_model_fields = fields
    st = jax.jit(lambda k: env.batch_reset(k, B))(jax.random.PRNGKey(0))
    ok0 = {k: bool(jnp.all(jnp.isfinite(v))) for k, v in st.obs.items()}
    step = jax.jit(env.batch_step)
    first_bad = None
    for i in range(30):
        st, info = step(st, jnp.zeros((B, env.nu)))
        if first_bad is None and not bool(jnp.all(jnp.isfinite(st.data.qpos))):
            first_bad = i
    print(f"{name:20s} reset_obs_finite={all(ok0.values())} {[k for k,v in ok0.items() if not v]} first_nan_step={first_bad} "
          f"z={np.asarray(st.data.qpos[:4, 2]).round(3)} rew_finite={bool(jnp.all(jnp.isfinite(info['reward'])))}", flush=True)
