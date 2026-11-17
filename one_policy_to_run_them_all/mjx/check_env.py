"""Checks: JAX descriptions == CPU limb_context descriptions at default DR values (all arms),
and a few vmapped env steps run.   python check_env.py [impl] [robots...]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

impl = sys.argv[1] if len(sys.argv) > 1 else "jax"
robots = sys.argv[2:] or ["unitree_go1", "unitree_h1", "hexapod"]
for r in robots:
    sp = load_spec(r)
    for arm in ("urma", "dk", "k", "d", "empty"):
        env = MJXRobotEnv(sp, context_arm=arm, impl=impl) if arm == "urma" else env.__class__(sp, context_arm=arm, impl=impl)
        jd, fd, _ = env.descriptions(env.init_dr)
        cj = np.asarray(sp.cfg["cpu_descriptions"][arm]["joints"]); cf = np.asarray(sp.cfg["cpu_descriptions"][arm]["feet"])
        ej, ef = np.abs(np.asarray(jd) - cj).max(), np.abs(np.asarray(fd) - cf).max()
        assert ej < 1e-5 and ef < 1e-5, (r, arm, ej, ef)
    print(r, "descriptions match CPU for all arms", flush=True)
    B = 4
    env = MJXRobotEnv(sp, context_arm="dk", impl=impl, nr_envs_per_type=B)
    t = time.time()
    st = jax.jit(lambda k: env.batch_reset(k, B))(jax.random.PRNGKey(0))
    step = jax.jit(env.batch_step)
    acts = jnp.zeros((B, env.nu))
    for i in range(30):
        st, info = step(st, acts)
    jax.block_until_ready(st.data.qpos)
    print(r, "30 steps ok", "t=%.1fs" % (time.time() - t), "height", np.asarray(st.data.qpos[:, 2]).round(3),
          "rew", np.asarray(info["reward"]).round(4), "done", np.asarray(info["done"]), "foot obs", np.asarray(st.obs["fs"][0, :, 0]), flush=True)
    for k_, v_ in st.obs.items():
        assert np.all(np.isfinite(np.asarray(v_))), (r, k_)
    keys = jax.random.split(jax.random.PRNGKey(1), B)
    dr2 = jax.vmap(env._sample_dr)(keys, st.dr)
    jd2 = jax.vmap(lambda d: env.descriptions(d)[0])(dr2)
    tl = env.layout.field_slice("gear")
    assert not np.allclose(np.asarray(jd2[0, :, tl]), np.asarray(jd2[1, :, tl])), "D not per-env"
    print(r, "per-env D follows DR ok", flush=True)
