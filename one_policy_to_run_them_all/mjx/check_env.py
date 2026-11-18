"""Checks: JAX descriptions == CPU limb_context descriptions at default DR values (all arms),
and a few vmapped env steps run.   python check_env.py [impl] [robots...]"""
import sys, time
import numpy as np
import jax, jax.numpy as jnp
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.context import limb_context as lc

LATENTS_DUMMY = "/network/scratch/c/charlie.gauthier/urma_context/latents_dummy"  # the latents extract_robot_config.py used

impl = sys.argv[1] if len(sys.argv) > 1 else "jax"
robots = sys.argv[2:] or ["unitree_go1", "unitree_h1", "hexapod"]
for r in robots:
    sp = load_spec(r)
    arms = ("urma",) + tuple(a for a in sp.cfg["cpu_descriptions"] if a != "urma")  # urma, dk, k, d, empty + dummy-latent arms
    stock_j = stock_f = None
    for arm in arms:
        env = MJXRobotEnv(sp, context_arm=arm, impl=impl, latent_dir=LATENTS_DUMMY if lc.is_latent_arm(arm) else "")
        jd, fd, _ = env.descriptions(env.init_dr)
        jd, fd = np.asarray(jd), np.asarray(fd)
        cj = np.asarray(sp.cfg["cpu_descriptions"][arm]["joints"]); cf = np.asarray(sp.cfg["cpu_descriptions"][arm]["feet"])
        assert jd.shape == cj.shape and fd.shape == cf.shape, (r, arm, jd.shape, cj.shape)
        ej, ef = np.abs(jd - cj).max(), np.abs(fd - cf).max()
        assert ej < 1e-5 and ef < 1e-5, (r, arm, ej, ef)
        if arm == "urma":
            stock_j, stock_f = jd, fd
        elif arm == "empty":
            assert not np.any(jd) and not np.any(fd), (r, "empty not all zero")
        else:  # stock_superset_v2: GPU STOCK prefix == GPU urma token exactly
            assert np.array_equal(jd[:, :lc.STOCK_WIDTH], stock_j) and np.array_equal(fd, stock_f), (r, arm, "STOCK prefix")
    print(r, "descriptions match CPU for all arms:", " ".join(f"{a}" for a in arms), flush=True)
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
    tl = env.layout.field_slice("urma_torque_limit")
    assert not np.allclose(np.asarray(jd2[0, :, tl]), np.asarray(jd2[1, :, tl])), "D not per-env"
    print(r, "per-env D follows DR ok", flush=True)
