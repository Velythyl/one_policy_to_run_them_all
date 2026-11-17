"""Compare per-world contact pair sets of Warp (global buffer) vs MuJoCo C mj_forward at the same
qpos, and the resulting termination decision.  python debug_contacts.py <robot> <B> <steps>"""
import sys, collections
import numpy as np, jax, jax.numpy as jnp, mujoco
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

robot, B, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=B)
st = jax.jit(lambda k: env.batch_reset(k, B))(jax.random.PRNGKey(0)); step = jax.jit(env.batch_step)
key = jax.random.PRNGKey(5)
only_w, only_c = collections.Counter(), collections.Counter()
term_mismatch = 0; checked = 0
name = lambda g: mujoco.mj_id2name(env.mj_model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"g{int(g)}"
for i in range(S):
    key, k = jax.random.split(key)
    st, info = step(st, 0.3 * jax.random.normal(k, (B, env.nu)))
    if i % 10 != 9:
        continue
    imp = st.data._impl
    n = int(np.asarray(imp.nacon).ravel()[0])
    g = np.asarray(imp.contact__geom).reshape(-1, 2)[:n]; w = np.asarray(imp.contact__worldid).ravel()[:n]
    dist = np.asarray(imp.contact__dist).ravel()[:n]; mg = np.asarray(imp.contact__includemargin).ravel()[:n]
    act = dist < mg
    warp_pairs = collections.defaultdict(set)
    for (a, b), ww, ok in zip(g, w, act):
        if ok:
            warp_pairs[int(ww)].add(tuple(sorted((int(a), int(b)))))
    foot, nr, term = env.contact_masks(st.data, B)
    term = np.asarray(term)
    m = env.mj_model; d = mujoco.MjData(m)
    for wi in range(B):
        if bool(np.asarray(st.dr.delay_mixed[wi])) or True:
            pass
        d.qpos[:] = np.asarray(st.data.qpos[wi]); d.qvel[:] = np.asarray(st.data.qvel[wi]); mujoco.mj_forward(m, d)
        cp = {tuple(sorted((c.geom1, c.geom2))) for c in d.contact[:d.ncon] if c.dist < c.includemargin}
        for p in warp_pairs[wi] - cp: only_w[(name(p[0]), name(p[1]))] += 1
        for p in cp - warp_pairs[wi]: only_c[(name(p[0]), name(p[1]))] += 1
        if env.term_kind == "contact":
            tm = env.term_mask
            c_term = any(tm[a] or tm[b] for a, b in cp)
            term_mismatch += int(c_term != bool(term[wi])); checked += 1
print("worlds checked", checked, "termination mismatches", term_mismatch)
print("pairs only in warp (top):", only_w.most_common(10))
print("pairs only in C (top):", only_c.most_common(10))
