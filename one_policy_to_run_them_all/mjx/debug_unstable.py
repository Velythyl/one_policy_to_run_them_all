"""Catch worlds that diverge in Warp, replay the same pre-step state + ctrl in MuJoCo C (float64)
and with Warp single-world, and report contacts. python debug_unstable.py <robot> <B> <steps>"""
import sys, json
import numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

robot, B, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
env = MJXRobotEnv(load_spec(robot), impl="warp", nr_envs_per_type=B)
reset = jax.jit(lambda k: env.batch_reset(k, B)); step = jax.jit(env.batch_step)
st = reset(jax.random.PRNGKey(0)); key = jax.random.PRNGKey(3)
m = env.mj_model; found = 0
for i in range(S):
    key, k = jax.random.split(key)
    prev = st
    st, info = step(st, jax.random.normal(k, (B, env.nu)))
    bad = np.where(np.asarray(info["sim_unstable"]))[0]
    for w in bad[:3]:
        found += 1
        q, v = np.asarray(prev.data.qpos[w], np.float64), np.asarray(prev.data.qvel[w], np.float64)
        dr = jax.tree_util.tree_map(lambda x: np.asarray(x[w]), prev.dr)
        # replay in C with the same DR model fields and the same torque (recomputed from the delayed action)
        mc = mujoco.MjModel.from_xml_path(str(env.spec.xml_path)); mc.opt.timestep = m.opt.timestep
        full = env._model_for_all(prev.dr.__class__(**{f: jnp.asarray(getattr(dr, f)) for f in dr.__dataclass_fields__}))
        for f in ("body_mass", "body_inertia", "body_ipos", "geom_size", "geom_friction", "geom_solref", "actuator_ctrlrange",
                  "jnt_range", "dof_damping", "dof_armature", "dof_frictionloss", "jnt_stiffness"):
            getattr(mc, f)[:] = np.asarray(getattr(full, f))
        mc.opt.gravity[:] = np.asarray(full.opt.gravity)
        d = mujoco.MjData(mc); d.qpos[:] = q; d.qvel[:] = v
        d.ctrl[:] = np.asarray(env._torques(jax.tree_util.tree_map(lambda x: x[w], prev.data), jax.tree_util.tree_map(lambda x: jnp.asarray(x), dr), jnp.asarray(prev.current_action[w])))
        mujoco.mj_forward(mc, d)
        cons = sorted({(mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_GEOM, c.geom1), mujoco.mj_id2name(mc, mujoco.mjtObj.mjOBJ_GEOM, c.geom2)) for c in d.contact[:d.ncon]})
        for _ in range(env.nsub):
            mujoco.mj_step(mc, d)
        cw = {k_: int(d.warning[getattr(mujoco.mjtWarning, k_)].number) for k_ in ("mjWARN_BADQACC", "mjWARN_BADQVEL", "mjWARN_BADQPOS")}
        print(json.dumps({"step": i, "world": int(w), "pre_z": float(q[2]), "pre_max_abs_qvel": float(np.abs(v).max()),
                          "C_after_finite": bool(np.isfinite(d.qpos).all()), "C_max_abs_qvel_after": float(np.abs(d.qvel).max()),
                          "C_warnings": cw, "contacts_pre": cons[:12]}), flush=True)
    if found >= 6:
        break
print("done; unstable found", found)
