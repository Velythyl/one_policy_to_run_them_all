"""Probe MJX impl='warp' semantics: data batching, contact buffer layout, batched-model DR, and
agreement with MuJoCo C for a few hundred steps. python debug_warp.py <robot> <B>"""
import sys, json, time
import numpy as np, jax, jax.numpy as jnp, mujoco
from mujoco import mjx
from one_policy_to_run_them_all.mjx.robot_spec import load_spec

robot, B = sys.argv[1], int(sys.argv[2])
sp = load_spec(robot); c = sp.cfg
m = mujoco.MjModel.from_xml_path(str(sp.xml_path)); m.opt.timestep = c["timestep"]
print("opt: cone", m.opt.cone, "solver", m.opt.solver, "iters", m.opt.iterations, "ls", m.opt.ls_iterations,
      "integrator", m.opt.integrator, "jacobian", m.opt.jacobian, "ngeom", m.ngeom, "margin max", m.geom_margin.max(),
      "condim set", sorted(set(m.geom_condim.tolist())))
qpos = np.array(m.qpos0); qpos[2] = c["initial_drop_height"]
qi = np.array(c["qpos_joint_idx"], int); qpos[qi] = c["nominal_joint_positions"]
mx = mjx.put_model(m, impl="warp")
K = 64
make = lambda k: mjx.make_data(m, impl="warp", naconmax=B * K, njmax=K * 8).replace(qpos=jnp.asarray(qpos, jnp.float32))
ds = jax.vmap(make)(jnp.arange(B))
print("batched qpos", ds.qpos.shape)
for name in ("contact__geom", "contact__dist", "contact__worldid", "contact__includemargin", "nacon"):
    x = getattr(ds._impl, name, None)
    print(name, None if x is None else (x.shape, x.dtype))
print("public nacon", getattr(ds, "nacon", None) is not None, [f for f in ("ncon", "nacon", "ne", "nefc") if hasattr(ds, f)])
step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
tau = np.zeros(m.nu)
nominal = np.array(c["nominal_joint_positions"]); kp, kd = c["params"]["control"]["p_gain"], c["params"]["control"]["d_gain"]
d = mujoco.MjData(m); d.qpos[:] = qpos; mujoco.mj_forward(m, d)
t = time.time()
for i in range(400):
    # same PD on both: per-world torque from world 0 state (all worlds identical)
    q = np.asarray(ds.qpos[0]); v = np.asarray(ds.qvel[0])
    tq = kp * (nominal - q[qi]) - kd * v[np.array(c["qvel_joint_idx"], int)]
    ds = step(mx, ds.replace(ctrl=jnp.broadcast_to(jnp.asarray(tq, jnp.float32), (B, m.nu))))
    tc = kp * (nominal - d.qpos[qi]) - kd * d.qvel[np.array(c["qvel_joint_idx"], int)]
    d.ctrl[:] = tc; mujoco.mj_step(m, d)
    if i in (0, 50, 100, 200, 399):
        print(f"step {i}: warp z={float(ds.qpos[0,2]):.4f} (all worlds equal: {bool(jnp.allclose(ds.qpos[0], ds.qpos[-1]))}) "
              f"C z={d.qpos[2]:.4f} |dq|max={np.abs(np.asarray(ds.qpos[0]) - d.qpos).max():.2e} ncon_C={d.ncon} "
              f"nacon={np.asarray(ds._impl.nacon).ravel()[:3]}", flush=True)
g = np.asarray(ds._impl.contact__geom); w = np.asarray(ds._impl.contact__worldid); dist = np.asarray(ds._impl.contact__dist)
print("contact buffer shapes", g.shape, w.shape, dist.shape)
n0 = int(np.asarray(ds._impl.nacon).ravel()[0])
print("first contacts (geom, world, dist):", [(tuple(g.reshape(-1, 2)[i]), int(w.ravel()[i]), float(dist.ravel()[i])) for i in range(min(n0, 8))])
print("C contacts:", [(d.contact[i].geom1, d.contact[i].geom2, round(d.contact[i].dist, 5)) for i in range(d.ncon)])
# batched model (DR) through vmap in_axes=0
mb = jax.vmap(lambda s: mx.replace(body_mass=mx.body_mass.at[1].set(mx.body_mass[1] * s),
                                   geom_friction=mx.geom_friction.at[0, 0].set(s)))(jnp.linspace(0.5, 1.5, B))
step_b = jax.jit(jax.vmap(mjx.step, in_axes=(0, 0)))
ds2 = step_b(mb, ds)
print("batched-model step ok; z spread", float(ds2.qpos[:, 2].max() - ds2.qpos[:, 2].min()))
