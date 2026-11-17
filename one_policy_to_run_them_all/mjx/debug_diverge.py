"""Find what makes Warp worlds diverge (sanity_warp.py protocol): per world, the contacts / solver
state of the last finite step before the blow-up, plus a MuJoCo C replay of that pre-state.
    OPT_OVERRIDE='{"tolerance":1e-6}' python debug_diverge.py <cfg_dir> robot [B] [S]
"""
import collections
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
cfg_dir, robot = Path(sys.argv[1]), sys.argv[2]
B = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
S = int(sys.argv[4]) if len(sys.argv) > 4 else 300
c = json.loads((cfg_dir / f"{robot}.json").read_text())
_spec = mujoco.MjSpec.from_file(os.environ.get("XML", str(ENV_ROOT / robot / "data" / c["xml_file"])))
VARIANT = os.environ.get("VARIANT", "")
_m0 = _spec.compile()
for _i, _g in enumerate(_spec.geoms):
    _col = _m0.geom_contype[_i] or _m0.geom_conaffinity[_i]
    if "condim3" in VARIANT and _m0.geom_condim[_i] == 6:
        _g.condim = 3
    if "capsule" in VARIANT and _col and _m0.geom_type[_i] == mujoco.mjtGeom.mjGEOM_CYLINDER:
        _g.type = mujoco.mjtGeom.mjGEOM_CAPSULE
        _g.size = [_m0.geom_size[_i, 0], _m0.geom_size[_i, 1], 0]
    if "clampimp" in VARIANT and _col and _m0.geom_solimp[_i, 1] > 0.99:
        _si = _m0.geom_solimp[_i].copy(); _si[1] = 0.99; _si[0] = min(_si[0], 0.99); _g.solimp = _si
    if "solimp" in VARIANT and "clampimp" not in VARIANT and _col:
        _g.solimp = [0.9, 0.95, 0.001, 0.5, 2]
if "elliptic" in VARIANT:
    _spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
m = _spec.compile()
if os.environ.get("SOLREF"):  # DR writes geom_solref[:, :2] = (-stiffness, -damping) for every geom
    m.geom_solref[:, :2] = [float(x) for x in os.environ["SOLREF"].split(",")]
if "clampimp" in VARIANT:  # same as MJXRobotEnv.solimp_dmax_clamp (also dmin), applied to the compiled model
    m.geom_solimp[:, 1] = np.minimum(m.geom_solimp[:, 1], 0.99); m.geom_solimp[:, 0] = np.minimum(m.geom_solimp[:, 0], 0.99)
print("variant", VARIANT, "solref", os.environ.get("SOLREF", "xml"), flush=True)
m.opt.timestep = c["timestep"]
for k_, v_ in json.loads(os.environ.get("OPT_OVERRIDE", "{}")).items():
    setattr(m.opt, k_, v_)
P = c["params"]["control"]
qj = np.asarray(c["qpos_joint_idx"], int); vj = np.asarray(c["qvel_joint_idx"], int)
nom = np.asarray(c["nominal_joint_positions"])
base = np.asarray(c["complete_nominal_joint_position"]) if c["complete_nominal_joint_position"] else m.qpos0.copy()
base = base.copy(); base[2] = c["initial_drop_height"]; base[3:7] = [1, 0, 0, 0]; base[qj] = nom
rng = np.random.default_rng(0)
targets = nom + rng.normal(0, 0.3, (S, 1024, len(nom)))[:, :B]
q0 = np.tile(base, (1024, 1)); q0[:, qj] += rng.uniform(-0.1, 0.1, (1024, len(nom))); q0 = q0[:B]
nsub = c["nr_substeps"]
CPW = int(os.environ.get("CPW", 64))
mx = mjx.put_model(m, impl="warp")
dw = jax.vmap(lambda _: mjx.make_data(m, impl="warp", naconmax=B * CPW, njmax=CPW * 8))(jnp.arange(B))
dw = jax.vmap(mjx.forward, in_axes=(None, 0))(mx, dw.replace(qpos=jnp.asarray(q0, jnp.float32)))
kp, kd = P["p_gain"], P["d_gain"]


@jax.jit
def substep(dw):
    return jax.vmap(mjx.step, in_axes=(None, 0))(mx, dw)


gname = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or f"g{int(g)}"
gtype = lambda g: mujoco.mjtGeom(int(m.geom_type[g])).name[7:]
first_bad = {}
pair_at_blowup = collections.Counter()
type_at_blowup = collections.Counter()
niter_hist = []
for s in range(S):
    tau = kp * (targets[s] - np.asarray(dw.qpos)[:, qj]) - kd * np.asarray(dw.qvel)[:, vj]
    dw = dw.replace(ctrl=jnp.asarray(tau, jnp.float32))
    for k in range(nsub):
        prev = dw
        dw = substep(dw)
        qv = np.asarray(dw.qvel)
        bad = ~(np.isfinite(qv).all(-1) & (np.abs(qv) < 1e4).all(-1))
        imp = prev._impl
        _ni = getattr(dw._impl, "solver_niter", None)
        niter_hist.append(np.asarray(_ni).ravel() if _ni is not None else np.zeros(1))
        new = [w for w in np.nonzero(bad)[0] if w not in first_bad]
        if new:
            # contacts of the step that blew up are in dw (computed at the pre-step state = prev qpos)
            ii = dw._impl
            n = int(np.asarray(ii.nacon).ravel()[0])
            geom = np.asarray(ii.contact__geom).reshape(-1, 2)[:n]; wid = np.asarray(ii.contact__worldid).ravel()[:n]
            dist = np.asarray(ii.contact__dist).ravel()[:n]
            for w in new:
                pairs = [(gname(a), gname(b), gtype(a) + "-" + gtype(b), float(dd)) for (a, b), ww, dd in zip(geom, wid, dist) if ww == w]
                first_bad[w] = {"step": s, "substep": k, "qpos": np.asarray(prev.qpos[w]).tolist(), "qvel": np.asarray(prev.qvel[w]).tolist(),
                                "ctrl": np.asarray(prev.ctrl[w]).tolist(), "warm": np.asarray(prev.qacc_warmstart[w]).tolist(), "pairs": pairs,
                                "maxvel_prev": float(np.abs(np.asarray(prev.qvel[w])).max()),
                                "niter": int(niter_hist[-1][w]) if niter_hist[-1].size == B else -1}
                for p in pairs:
                    pair_at_blowup[p[:2]] += 1; type_at_blowup[p[2]] += 1
                # replace the diverged world by its pre-state so it does not count twice
        if bad.any():
            dw = jax.tree_util.tree_map(lambda a, b: a, dw, dw)  # keep going; diverged worlds stay bad
print(robot, "variant", VARIANT, "solref", os.environ.get("SOLREF", "xml"), "opt", os.environ.get("OPT_OVERRIDE", "{}"), "diverged", len(first_bad), "/", B)
nit = np.concatenate(niter_hist)
print("solver_niter mean %.1f p99 %.0f at_limit %.4f" % (nit.mean(), np.percentile(nit, 99), (nit >= m.opt.iterations).mean()))
print("contact pairs in blow-up step:", pair_at_blowup.most_common(10))
print("pair types:", type_at_blowup.most_common(10))
# MuJoCo C replay of the diverging pre-states
d = mujoco.MjData(m); c_bad = 0
for w, info in list(first_bad.items())[:50]:
    d.qpos[:] = info["qpos"]; d.qvel[:] = info["qvel"]; d.ctrl[:] = info["ctrl"]; d.qacc_warmstart[:] = info["warm"]
    mujoco.mj_step(m, d)
    if not np.all(np.isfinite(d.qvel)) or np.abs(d.qvel).max() > 1e4:
        c_bad += 1
print("C replay of", min(50, len(first_bad)), "diverging pre-states: C also diverges in", c_bad)
for w, info in list(first_bad.items())[:3]:
    print("world", w, "step", info["step"], "maxvel_prev %.2f" % info["maxvel_prev"], "niter", info["niter"], "pairs", info["pairs"][:8])
out = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/sanity") / f"diverge_{robot}.json"
out.write_text(json.dumps({str(k): v for k, v in first_bad.items()}))
