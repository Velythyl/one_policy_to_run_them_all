"""MuJoCo C (float64) only: how much does lowering impratio (adaptive rule, or elliptic cone) change the
physics vs the upstream model (pyramidal, impratio 100) at a given ground/feet tangential friction?
Same random PD targets (std 0.3 rad around nominal), B rollouts x S control steps from a standing drop.
Reports mean trunk xy slip distance, |dq| between variants at the end, and fraction of rollouts fallen.
    python c_impratio_effect.py <robot> [B=8] [S=500]"""
import copy
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

robot = sys.argv[1]
B = int(sys.argv[2]) if len(sys.argv) > 2 else 8
S = int(sys.argv[3]) if len(sys.argv) > 3 else 500
CFG = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2")
c = json.loads((CFG / f"{robot}.json").read_text())
ENV = Path(__file__).resolve().parents[1] / "environments"
m0 = mujoco.MjModel.from_xml_path(str(ENV / robot / "data" / c["xml_file"]))
m0.opt.timestep = c["timestep"]
hit = m0.geom_solimp[:, 1] > 0.99
m0.geom_solimp[hit, 1] = 0.99; m0.geom_solimp[:, 0] = np.minimum(m0.geom_solimp[:, 0], 0.99)
P = c["params"]["control"]; kp, kd = P["p_gain"], P["d_gain"]
qj = np.asarray(c["qpos_joint_idx"], int); vj = np.asarray(c["qvel_joint_idx"], int)
nom = np.asarray(c["nominal_joint_positions"]); nsub = c["nr_substeps"]
feet = np.asarray(c["struct"]["model_dr"]["friction_feet_geoms"], int).ravel()


def g(u):
    return u * u * (1 + u * u)


def model(mu, variant):
    m = copy.deepcopy(m0)
    m.geom_friction[0, 0] = mu
    if feet is not None:
        m.geom_friction[feet, 0] = mu
    if variant.startswith("adapt"):
        mu0 = float(variant[5:]) / 10
        m.opt.impratio = m0.opt.impratio * min(1.0, g(mu) / g(mu0))
    elif variant == "elliptic":
        m.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    return m


def rollout(m, seed, eps=0.0):
    rng = np.random.default_rng(seed)
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    if c["complete_nominal_joint_position"]:
        d.qpos[:] = c["complete_nominal_joint_position"]
    d.qpos[2] = c["initial_drop_height"]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[qj] = nom
    d.qpos[qj] += eps
    mujoco.mj_forward(m, d)
    xy0 = d.qpos[:2].copy()
    for t in range(S):
        tgt = nom + rng.normal(0, 0.3, len(nom))
        d.ctrl[:len(nom)] = kp * (tgt - d.qpos[qj]) - kd * d.qvel[vj]
        mujoco.mj_step(m, d, nsub)
    return np.linalg.norm(d.qpos[:2] - xy0), d.qpos.copy(), float(d.qpos[2])


out = {}
for mu in [1.0, 0.3, 0.1, 0.03, 0.01, 0.001]:
    row = {}
    ref = [rollout(model(mu, "base"), s) for s in range(B)]
    row["base_slip"] = float(np.mean([r[0] for r in ref]))
    row["base_z"] = float(np.mean([r[2] for r in ref]))
    for v in ["chaos", "adapt3", "elliptic"]:
        rr = [rollout(model(mu, "base"), s, 1e-9) if v == "chaos" else rollout(model(mu, v), s) for s in range(B)]
        row[f"{v}_slip"] = float(np.mean([r[0] for r in rr]))
        row[f"{v}_z"] = float(np.mean([r[2] for r in rr]))
        row[f"{v}_dxy_vs_base"] = float(np.mean([np.linalg.norm(a[1][:2] - b[1][:2]) for a, b in zip(rr, ref)]))
    out[mu] = row
    print(robot, mu, json.dumps({k: round(v, 4) for k, v in row.items()}), flush=True)
