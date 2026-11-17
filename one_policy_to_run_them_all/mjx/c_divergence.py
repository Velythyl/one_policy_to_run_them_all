"""MuJoCo C (float64) divergence rate under the same random-PD protocol as sanity_warp.py (CPU only).
    python c_divergence.py <cfg_dir> <nworlds> robot [robot ...]"""
import json, sys
from pathlib import Path
import mujoco, numpy as np
ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
cfg_dir, W = Path(sys.argv[1]), int(sys.argv[2]); S = 300
for robot in sys.argv[3:]:
    c = json.loads((cfg_dir / f"{robot}.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(ENV_ROOT / robot / "data" / c["xml_file"])); m.opt.timestep = c["timestep"]
    P = c["params"]["control"]; qj = np.asarray(c["qpos_joint_idx"], int); vj = np.asarray(c["qvel_joint_idx"], int)
    nom = np.asarray(c["nominal_joint_positions"])
    base = np.asarray(c["complete_nominal_joint_position"]) if c["complete_nominal_joint_position"] else m.qpos0.copy()
    base = base.copy(); base[2] = c["initial_drop_height"]; base[3:7] = [1, 0, 0, 0]; base[qj] = nom
    rng = np.random.default_rng(0)
    targets = nom + rng.normal(0, 0.3, (S, 1024, len(nom)))
    q0 = np.tile(base, (1024, 1)); q0[:, qj] += rng.uniform(-0.1, 0.1, (1024, len(nom)))
    d = mujoco.MjData(m); div = 0; iters = []
    for w in range(W):
        mujoco.mj_resetData(m, d); d.qpos[:] = q0[w]; mujoco.mj_forward(m, d)
        for s in range(S):
            d.ctrl[:] = P["p_gain"] * (targets[s, w] - d.qpos[qj]) - P["d_gain"] * d.qvel[vj]
            for _ in range(c["nr_substeps"]):
                mujoco.mj_step(m, d); iters.append(d.solver_niter[0])
            if not np.all(np.isfinite(d.qvel)) or np.abs(d.qvel).max() > 1e4 or d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number:
                div += 1; break
    it = np.asarray(iters)
    print(robot, "C diverged", div, "/", W, "solver iters mean %.1f p99 %d max %d at_limit %.4f" % (it.mean(), np.percentile(it, 99), it.max(), (it >= m.opt.iterations).mean()),
          "tol", m.opt.tolerance, "iters", m.opt.iterations, "ls", m.opt.ls_iterations, flush=True)
