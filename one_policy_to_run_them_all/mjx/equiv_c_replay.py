"""Replay the CPU dump (equiv_cpu_dump.py, recorded with the CPU venv's MuJoCo) one step at a time in
the MuJoCo C of this venv: separates MuJoCo-version effects from Warp effects.
    python equiv_c_replay.py <dump_dir> robot [robot ...]"""
import json, sys
from pathlib import Path
import mujoco, numpy as np
ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
CFG = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2")
dump = Path(sys.argv[1])
for robot in sys.argv[2:]:
    c = json.loads((CFG / f"{robot}.json").read_text()); z = np.load(dump / f"{robot}.npz")
    m = mujoco.MjModel.from_xml_path(str(ENV_ROOT / robot / "data" / c["xml_file"])); m.opt.timestep = c["timestep"]
    d = mujoco.MjData(m)
    P = c["params"]["control"]; qj = np.asarray(c["qpos_joint_idx"], int); vj = np.asarray(c["qvel_joint_idx"], int)
    nom = np.asarray(c["nominal_joint_positions"]); vel = np.asarray(c["max_joint_velocities"])
    floor = set(c["collision_groups"]["floor"])
    feet = [set(c["collision_groups"].get(f, [])) for f in c["foot_names"]]
    qe, fe = [], 0
    for i in range(len(z["reward"])):
        d.qpos[:] = z["qpos"][i]; d.qvel[:] = z["qvel"][i]; d.qacc_warmstart[:] = z["warm"][i]
        a = z["action"][i]
        d.ctrl[:] = P["p_gain"] * (nom + a * P["scaling_factor"] - d.qpos[qj]) - P["d_gain"] * d.qvel[vj]
        mujoco.mj_step(m, d, c["nr_substeps"])
        d.qvel[vj] = np.clip(d.qvel[vj], -vel, vel)
        qe.append(np.abs(d.qpos - z["qpos_next"][i]).max())
        fc = [any((cc.geom1 in floor and cc.geom2 in f) or (cc.geom2 in floor and cc.geom1 in f) for cc in d.contact[:d.ncon]) for f in feet]
        if c["struct"]["foot_obs_mode"] == "live":
            nf = len(feet); fs = z["obs"][i][-(20 + 12 * nf):-20].reshape(nf, 12)[:, 10]
            fe += int(np.sum(np.asarray(fc, float) / 0.5 - 1.0 != fs))
    qe = np.asarray(qe)
    print(robot, "C(%s) vs CPU-env: qpos err max %.2e p99 %.2e; foot-contact obs mismatches %d / %d" % (mujoco.__version__, qe.max(), np.percentile(qe, 99), fe, len(qe) * len(feet)), flush=True)
