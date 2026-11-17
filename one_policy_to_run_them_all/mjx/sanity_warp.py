"""Quick MuJoCo-Warp sanity pass per robot (physics only, no RL env):
loads the XML into MJX-Warp, lists features Warp may not support, runs B worlds of PD control
around the nominal pose with random targets (URMA gains / substeps / init ranges from the
extracted config) for S control steps, and the same for a few worlds in MuJoCo C.
Reports: divergence rate, contacts/world (Warp vs C), contact-buffer overflow, trunk height,
step throughput.   python sanity_warp.py <cfg_dir> <out_json> <xml_override_json|-> robot [robot ...]
"""
import json
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
cfg_dir, out_path, override = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
robots = sys.argv[4:]
xml_over = json.loads(Path(override).read_text()) if override != "-" else {}
import os
B, S, CB, CPW = int(os.environ.get("B", 1024)), int(os.environ.get("S", 300)), 16, 64
results = json.loads(out_path.read_text()) if out_path.is_file() else {}


def features(m):
    f = {"nq": m.nq, "nv": m.nv, "nu": m.nu, "ngeom": m.ngeom, "nbody": m.nbody,
         "neq": m.neq, "eq_types": sorted({int(t) for t in m.eq_type}), "ntendon": m.ntendon,
         "nmesh_colliders": int(sum(1 for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH and (m.geom_contype[i] or m.geom_conaffinity[i]))),
         "ncolliders": int(sum(1 for i in range(m.ngeom) if (m.geom_contype[i] or m.geom_conaffinity[i]))),
         "geom_types": sorted({mujoco.mjtGeom(int(t)).name for i, t in enumerate(m.geom_type) if (m.geom_contype[i] or m.geom_conaffinity[i])}),
         "solver": int(m.opt.solver), "iterations": m.opt.iterations, "ls_iterations": m.opt.ls_iterations,
         "cone": int(m.opt.cone), "integrator": int(m.opt.integrator), "timestep": m.opt.timestep,
         "nmeshvert_max": int(max([m.mesh_vertnum[i] for i in range(m.nmesh)] or [0]))}
    return f


for robot in robots:
    c = json.loads((cfg_dir / f"{robot}.json").read_text())
    xml = xml_over.get(robot, str(ENV_ROOT / robot / "data" / c["xml_file"]))
    r = {"xml": xml}
    try:
        m = mujoco.MjModel.from_xml_path(xml)
        m.opt.timestep = c["timestep"]
        import os
        for k_, v_ in json.loads(os.environ.get("OPT_OVERRIDE", "{}")).items():
            setattr(m.opt, k_, v_)
        r["opt_override"] = os.environ.get("OPT_OVERRIDE", "{}")
        r["tolerance"] = float(m.opt.tolerance)
        r["features"] = features(m)
        P = c["params"]["control"]
        qj = np.asarray(c["qpos_joint_idx"], int); vj = np.asarray(c["qvel_joint_idx"], int)
        nom = np.asarray(c["nominal_joint_positions"])
        base = np.asarray(c["complete_nominal_joint_position"]) if c["complete_nominal_joint_position"] else m.qpos0.copy()
        base = base.copy(); base[2] = c["initial_drop_height"]; base[3:7] = [1, 0, 0, 0]; base[qj] = nom
        nsub = c["nr_substeps"]
        rng = np.random.default_rng(0)
        targets = nom + rng.normal(0, 0.3, (S, B, len(nom))) * P["scaling_factor"] / 0.25 * 0.25
        q0 = np.tile(base, (B, 1)); q0[:, qj] += rng.uniform(-0.1, 0.1, (B, len(nom)))
        # ---- MuJoCo C reference on CB worlds
        d = mujoco.MjData(m)
        c_div, c_ncon, c_h, c_ncon_max = 0, [], [], 0
        t0 = time.time()
        for w in range(0 if os.environ.get("SKIP_C") else CB):
            mujoco.mj_resetData(m, d); d.qpos[:] = q0[w]; mujoco.mj_forward(m, d)
            for s in range(S):
                d.ctrl[:] = P["p_gain"] * (targets[s, w] - d.qpos[qj]) - P["d_gain"] * d.qvel[vj]
                mujoco.mj_step(m, d, nsub)
                if not np.all(np.isfinite(d.qvel)) or np.abs(d.qvel).max() > 1e4 or any(d.warning[i].number for i in range(len(d.warning))):
                    c_div += 1; break
                act = sum(1 for k in range(d.ncon) if d.contact[k].dist < d.contact[k].includemargin)
                c_ncon.append(act); c_ncon_max = max(c_ncon_max, d.ncon)
            c_h.append(float(d.qpos[2]))
        r["c"] = {"diverged_worlds": c_div, "of": CB, "contacts_mean": float(np.mean(c_ncon)) if c_ncon else 0.0,
                  "contacts_max": c_ncon_max, "final_height_mean": float(np.mean(c_h)) if c_h else 0.0, "ctrl_steps_per_s": round(CB * S / (time.time() - t0 + 1e-9))}
        # ---- Warp
        mx = mjx.put_model(m, impl="warp")
        dw = jax.vmap(lambda _: mjx.make_data(m, impl="warp", naconmax=B * CPW, njmax=CPW * 8))(jnp.arange(B))
        dw = dw.replace(qpos=jnp.asarray(q0, jnp.float32))
        dw = jax.vmap(mjx.forward, in_axes=(None, 0))(mx, dw)
        kp, kd = P["p_gain"], P["d_gain"]

        @jax.jit
        def ctrl_step(dw, tgt):
            tau = kp * (tgt - dw.qpos[:, qj]) - kd * dw.qvel[:, vj]
            dw = dw.replace(ctrl=tau)
            dw = jax.lax.fori_loop(0, nsub, lambda i, x: jax.vmap(mjx.step, in_axes=(None, 0))(mx, x), dw)
            imp = dw._impl
            n = jnp.asarray(imp.nacon).reshape(-1)[0]
            bad = ~(jnp.all(jnp.isfinite(dw.qvel), -1) & jnp.all(jnp.abs(dw.qvel) < 1e4, -1))
            valid = (jnp.arange(imp.contact__dist.reshape(-1).shape[0]) < n) & (imp.contact__dist.reshape(-1) < imp.contact__includemargin.reshape(-1))
            return dw, bad, n, valid.sum()
        tc = time.time()
        dw1, bad, n, nv = ctrl_step(dw, jnp.asarray(targets[0], jnp.float32)); jax.block_until_ready(bad)
        r["warp_compile_s"] = round(time.time() - tc, 1)
        ever_bad = np.zeros(B, bool); nacon_max = 0; ncon_valid = []
        t0 = time.time()
        dw = dw1
        for s in range(1, S):
            dw, bad, n, nv = ctrl_step(dw, jnp.asarray(targets[s], jnp.float32))
            if s % 10 == 0:
                ever_bad |= np.asarray(bad); nacon_max = max(nacon_max, int(n)); ncon_valid.append(int(nv) / B)
        jax.block_until_ready(dw.qpos)
        dt = time.time() - t0
        h = np.asarray(dw.qpos[:, 2])
        r["warp"] = {"diverged_worlds": int(ever_bad.sum()), "of": B, "contacts_mean": float(np.mean(ncon_valid)),
                     "nacon_max": nacon_max, "naconmax": B * CPW, "overflow": bool(nacon_max >= B * CPW),
                     "final_height_mean": float(np.nanmean(np.where(ever_bad, np.nan, h))),
                     "ctrl_steps_per_s": round(B * (S - 1) / dt)}
        r["ok"] = True
    except Exception as ex:  # report and continue with the next robot
        import traceback
        r["ok"] = False; r["error"] = traceback.format_exc()[-1500:]
    results[robot] = r
    print(robot, json.dumps({k: v for k, v in r.items() if k not in ("features",)})[:900], flush=True)
    print(robot, "features", json.dumps(r.get("features")), flush=True)
    out_path.write_text(json.dumps(results, indent=1))
print("DONE")
