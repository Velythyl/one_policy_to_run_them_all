"""Unimal probe, GPU sanity per robot (MuJoCo-Warp via MJX, the training env MJXRobotEnv with train defaults:
solimp cap 0.99, adaptive impratio mu0 0.3):
  1. descriptions: GPU MJXRobotEnv.descriptions == unimal_convert's limb_context descriptions (urma/dk/k/d/empty)
  2. zero action (PD holds the nominal pose) and Gaussian actions (std 1) for S control steps x N worlds, full
     training env (DR resampling, pushes, delays, random initial state, autoreset): sim_unstable rate, termination
     rate, trunk height / tilt stats, contact-buffer use, overflow bits, steps/s.
  3. Warp vs MuJoCo C, deterministic: drop at nominal, PD at nominal (zero action), no DR, 250 control steps,
     1 world; trunk z / joint q traces.
    python unimal_sanity.py <out_json> robot [robot ...]      (env N, S)
"""
import json
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv
from one_policy_to_run_them_all.mjx.robot_spec import load_spec

out_path = sys.argv[1]
robots = sys.argv[2:]
N, S = int(os.environ.get("N", 1024)), int(os.environ.get("S", 250))
PHYS = '{"impratio_adapt_mu0": 0.3}'
res = json.load(open(out_path)) if os.path.isfile(out_path) else {}


def tilt_deg(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r22 = 1 - 2 * (x * x + y * y)
    return np.degrees(np.arccos(np.clip(r22, -1, 1)))


for r in robots:
    sp = load_spec(r)
    out = {"n_joints": len(sp.cfg["joint_names"]), "n_feet": len(sp.cfg["foot_names"]),
           "nominal_trunk_z": sp.cfg["params"]["reward"]["nominal_trunk_z"], "drop": sp.cfg["initial_drop_height"],
           "term_threshold": sp.cfg["struct"]["termination"]["threshold"]}
    # 1. descriptions
    errs = {}
    for arm in ("urma", "dk", "k", "d", "empty"):
        e = MJXRobotEnv(sp, context_arm=arm, impl="warp", solimp_dmax_clamp=0.99, physics_overrides=PHYS)
        jd, fd, _ = e.descriptions(e.init_dr)
        cj, cf = np.asarray(sp.cfg["cpu_descriptions"][arm]["joints"]), np.asarray(sp.cfg["cpu_descriptions"][arm]["feet"])
        errs[arm] = float(max(np.abs(np.asarray(jd) - cj).max(), np.abs(np.asarray(fd) - cf).max()))
    out["desc_max_abs_err"] = errs
    # 2. training env rollouts
    env = MJXRobotEnv(sp, context_arm="urma", impl="warp", nr_envs_per_type=N, solimp_dmax_clamp=0.99, physics_overrides=PHYS)
    t0 = time.time()
    reset = jax.jit(partial(env.batch_reset, nr_envs=N))
    step = jax.jit(env.batch_step)
    for mode in ("zero", "gauss1"):
        s = reset(jax.random.PRNGKey(0))
        key = jax.random.PRNGKey(1)
        zs, tilts, unst, term, trunc, nacon, nefc, ovf = [], [], 0, 0, 0, [], [], {}
        a0 = jnp.zeros((N, env.nu))
        tc = time.time()
        for t in range(S):
            key, ka = jax.random.split(key)
            a = a0 if mode == "zero" else jax.random.normal(ka, (N, env.nu))
            s, info = step(s, a)
            if t == 0:
                jax.block_until_ready(s.data.qpos)
                compile_s = time.time() - tc
                tr = time.time()
            if t % 10 == 9 or t == S - 1:
                q = np.asarray(info["qpos_next"])
                zs.append(q[:, 2]); tilts.append(tilt_deg(q[:, 3:7]))
            unst += int(np.asarray(info["sim_unstable"]).sum())
            term += int(np.asarray(info["terminated"]).sum()); trunc += int(np.asarray(info["truncated"]).sum())
            nacon.append(float(info["nacon_frac"])); nefc.append(int(info["nefc_max"]))
            for k in [k for k in info if k.startswith("ovf_")]:
                ovf[k] = ovf.get(k, 0) + int(np.asarray(info[k]).sum())
        jax.block_until_ready(s.data.qpos)
        sps = N * (S - 1) / (time.time() - tr)
        zs, tilts = np.stack(zs), np.stack(tilts)
        out[mode] = {"compile_s": round(compile_s, 1), "env_steps_per_s": round(sps), "unstable": unst,
                     "unstable_rate": unst / (N * S), "terminated_rate_per_step": term / (N * S),
                     "trunk_z_mean_last": float(zs[-1].mean()), "trunk_z_p10_last": float(np.percentile(zs[-1], 10)),
                     "trunk_z_mean_traj": zs.mean(1).round(3).tolist(), "tilt_median_last": float(np.median(tilts[-1])),
                     "nacon_frac_max": max(nacon), "nefc_max": max(nefc), "overflow_world_steps": ovf}
        print(r, mode, json.dumps({k: v for k, v in out[mode].items() if k != "trunk_z_mean_traj"}), flush=True)
    # 3. Warp vs C, deterministic PD hold from the drop pose (default model, train-default solimp cap)
    m = env.mj_model
    c = sp.cfg
    nom = np.asarray(c["nominal_joint_positions"])
    p, dg = c["params"]["control"]["p_gain"], c["params"]["control"]["d_gain"]
    qpos0 = m.qpos0.copy(); qpos0[0:3] = [0, 0, c["initial_drop_height"]]; qpos0[3:7] = [1, 0, 0, 0]; qpos0[7:] = nom
    dC = mujoco.MjData(m); dC.qpos[:] = qpos0; mujoco.mj_forward(m, dC)
    mw = env.mx
    dW = mjx.make_data(m, impl="warp", naconmax=64, njmax=512)
    dW = dW.replace(qpos=jnp.asarray(qpos0, jnp.float32))

    @jax.jit
    def wstep(d):
        tau = p * (jnp.asarray(nom) - d.qpos[7:]) - dg * d.qvel[6:]
        d = d.replace(ctrl=tau)
        for _ in range(env.nsub):
            d = mjx.step(mw, d)
        return d
    zc, zw, qerr = [], [], []
    for t in range(250):
        dC.ctrl[:] = p * (nom - dC.qpos[7:]) - dg * dC.qvel[6:]
        for _ in range(env.nsub):
            mujoco.mj_step(m, dC)
        dW = wstep(dW)
        qw = np.asarray(dW.qpos)
        zc.append(dC.qpos[2]); zw.append(qw[2]); qerr.append(float(np.abs(qw[7:] - dC.qpos[7:]).max()))
    out["warp_vs_c"] = {"z_c_end": float(zc[-1]), "z_w_end": float(zw[-1]), "max_abs_dz": float(np.max(np.abs(np.subtract(zc, zw)))),
                        "joint_err_end_deg": float(np.degrees(qerr[-1])), "joint_err_max_deg": float(np.degrees(max(qerr))),
                        "tilt_c_end": float(tilt_deg(dC.qpos[3:7])), "tilt_w_end": float(tilt_deg(np.asarray(dW.qpos[3:7]))),
                        "ncon_c_end": int(dC.ncon)}
    print(r, "warp_vs_c", json.dumps(out["warp_vs_c"]), "desc_err", json.dumps(errs), "total_s %.0f" % (time.time() - t0), flush=True)
    res[r] = out
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
