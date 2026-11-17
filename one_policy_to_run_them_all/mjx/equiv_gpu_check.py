"""GPU side of the per-step equivalence check (see equiv_cpu_dump.py).  Each recorded CPU step
becomes one world: pre-state loaded, same action applied, outputs compared.
    python equiv_gpu_check.py <dump_dir> <out_json> robot [robot ...]
"""
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

dump, out = Path(sys.argv[1]), Path(sys.argv[2])
res = json.loads(out.read_text()) if out.is_file() else {}
for robot in sys.argv[3:]:
    try:
        z = np.load(dump / f"{robot}.npz")
        K = len(z["reward"])
        env = MJXRobotEnv(load_spec(robot), context_arm="urma", impl="warp", nr_envs_per_type=1,
                          solimp_dmax_clamp=float(os.environ.get("SOLIMP_CLAMP", 0)))
        P = env.P
        P["noise"] = {k: 0.0 for k in P["noise"]}
        P["dropout"] = {**P["dropout"], "dynamic_dropout_chance": 0.0}
        P["dr_sampling_prob"] = 0.0; P["perturb_sampling_prob"] = 0.0; P["command_sampling_prob"] = 0.0
        st = jax.jit(lambda k: env.batch_reset(k, K))(jax.random.PRNGKey(0))
        f32 = lambda x: jnp.asarray(x, jnp.float32)
        data = st.data.replace(qpos=f32(z["qpos"]), qvel=f32(z["qvel"]), qacc_warmstart=f32(z["warm"]))
        st = st.replace(data=data, goal=f32(z["goal"]), timers=f32(z["timers"]), prev_joint_vel=f32(z["prev_vel"]),
                        last_action=f32(z["last_action"]), current_action=f32(z["last_action"]),
                        episode_step=jnp.asarray(z["episode_step"], jnp.int32),
                        steps_done=jnp.asarray(np.maximum(z["total_timesteps"], 1), jnp.int32))
        st2, info = jax.jit(env.batch_step)(st, f32(z["action"]))
        o = info["final_obs"]
        n, nf = env.n, env.nf
        flat = np.concatenate([np.concatenate([np.asarray(o["jd"]), np.asarray(o["js"])], -1).reshape(K, -1),
                               np.concatenate([np.asarray(o["fd"]), np.asarray(o["fs"])], -1).reshape(K, -1),
                               np.asarray(o["gc"])], -1)
        cobs = z["obs"]
        assert flat.shape == cobs.shape, (flat.shape, cobs.shape)
        unst = np.asarray(info["sim_unstable"])
        ok = ~unst
        rew_err = np.abs(np.asarray(info["reward"]) - z["reward"])
        term_mis = np.asarray(info["terminated"]) != z["terminated"]
        oerr = np.abs(flat - cobs)
        qerr = np.abs(np.asarray(info["qpos_next"]) - z["qpos_next"])[ok]
        verr = np.abs(np.asarray(info["qvel_next"]) - z["qvel_next"])[ok]
        worst_obs = np.argsort(-oerr.max(0))[:5]
        r = {"steps": K, "cpu_terminations": int(z["terminated"].sum()), "gpu_terminations": int(np.asarray(info["terminated"]).sum()),
             "termination_mismatch": int(term_mis.sum()), "sim_unstable": int(unst.sum()),
             "reward_err_max": float(rew_err[ok].max()), "reward_err_p99": float(np.percentile(rew_err[ok], 99)),
             "reward_mean_cpu": float(z["reward"].mean()),
             "obs_err_max": float(oerr[ok].max()), "obs_err_p99": float(np.percentile(oerr[ok].max(1), 99)),
             "obs_worst_idx": worst_obs.tolist(), "obs_worst_err": oerr.max(0)[worst_obs].round(5).tolist(),
             "qpos_err_max": float(qerr.max()), "qpos_err_p99": float(np.percentile(qerr.max(1), 99)),
             "qvel_err_max": float(verr.max()), "qvel_err_p99": float(np.percentile(verr.max(1), 99)),
             "mismatch_steps": np.nonzero(term_mis)[0][:10].tolist()}
    except Exception:
        import traceback
        r = {"error": traceback.format_exc()[-2500:]}
    res[robot] = r
    print(robot, json.dumps(r), flush=True)
    out.write_text(json.dumps(res, indent=1))
print("DONE")
