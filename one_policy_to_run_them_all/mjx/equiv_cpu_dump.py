"""CPU side of the per-step equivalence check (run in the CPU venv).

Runs the real CPU robot env (multi_robot settings, random init states) with DR resampling,
perturbations, observation noise/dropout and action delay switched off (all left at their default
values), random actions, and records for every step the full pre-step state and the CPU outputs.
equiv_gpu_check.py then loads each recorded pre-state into the MJX env (one world per recorded
step), applies the same action and compares reward / termination / observation / next state.

    python equiv_cpu_dump.py <out_dir> <nr_steps> robot [robot ...]
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "context_tools"))
from envs import build_env  # noqa: E402

CFG = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2")


def main():
    out_dir, S = Path(sys.argv[1]), int(sys.argv[2])
    out_dir.mkdir(parents=True, exist_ok=True)
    for robot in sys.argv[3:]:
        cfg = json.loads((CFG / f"{robot}.json").read_text())
        st = cfg["struct"]
        env = build_env(robot, seed=1, context_arm="urma")
        env.domain_randomization_sampling_function.probability = 0.0
        env.domain_randomization_perturbation_sampling_function.probability = 0.0
        env.observation_noise_function.modify_observation = lambda o: o
        env.observation_dropout_function.modify_observation = lambda o: o
        rf = env.reward_function
        t2f = st["timer_attr_to_foot"]
        nf = len(env.foot_names)
        rng = np.random.default_rng(0)
        env.reset()
        cur_steps = int(0.5 * rf.curriculum_steps)  # mid-curriculum so every penalty term is active
        env.total_timesteps = cur_steps
        rec = {k: [] for k in ("qpos", "qvel", "warm", "goal", "timers", "prev_vel", "last_action", "episode_step",
                               "total_timesteps", "action", "reward", "terminated", "truncated", "obs", "qpos_next", "qvel_next")}
        nu = env.model.nu
        n_obs = None
        for i in range(S):
            timers = np.zeros(nf)
            for a, f in t2f.items():
                timers[f] = getattr(rf, a)
            pv = np.asarray(rf._prev_joint_vel if rf._prev_joint_vel is not None else np.zeros(len(env.joint_names)), dtype=np.float64)
            pre = dict(qpos=env.data.qpos.copy(), qvel=env.data.qvel.copy(), warm=env.data.qacc_warmstart.copy(),
                       timers=timers, prev_vel=pv.copy(), last_action=np.asarray(env.last_action, dtype=np.float64).copy(),
                       episode_step=env.episode_step, total_timesteps=env.total_timesteps)
            action = rng.normal(0.0, 0.6, nu)
            obs, reward, term, trunc, info = env.step(action)
            n_obs = n_obs or len(obs)
            for k, v in pre.items():
                rec[k].append(v)
            rec["goal"].append([env.goal_x_velocity, env.goal_y_velocity, env.goal_yaw_velocity])
            rec["action"].append(action); rec["reward"].append(reward); rec["terminated"].append(bool(term)); rec["truncated"].append(bool(trunc))
            rec["obs"].append(np.asarray(obs, dtype=np.float64)); rec["qpos_next"].append(env.data.qpos.copy()); rec["qvel_next"].append(env.data.qvel.copy())
            if term or trunc or (i % 97 == 96):  # also restart now and then for more initial states
                env.reset()
                env.total_timesteps = cur_steps
        np.savez_compressed(out_dir / f"{robot}.npz", **{k: np.asarray(v) for k, v in rec.items()})
        print(robot, "steps", S, "terminations", int(np.sum(rec["terminated"])), "obs_len", n_obs, flush=True)
        env.close()


if __name__ == "__main__":
    main()
