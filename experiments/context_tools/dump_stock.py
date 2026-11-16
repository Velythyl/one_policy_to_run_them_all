"""Dump stock-URMA description vectors (default + after N DR samples) for every robot.
Run once on the unmodified code; verify_context.py compares the 'urma' arm and dk against it."""
import sys
import numpy as np
from envs import build_env, ALL_ROBOTS

out = sys.argv[1]
extra = {} if len(sys.argv) < 3 else {"context_arm": sys.argv[2]}
res = {}
for name in ALL_ROBOTS:
    env = build_env(name, seed=0, **extra)
    env.multi_robot_max_observation_size = -1
    for k, v in env.name_to_description_vector.items():
        res[f"{name}/0/{k}"] = v
    for s in range(1, 4):  # DR samples (seed-deterministic)
        env.domain_randomization_unseen_robot_function.sample()
        env.domain_randomization_seen_robot_function.sample()
        env.domain_randomization_mujoco_model_function.sample()
        for k, v in env.get_name_to_description_vector().items():
            res[f"{name}/{s}/{k}"] = v
    print(name, len(env.joint_names), "joints", env.dynamic_joint_description_size, env.dynamic_foot_description_size, flush=True)
np.savez(out, **res)
