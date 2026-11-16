"""Build single robot envs exactly as multi_robot/create_env.py does (no vectorization)."""
from one_policy_to_run_them_all.environments.multi_robot.default_config import get_config
from one_policy_to_run_them_all.environments.multi_robot.robot_helper import ROBOTS


def build_env(long_name, seed=0, **extra):
    c = get_config("multi_robot")
    robot = next(r for r in ROBOTS if r.long_name == long_name)
    return robot.cls(
        seed=seed, render=False, mode="train",
        control_type=c.control_type, command_type=c.command_type,
        command_sampling_type=c.command_sampling_type, initial_state_type=c.initial_state_type,
        reward_type=c.reward_type, termination_type=getattr(c, f"termination_type_{robot.short_name}"),
        domain_randomization_sampling_type=c.domain_randomization_sampling_type,
        domain_randomization_action_delay_type=c.domain_randomization_action_delay_type,
        domain_randomization_mujoco_model_type=c.domain_randomization_mujoco_model_type,
        domain_randomization_seen_robot_type=c.domain_randomization_seen_robot_type,
        domain_randomization_unseen_robot_type=c.domain_randomization_unseen_robot_type,
        domain_randomization_perturbation_type=c.domain_randomization_perturbation_type,
        domain_randomization_perturbation_sampling_type=c.domain_randomization_perturbation_sampling_type,
        observation_noise_type=c.observation_noise_type, observation_dropout_type=c.observation_dropout_type,
        terrain_type=c.terrain_type, missing_value=c.missing_value, add_goal_arrow=False,
        timestep=c.timestep, episode_length_in_seconds=c.episode_length_in_seconds, total_nr_envs=1,
        **extra,
    )


ALL_ROBOTS = [r.long_name for r in ROBOTS]
