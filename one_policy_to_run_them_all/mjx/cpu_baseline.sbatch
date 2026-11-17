#!/bin/bash
# Stock CPU URMA (RL-X, MuJoCo C, async envs) on the 3 spike robots, learner on CPU, paper batch size.
#SBATCH --partition=long-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=30:00:00
#SBATCH --no-requeue
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu MUJOCO_GL=egl
cd /network/scratch/c/charlie.gauthier/urma_context/mjx/cpu_baseline
/network/scratch/c/charlie.gauthier/urma_context/venv/bin/python /network/scratch/c/charlie.gauthier/urma_context/urma-wt/experiments/experiment.py \
  --algorithm.name=uni_ppo.ppo --algorithm.total_timesteps=89825280 \
  --algorithm.nr_steps=34816 --algorithm.minibatch_size=32640 --algorithm.nr_epochs=10 \
  --algorithm.start_learning_rate=0.0004 --algorithm.end_learning_rate=0.0 --algorithm.entropy_coef=0.0 \
  --algorithm.gae_lambda=0.9 --algorithm.critic_coef=1.0 --algorithm.max_grad_norm=5.0 --algorithm.clip_range=0.1 \
  --algorithm.evaluation_frequency=-1 --algorithm.save_latest_frequency=5222400 --algorithm.device=cpu \
  --environment.name=multi_robot --environment.nr_envs=15 "--environment.train_robot_types=('unitree_go1','unitree_h1','hexapod')" \
  --environment.cycle_cpu_affinity=True --environment.seed=0 --environment.context_arm=urma \
  --runner.mode=train --runner.track_console=True --runner.track_tb=True --runner.track_wandb=False --runner.save_model=True \
  --runner.project_name=urma_mjx_spike --runner.exp_name=cpu_baseline_3robots
