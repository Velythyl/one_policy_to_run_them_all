#!/bin/bash
# Short URMA training run (trial / per-arm smoke). Usage:
#   sbatch --job-name=urma-<tag> train_short.sbatch <context_arm> <total_timesteps> [extra args...]
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
set -euo pipefail
ARM=${1:-urma}; TOTAL=${2:-10240}; shift 2 || true
ROOT=/network/scratch/c/charlie.gauthier/urma_context
cd $ROOT/runs_cwd
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
echo "host $(hostname) arm $ARM total $TOTAL"; nvidia-smi -L || true
# venv mixes cu11 (torch, unused by uni_ppo) and cu12 (jax) wheels; jax needs cu12 cudnn 8.9 + unversioned libnvrtc.so
NV=$ROOT/venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cuda_nvrtc/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
$ROOT/venv/bin/python -c "import jax, jax.numpy as jnp; print(jax.devices()); assert jax.default_backend() == \"gpu\", \"no GPU backend\"; print(jnp.ones((64,64)) @ jnp.ones((64,64)))" | tail -2
# nr_envs must be a multiple of the 16 train robot types; nr_steps*nr_envs = 2048 per rollout
srun --cpu-bind=none $ROOT/venv/bin/python $ROOT/urma-wt/experiments/experiment.py \
    --algorithm.name="uni_ppo.ppo" \
    --algorithm.total_timesteps=$TOTAL \
    --algorithm.nr_steps=128 \
    --algorithm.minibatch_size=512 \
    --algorithm.nr_epochs=2 \
    --algorithm.start_learning_rate=0.0004 --algorithm.end_learning_rate=0.0 \
    --algorithm.entropy_coef=0.0 --algorithm.gae_lambda=0.9 --algorithm.critic_coef=1.0 \
    --algorithm.max_grad_norm=5.0 --algorithm.clip_range=0.1 \
    --algorithm.evaluation_frequency=-1 \
    --algorithm.save_latest_frequency=4096 \
    --algorithm.device="gpu" \
    --environment.name="multi_robot" \
    --environment.nr_envs=16 \
    --environment.seed=0 \
    --environment.context_arm=$ARM \
    --runner.mode="train" \
    --runner.track_console=True --runner.track_tb=True --runner.track_wandb=False \
    --runner.save_model=True \
    --runner.project_name="urma_context" --runner.exp_name="${SLURM_JOB_NAME}" \
    "$@"
echo "exit OK"; ls -R runs/urma_context/${SLURM_JOB_NAME} | head -20
