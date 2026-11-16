#!/bin/bash
# Smoke-train each context arm briefly, one after another, in one unkillable job.
#   sbatch --job-name=urma-smoke smoke_arms.sbatch <total_timesteps_per_arm> <latent_dir> [arms...]
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
set -uo pipefail
TOTAL=$1; LATENT_DIR=$2; shift 2
ARMS=${@:-empty dk d k vme matme}
ROOT=/network/scratch/c/charlie.gauthier/urma_context
cd $ROOT/runs_cwd
export XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYTHONUNBUFFERED=1
NV=$ROOT/venv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cuda_nvrtc/lib:$NV/cudnn/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
# peak VRAM is not in sacct TRES on Mila; sample it
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > $ROOT/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv &
for ARM in $ARMS; do
  echo "===== ARM $ARM start $(date +%T)"
  srun --cpu-bind=none $ROOT/venv/bin/python $ROOT/urma-wt/experiments/experiment.py \
    --algorithm.name="uni_ppo.ppo" --algorithm.total_timesteps=$TOTAL \
    --algorithm.nr_steps=128 --algorithm.minibatch_size=512 --algorithm.nr_epochs=2 \
    --algorithm.start_learning_rate=0.0004 --algorithm.end_learning_rate=0.0 \
    --algorithm.entropy_coef=0.0 --algorithm.gae_lambda=0.9 --algorithm.critic_coef=1.0 \
    --algorithm.max_grad_norm=5.0 --algorithm.clip_range=0.1 \
    --algorithm.evaluation_frequency=-1 --algorithm.save_latest_frequency=$TOTAL \
    --algorithm.device="gpu" --environment.name="multi_robot" --environment.nr_envs=16 --environment.seed=0 \
    --environment.context_arm=$ARM --environment.latent_dir=$LATENT_DIR \
    --runner.mode="train" --runner.track_console=True --runner.track_tb=True --runner.track_wandb=False \
    --runner.save_model=True --runner.project_name="urma_context" --runner.exp_name="${SLURM_JOB_NAME}-${ARM}"
  echo "===== ARM $ARM exit $? end $(date +%T)"
done
