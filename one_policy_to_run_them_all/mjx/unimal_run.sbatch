#!/bin/bash
# Unimal probe generic GPU job: sbatch --job-name=urma-unimal-<tag> unimal_run.sbatch <script.py> [args...]
#SBATCH --partition=main,unkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --exclude=cn-g028
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/unimal_probe/jobs/%x-%j.out
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export PYTHONPATH=/network/scratch/c/charlie.gauthier/urma_context/urma-unimal-wt
export URMA_MJX_CFG_EXTRA=/network/scratch/c/charlie.gauthier/urma_context/unimal_probe/gen/configs
export WARP_CACHE_PATH=/network/scratch/c/charlie.gauthier/urma_context/mjx/warp_cache JAX_COMPILATION_CACHE_DIR=/network/scratch/c/charlie.gauthier/urma_context/mjx/jax_cache
export XDG_CACHE_HOME=/network/scratch/c/charlie.gauthier/urma_context/mjx/xdg_cache
nvidia-smi -L
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > /network/scratch/c/charlie.gauthier/urma_context/unimal_probe/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv &
cd /network/scratch/c/charlie.gauthier/urma_context/unimal_probe
if [[ "$1" == *.sh ]]; then bash "$@"; else /network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python "$@"; fi
echo "EXIT $?"
