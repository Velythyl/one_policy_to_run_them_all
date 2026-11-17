#!/bin/bash
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
PY=/network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python
B=/network/scratch/c/charlie.gauthier/urma_context/urma-mjx-wt/one_policy_to_run_them_all/mjx/bench_physics.py
nvidia-smi -L
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 5 > /network/scratch/c/charlie.gauthier/urma_context/jobs/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.gpu.csv &
for spec in "$@"; do
  set -- $(echo $spec | tr ':' ' ')
  echo "### $spec"
  timeout 1200 $PY $B "$@" 2>&1 | grep -E "^\{|Error|error|NotImplemented|Traceback|not supported" | head -5
done
