#!/bin/bash
# Throughput of the full jitted PPO iteration (rollout + update) per config.
#   sbatch --job-name=urma-mjx-bench bench_train.sbatch "unitree_go1:4096" "unitree_go1,unitree_h1,hexapod:4096:native" ...
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
export PYTHONPATH=/network/scratch/c/charlie.gauthier/urma_context/urma-mjx-wt
export WARP_CACHE_PATH=/network/scratch/c/charlie.gauthier/urma_context/mjx/warp_cache JAX_COMPILATION_CACHE_DIR=/network/scratch/c/charlie.gauthier/urma_context/mjx/jax_cache
export XDG_CACHE_HOME=/network/scratch/c/charlie.gauthier/urma_context/mjx/xdg_cache
nvidia-smi -L
for spec in "$@"; do
  IFS=':' read -r ROBOTS NENVS VARIANT IMPL <<< "$spec"
  VARIANT=${VARIANT:-native}; IMPL=${IMPL:-warp}
  TAG=$(echo "${ROBOTS}_${NENVS}_${VARIANT}_${IMPL}" | tr ',' '+')
  OUT=/network/scratch/c/charlie.gauthier/urma_context/mjx/bench/${SLURM_JOB_ID}/$TAG
  echo "### $spec"
  mkdir -p $(dirname $OUT); nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 2 > $OUT.gpu.csv &
  SMI=$!
  timeout 2400 /network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python \
    /network/scratch/c/charlie.gauthier/urma_context/urma-mjx-wt/one_policy_to_run_them_all/mjx/train_mjx.py \
    --robots $ROBOTS --nr_envs $NENVS --model_variant $VARIANT --impl $IMPL --bench_iters 6 --out $OUT 2>&1 \
    | grep -E '^\{"robots|compile_plus|"iter": [56],|Error|error:|Traceback|DONE' | cut -c1-600
  kill $SMI 2>/dev/null
  echo "peak_vram_MiB $(sort -n -t, -k1 $OUT.gpu.csv | tail -1 | cut -d, -f1)"
done
