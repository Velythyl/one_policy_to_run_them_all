#!/bin/bash
# Chained, resumable URMA MJX training. Each job trains until MAX_WALL_S, checkpoints, exits 3 and
# resubmits itself with the same arguments; the next job resumes from <out>/checkpoint.pkl.
# A finished run writes <out>/DONE. Stop a chain: touch <out>/STOP (no further resubmission).
#   sbatch --job-name=urma-mjx-<tag> --time=2-00:00:00 train_chain.sbatch --out <run_dir> [train_mjx.py args...]
#   MAX_WALL_S defaults to the job's time limit minus 20 min.
# GPU recipe deviations (train_mjx.py defaults, logged as "recipe_deviations_gpu" + <out>/config-<job>.json):
#   --solimp_dmax_clamp 0.99, --physics_overrides '{"impratio_adapt_mu0": 0.3}' (per-world impratio lowered only for
#   DR tangential mu < 0.3; model identical to upstream for mu >= 0.3); --friction_tangential_floor 0 (upstream DR range).
# wandb (one_policy_to_run_them_all/mjx/wandb_logger.py; ModuMorph-wandbutils-compatible config + metric names):
#   WANDB_MODE online (default) | offline (sync later: wandb sync <out>/wandb/offline-run-*) | disabled; URMA_WANDB=0 also disables.
#   URMA_WANDB_ENTITY (default velythyl), URMA_WANDB_PROJECT (default urma_context). Key: WANDB_API_KEY or ~/.netrc.
#   The run id lives in <out>/wandb_run_id, so every chained job resumes the same wandb run. wandb errors never stop training.
#SBATCH --partition=unkillable
#SBATCH --gres=gpu:1
#SBATCH --constraint=ampere|lovelace
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/network/scratch/c/charlie.gauthier/urma_context/jobs/%x-%j.out
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 MUJOCO_GL=egl
# URMA_MJX_CODE: code tree to run (default: the mjx-spike worktree). For real runs point it at a frozen
# checkout so later edits never change a running chain; it is exported to every resubmitted job.
export URMA_MJX_CODE=${URMA_MJX_CODE:-/network/scratch/c/charlie.gauthier/urma_context/urma-mjx-wt}
export PYTHONPATH=$URMA_MJX_CODE
export WARP_CACHE_PATH=/network/scratch/c/charlie.gauthier/urma_context/mjx/warp_cache JAX_COMPILATION_CACHE_DIR=/network/scratch/c/charlie.gauthier/urma_context/mjx/jax_cache
export XDG_CACHE_HOME=/network/scratch/c/charlie.gauthier/urma_context/mjx/xdg_cache
OUT=""; prev=""
for x in "$@"; do [ "$prev" = "--out" ] && OUT="$x"; prev="$x"; done
[ -n "$OUT" ] || { echo "need --out"; exit 1; }
LIMIT_S=$(squeue -h -j "$SLURM_JOB_ID" -o %L | awk -F'[-:]' '{ if (NF==4) print ($1*86400+$2*3600+$3*60+$4); else if (NF==3) print ($1*3600+$2*60+$3); else print ($1*60+$2) }')
MAX_WALL_S=${MAX_WALL_S:-$((LIMIT_S - 1200))}
nvidia-smi -L
mkdir -p "$OUT"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader,nounits -l 30 > "$OUT/gpu-${SLURM_JOB_ID}.csv" &
/network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python \
  $URMA_MJX_CODE/one_policy_to_run_them_all/mjx/train_mjx.py \
  --max_wall_s "$MAX_WALL_S" "$@" 2>&1 | grep --line-buffered -v -E "solver iterations limit reached|To disable the print warning"
rc=${PIPESTATUS[0]}  # (Warp prints one iteration-limit warning per step when any world hits 100 Newton iterations)
echo "EXIT $rc"
if [ $rc -eq 3 ] && [ ! -e "$OUT/STOP" ] && [ ! -e "$OUT/DONE" ]; then
  sbatch --job-name="$SLURM_JOB_NAME" --partition="$SLURM_JOB_PARTITION" --time="$(squeue -h -j $SLURM_JOB_ID -o %l)" \
    --export=ALL,URMA_MJX_CODE=$URMA_MJX_CODE $URMA_MJX_CODE/one_policy_to_run_them_all/mjx/train_chain.sbatch "$@"
fi
