#!/bin/bash
# Learnability smoke: bash train_smoke.sh TAG ROBOTS(comma) TOTAL_STEPS [ENVS=4096]
TAG=$1; R=$2; TOT=$3; ENVS=${4:-4096}
export URMA_WANDB=0
/usr/bin/time -v /network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python \
  /network/scratch/c/charlie.gauthier/urma_context/urma-unimal-wt/one_policy_to_run_them_all/mjx/train_mjx.py \
  --robots $R --eval_robots "" --nr_envs $ENVS --nr_steps 16 --total_steps $TOT \
  --out /network/scratch/c/charlie.gauthier/urma_context/unimal_probe/runs/smoke_$TAG
