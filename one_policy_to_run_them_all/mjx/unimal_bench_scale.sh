#!/bin/bash
# Scale bench: 16 URMA train robots + first N Unimals of train100.txt, ENVS envs/robot, 16 steps, ITERS iterations.
# usage (inside unimal_run.sbatch env): bash bench_scale.sh N ENVS [ITERS=4]
N=$1; ENVS=$2; ITERS=${3:-4}
U=$(head -n $N /network/scratch/c/charlie.gauthier/urma_context/unimal_probe/train100.txt | paste -sd,)
R=unitree_a1,unitree_go1,unitree_go2,anymal_b,anymal_c,barkour_v0,barkour_vb,badger,bittle,unitree_h1,unitree_g1,talos,robotis_op3,nao_v5,cassie,hexapod
[ "$N" -gt 0 ] && R=$R,$U
export URMA_WANDB=0
/usr/bin/time -v /network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python \
  /network/scratch/c/charlie.gauthier/urma_context/urma-unimal-wt/one_policy_to_run_them_all/mjx/train_mjx.py \
  --robots $R --eval_robots "" --nr_envs $ENVS --nr_steps 16 --bench_iters $ITERS \
  --out /network/scratch/c/charlie.gauthier/urma_context/unimal_probe/runs/bench_16p${N}_${ENVS}
