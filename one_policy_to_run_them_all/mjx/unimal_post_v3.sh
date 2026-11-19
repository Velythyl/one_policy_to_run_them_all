#!/bin/bash
# Rollouts (fixed command vx=0.5) of the smoke checkpoints + GIFs, and the standing PNG grid.
P=/network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python
M=/network/scratch/c/charlie.gauthier/urma_context/urma-unimal-wt/one_policy_to_run_them_all/mjx
D=/network/scratch/c/charlie.gauthier/urma_context/unimal_probe
mkdir -p $D/png_v3
$P $M/unimal_render.py stand $D/png_v3/unimals_standing.png $(cat $D/pick10.txt)
for pair in V3:unimal_1409_11_11_01_08_26_42 V3:unimal_1409_7_6_01_12_11_58 V3:unimal_mvt_5506_4_9_17_09_31_55 V3B:unimal_mvt_5506_14_2_17_07_59_45 V3B:unimal_1409_8_2_01_16_01_05 V3B:unimal_vt_5506_7_17_02_15_44_51; do
  run=${pair%%:*}; r=${pair#*:}
  [ -f $D/runs/smoke_$run/checkpoint.pkl ] || continue
  $P $M/unimal_rollout.py $D/runs/smoke_$run/checkpoint.pkl $r $D/png_v3/${r}_vx05.npz 0.5 0 0 300 64
  $P $M/unimal_rollout.py $D/runs/smoke_$run/checkpoint.pkl $r $D/png_v3/${r}_zero.npz 0 0 0 300 64
  $P $M/unimal_render.py replay $D/png_v3/${r}_vx05.npz $r $D/png_v3/${r}_vx05.gif
done
