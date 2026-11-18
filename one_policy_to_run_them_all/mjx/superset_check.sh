# Context stock_superset_v2 (2026-11-18): GPU==CPU descriptions for all 20 robots (all arms incl. dummy-latent arms,
# STOCK prefix == urma token), then 16-robot smokes (default physics, 2048x16, 10 it + 1 eval) for urma, vme_sa, dk
# with wandb online to project urma_context_smoke.
P=/network/scratch/c/charlie.gauthier/urma_context/venv_mjx/bin/python
M=/network/scratch/c/charlie.gauthier/urma_context/urma-mjx-wt/one_policy_to_run_them_all/mjx
$P $M/check_env.py warp unitree_a1 unitree_go1 unitree_go2 anymal_b anymal_c barkour_v0 barkour_vb badger bittle unitree_h1 unitree_g1 talos robotis_op3 nao_v5 cassie hexapod honey_badger badger_locked atlas sea_snake 2>&1 | grep -E "descriptions match|per-env D|Error|assert|Traceback"
export URMA_WANDB_PROJECT=urma_context_smoke WANDB_MODE=online
for arm in urma vme_sa dk; do
  echo "=== smoke $arm"
  LAT=""; [ "$arm" = vme_sa ] && LAT="--latent_dir /network/scratch/c/charlie.gauthier/urma_context/latents"
  $P $M/train_mjx.py --nr_envs 2048 --nr_steps 16 --total_steps 5242880 --eval_every 10 --context_arm $arm $LAT \
    --out /network/scratch/c/charlie.gauthier/urma_context/mjx/runs/smoke16_2048x16_${arm}_superset_v2 2>&1 | grep -E "recipe_dev|wandb_logger|compile|\"iter\"|Error|Traceback|DONE|nan|NaN" | cut -c1-400
done
