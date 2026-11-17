"""Fitted limb capsules for every robot (raw metres): ranges + per-robot table -> json."""
import json, sys
import numpy as np
from envs import build_env, ALL_ROBOTS
from one_policy_to_run_them_all.context import limb_context as lc

out = {}
for robot in ALL_ROBOTS:
    env = build_env(robot, seed=0, context_arm="dk")
    st = env._limb_context_static
    out[robot] = {n: {"quat": q.round(4).tolist(), "radius": float(sh[0]), "half_length": float(sh[1]), "pos": p.round(4).tolist(), "geom": bool(h)}
                  for n, q, sh, p, h in zip(st["body_names"], st["capsule_quat"], st["capsule_shape"], st["capsule_pos"], st["has_geom"])}
    r = st["capsule_shape"][st["has_geom"]]
    print(f"{robot:14s} limbs {len(st['bodies']):2d} nogeom {int((~st['has_geom']).sum())} radius [{r[:,0].min():.3f},{r[:,0].max():.3f}] half_len [{r[:,1].min():.3f},{r[:,1].max():.3f}] |pos|max {np.abs(st['capsule_pos']).max():.3f}", flush=True)
    env.close()
json.dump(out, open(sys.argv[1], "w"), indent=1)
