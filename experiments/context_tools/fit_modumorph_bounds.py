"""Fit ModuMorph min-max bounds for the URMA per-limb K / D / P fields over all 20 robots
(nominal model + N DR samples of the seen values, 2% span padding), write
one_policy_to_run_them_all/context/modumorph_bounds.json.   python fit_modumorph_bounds.py [N=40]"""
import json, sys
import numpy as np
from envs import build_env, ALL_ROBOTS
from one_policy_to_run_them_all.context import limb_context as lc

assert not lc.MODUMORPH_BOUNDS_FILE.is_file(), "remove the old bounds file first (fitting needs raw fields)"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
vals = {}
for robot in ALL_ROBOTS:
    env = build_env(robot, seed=0, context_arm="dk")
    for k in range(N + 1):
        if k:
            env.domain_randomization_unseen_robot_function.sample()
            env.domain_randomization_seen_robot_function.sample()
            env.domain_randomization_mujoco_model_function.sample()
            env.get_name_to_description_vector()
        raw, hg = env._limb_context_raw, env._limb_context_static["has_geom"]
        for f, v in raw.items():
            v = v[hg] if f in ("geom_quat", "body_shape") else v
            vals.setdefault(f, []).append(v)
    print(robot, flush=True)
    env.close()
bounds = {}
for f, vs in vals.items():
    v = np.concatenate(vs)
    lo, hi = v.min(0), v.max(0)
    pad = 0.02 * (hi - lo)
    if f in ("body_iquat", "geom_quat", "jnt_pos", "body_pos", "body_ipos", "joint_axis", "body_shape", "body_parent"):
        pad = np.zeros_like(pad)  # static model geometry: exact range
    bounds[f] = [(lo - pad).round(6).tolist(), (hi + pad).round(6).tolist()]
    print(f, bounds[f])
lc.MODUMORPH_BOUNDS_FILE.write_text(json.dumps({"doc": "ModuMorph min-max bounds refit over the 20 URMA robots "
    f"(nominal + {N} DR samples of seen values, 2% padding on DR'd fields); x' = -1*(lo!=hi) + 2(x-lo)/(hi-lo+1e-8)",
    "robots": list(ALL_ROBOTS), "bounds": bounds}, indent=1))
print("wrote", lc.MODUMORPH_BOUNDS_FILE)
