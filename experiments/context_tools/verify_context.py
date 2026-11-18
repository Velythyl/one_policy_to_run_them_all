"""Verify the per-limb context arms (context version stock_superset_v2) on every robot (real envs, no pytest).

    python verify_context.py <stock_npz> <latent_dir> [out_npz]

For every robot x arm (default pose and 3 DR samples, seed 0):
  * urma arm == stock (bitwise, joints and feet, every sample)
  * every arm except empty: joint token[:, :23] == urma arm's joint token (bitwise, every sample);
    foot token == stock foot token (bitwise)
  * empty: joint AND foot tokens all zero, width 23
  * extras (columns 23:) = exactly the arm's extras layout; every extras field present (not all zero) per robot;
    latent columns == the latent file; body_mass / K fields / body_parent vary across limbs (pooled over robots)
  * joint width constant per arm across robots
  * no two arms identical for a robot (full token; and extras of equal-width arms differ)
Prints the field table for unitree_a1 and atlas.
"""
import sys
from pathlib import Path

import numpy as np
from envs import build_env, ALL_ROBOTS
from one_policy_to_run_them_all.context import limb_context as lc

stock = np.load(sys.argv[1])
latent_dir = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else None
LAT_ARMS = tuple(sorted(p.name for p in Path(latent_dir).iterdir() if p.is_dir() and lc.is_latent_arm(p.name)))
ARMS = ("urma",) + lc.NON_LATENT_ARMS + LAT_ARMS
SW = lc.STOCK_WIDTH
assert SW == 23 and lc.stock_layout().width == 23
dump, widths = {}, {}
pooled = {}  # (arm, field) -> list of per-limb rows (default sample), for the variation check


def descr(env):
    return (np.stack([env.name_to_description_vector[j] for j in env.joint_names]),
            np.stack([env.name_to_description_vector[f] for f in env.foot_names]))


def dr_sample(env):
    env.domain_randomization_unseen_robot_function.sample()
    env.domain_randomization_seen_robot_function.sample()
    env.domain_randomization_mujoco_model_function.sample()
    env.name_to_description_vector = env.get_name_to_description_vector()


def print_table(robot, env, mats):
    print(f"\n=== {robot}: joint-token layout (context {lc.CONTEXT_VERSION}; limb 0 = {env.joint_names[0]}) ===")
    for a in ARMS:
        J = mats[a][0]
        lay = lc.joint_layout(a, lc.latent_dim_from_width(a, J.shape[1]))
        print(f"-- {a}: width {lay.width}, foot 10" + (" (all zeros)" if a == "empty" else ""))
        if a in ("dk", "urma") or lc.is_latent_arm(a) and a == LAT_ARMS[0]:
            for block, f, s, e in lay.columns:
                vals = np.array2string(J[0, s:e], precision=3, suppress_small=True, max_line_width=200)
                print(f"   {block:3s} {f:22s} {s:3d}:{e:<3d} {vals}")
        else:
            print("   " + ", ".join(f"{b}:{f}[{s}:{e}]" for b, f, s, e in lay.columns if s >= SW) or "   (no extras)")
    print("foot token: " + ", ".join(f"{b}:{f}[{s}:{e}]" for b, f, s, e in lc.FOOT_LAYOUT.columns) + "; zero only in empty")


for robot in ALL_ROBOTS:
    mats, envs = {}, {}
    for arm in ARMS:
        env = build_env(robot, seed=0, context_arm=arm, latent_dir=latent_dir)
        env.multi_robot_max_observation_size = -1
        per_sample = [descr(env)]
        for _ in range(3):
            dr_sample(env)
            per_sample.append(descr(env))
        mats[arm] = per_sample[0]
        envs[arm] = (env, per_sample)
        for s_i, (J, F) in enumerate(per_sample):
            dump[f"{robot}/{arm}/{s_i}/joints"] = J
            dump[f"{robot}/{arm}/{s_i}/feet"] = F
        widths.setdefault(arm, set()).add(J.shape[1])
        assert all(F.shape[1] == 10 for _, F in per_sample), (robot, arm)
        assert env.dynamic_joint_description_size == J.shape[1], (robot, arm)

    names = envs["urma"][0].joint_names
    feet = envs["urma"][0].foot_names
    for s_i in range(4):
        Js = np.stack([stock[f"{robot}/{s_i}/{j}"] for j in names])
        Fs = np.stack([stock[f"{robot}/{s_i}/{f}"] for f in feet])
        Ju, Fu = envs["urma"][1][s_i]
        assert Ju.shape[1] == SW and np.array_equal(Ju, Js) and np.array_equal(Fu, Fs), (robot, "urma != stock", s_i)
        for arm in ARMS:
            J, F = envs[arm][1][s_i]
            if arm == "empty":
                assert J.shape[1] == SW and not np.any(J) and not np.any(F), (robot, s_i, "empty not all zero")
                continue
            assert np.array_equal(J[:, :SW], Ju), (robot, arm, s_i, "STOCK prefix != urma token")
            assert np.array_equal(F, Fs), (robot, arm, s_i, "foot token != stock")

    # STOCK D fields recomputed under DR (torque limit always resampled)
    tl = lc.stock_layout().field_slice("urma_torque_limit")
    assert not np.array_equal(envs["urma"][1][0][0][:, tl], envs["urma"][1][1][0][:, tl]), (robot, "D not recomputed after DR")

    for arm in ARMS:
        if arm in ("urma", "empty"):
            continue
        env, per_sample = envs[arm]
        J0 = per_sample[0][0]
        lay = lc.joint_layout(arm, lc.latent_dim_from_width(arm, J0.shape[1]))
        ext = lc.extras_layout(arm, lc.latent_dim_from_width(arm, J0.shape[1]))
        assert lay.width == J0.shape[1] and ext.columns and ext.columns[0][2] == SW and ext.width == lay.width, (robot, arm)
        assert lay.columns[:len(lc.stock_layout().columns)] == lc.stock_layout().columns, (robot, arm, "stock layout prefix")
        for block, f, s, e in ext.columns:
            for J, _ in per_sample:
                assert np.all(np.isfinite(J[:, s:e])), (robot, arm, f, "non-finite")
                assert np.any(J[:, s:e]), (robot, arm, f, "extras field all zero")
            pooled.setdefault((arm, f), []).append(J0[:, s:e])
        if lc.is_latent_arm(arm):
            lat = lc.load_latents(latent_dir, arm, lc.robot_dir_name(env), env.model)
            want = np.stack([lat[b] for b in env._limb_context_static["body_names"]])
            assert np.array_equal(J0[:, ext.block_columns("LAT")], want), (robot, arm, "latent mismatch")
        expect = {"d": {"D", "P"}, "k": {"K", "P"}, "dk": {"K", "D", "P"}}.get(arm, {"LAT", "P"})
        assert {b for b, *_ in ext.columns} == expect, (robot, arm, "extras blocks")

    # no two arms identical (full token), and equal-width arms differ in their extras
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            A, B = mats[a][0], mats[b][0]
            if A.shape == B.shape:
                assert not np.array_equal(A, B), (robot, a, b, "identical")
                if a not in ("urma", "empty") and b not in ("urma", "empty"):
                    assert not np.array_equal(A[:, SW:], B[:, SW:]), (robot, a, b, "identical extras")
    n = len(names)
    print(f"{robot:14s} OK  limbs={n:2d}  widths " + " ".join(f"{a}={mats[a][0].shape[1]}" for a in ARMS), flush=True)
    if robot in ("unitree_a1", "atlas"):
        print_table(robot, envs["urma"][0], mats)
    for env, _ in envs.values():
        env.close()

for arm, w in widths.items():
    assert len(w) == 1, (arm, "width differs across robots", w)
print("\nextras field statistics over all limbs of all robots (default sample):")
for (arm, f), rows in sorted(pooled.items()):
    X = np.concatenate(rows)
    std = X.std(0)
    print(f"  {arm:11s} {f:22s} n={len(X):3d} min {X.min():+.3f} max {X.max():+.3f} per-col std {np.array2string(std, precision=3, max_line_width=200)}")
    assert np.all(np.isfinite(X)) and std.max() > 1e-3, (arm, f, "degenerate (constant over all limbs)")
print(f"\nALL CHECKS PASSED (context {lc.CONTEXT_VERSION}); joint widths:", {a: widths[a].copy().pop() for a in ARMS})
if out:
    np.savez_compressed(out, **dump)
    print("dumped", out)
