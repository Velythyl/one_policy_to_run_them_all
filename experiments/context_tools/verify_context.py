"""Verify the per-limb context arms on every robot (real envs, no pytest).

    python verify_context.py <stock_npz> <latent_dir> [out_npz]

For every robot x arm (default pose and 3 DR samples, seed 0):
  * widths: equal within an arm family across all robots; foot width 10 everywhere
  * K / D / LAT / G column sets disjoint; switched-off blocks exactly zero
  * empty: joint AND foot tokens all zero (G copies zeroed too); other arms keep G
  * latent arms (every <latent_dir>/<arm>/ dir: vme, vmestar, matme, vme_sa, vmestar_sa, ...): only latent + D (+ G) columns non-zero; latent equals the file's latent
  * dk restricted to URMA's fields == stock URMA values (bitwise), G == stock G,
    D link fields recomputed after DR (seen values), and urma arm == stock
  * no two arms identical for a robot
Prints the field tables for unitree_a1 and atlas.
"""
import sys
import numpy as np
from envs import build_env, ALL_ROBOTS
from one_policy_to_run_them_all.context import limb_context as lc

stock = np.load(sys.argv[1])
latent_dir = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else None
ARMS = ("urma",) + lc.NON_LATENT_ARMS + tuple(sorted(p.name for p in __import__("pathlib").Path(latent_dir).iterdir() if p.is_dir() and lc.is_latent_arm(p.name)))
LAT_ARMS = [a for a in ARMS if lc.is_latent_arm(a)]
dump = {}
widths = {}
stock_layout = lc.joint_layout("urma")


def descr(env):
    return (np.stack([env.name_to_description_vector[j] for j in env.joint_names]),
            np.stack([env.name_to_description_vector[f] for f in env.foot_names]))


def dr_sample(env):
    env.domain_randomization_unseen_robot_function.sample()
    env.domain_randomization_seen_robot_function.sample()
    env.domain_randomization_mujoco_model_function.sample()
    env.name_to_description_vector = env.get_name_to_description_vector()


def print_table(robot, env, mats):
    print(f"\n=== {robot}: joint-token field table (limb 0 = {env.joint_names[0]} / body {env._limb_context_static['body_names'][0]}) ===")
    lay = lc.joint_layout("dk")
    print(f"{'block':5s} {'field':16s} {'cols(dk)':9s} " + " ".join(f"{a:>7s}" for a in ("dk", "k", "d", "empty")) + "   dk value(s) limb 0")
    for block, f, s, e in lay.columns:
        on = " ".join(f"{'x' if block in lc.ARM_BLOCKS_ON[a] else '.':>7s}" for a in ("dk", "k", "d", "empty"))
        vals = np.array2string(mats["dk"][0][0, s:e], precision=3, suppress_small=True, max_line_width=200)
        print(f"{block:5s} {f:16s} {s:3d}:{e:<5d} {on}   {vals}")
    for la in LAT_ARMS:
        ll = lc.joint_layout(la, mats[la][0].shape[1] - lc.D_WIDTH - lc.G_WIDTH)
        print(f"{la}: " + ", ".join(f"{b}:{f}[{s}:{e}]" for b, f, s, e in ll.columns if b != "D"))
    print(f"foot token (all arms): " + ", ".join(f"{b}:{f}[{s}:{e}]" for b, f, s, e in lc.FOOT_LAYOUT.columns)
          + "; foot_pos zeroed in arms without K")


for robot in ALL_ROBOTS:
    mats = {}
    envs = {}
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
        widths.setdefault(arm, set()).add((J.shape[1], F.shape[1]))
        assert F.shape[1] == 10, (robot, arm, F.shape)

    # stock reproduction: urma arm bitwise, dk restricted to URMA fields bitwise, every sample
    for s_i in range(4):
        Js = np.stack([stock[f"{robot}/{s_i}/{j}"] for j in envs["urma"][0].joint_names])
        Fs = np.stack([stock[f"{robot}/{s_i}/{f}"] for f in envs["urma"][0].foot_names])
        Ju, Fu = envs["urma"][1][s_i]
        assert np.array_equal(Ju, Js) and np.array_equal(Fu, Fs), (robot, "urma != stock", s_i)
        Jdk, Fdk = envs["dk"][1][s_i]
        dk_lay = lc.joint_layout("dk")
        for _, f, s, e in stock_layout.columns:
            assert np.array_equal(Jdk[:, dk_lay.field_slice(f)], Js[:, s:e]), (robot, s_i, f, "dk != stock")
        assert np.array_equal(Fdk, Fs), (robot, s_i, "dk feet != stock")

    # D recomputed under DR: torque limit (always resampled) must change across samples
    dk_lay = lc.joint_layout("dk")
    tl = [envs["dk"][1][s][0][:, dk_lay.field_slice("torque_limit")] for s in range(4)]
    assert not np.array_equal(tl[0], tl[1]), (robot, "D not recomputed after DR")

    for arm in ARMS:
        if arm == "urma":
            continue
        env, per_sample = envs[arm]
        latent_dim = per_sample[0][0].shape[1] - lc.D_WIDTH - lc.G_WIDTH if lc.is_latent_arm(arm) else 0
        lay = lc.joint_layout(arm, latent_dim)
        blocks = {b: set(lay.block_columns(b).tolist()) for b in ("K", "D", "LAT", "G")}
        allc = [c for b in blocks.values() for c in b]
        assert len(allc) == len(set(allc)) == lay.width, (robot, arm, "blocks not disjoint / not covering")
        for J, F in per_sample:
            for b, cols in blocks.items():
                if not cols:
                    continue
                sub = J[:, sorted(cols)]
                if b in lc.ARM_BLOCKS_ON[arm]:
                    assert np.any(sub), (robot, arm, b, "switched-on block is zero")
                else:
                    assert not np.any(sub), (robot, arm, b, "switched-off block non-zero")
            if "K" not in lc.ARM_BLOCKS_ON[arm]:
                assert not np.any(F[:, :3]), (robot, arm, "foot pos must be zero without K")
            if "G" not in lc.ARM_BLOCKS_ON[arm]:
                assert not np.any(F[:, 3:]), (robot, arm, "foot G must be zero")
            else:
                assert np.any(F[:, 3:]), (robot, arm, "foot G missing")
        if arm == "empty":
            for J, F in per_sample:
                assert not np.any(J) and not np.any(F), (robot, "empty tokens not all zero")
        if lc.is_latent_arm(arm):
            lat = lc.load_latents(latent_dir, arm, lc.robot_dir_name(env), env.model)
            want = np.stack([lat[b] for b in env._limb_context_static["body_names"]])
            assert np.array_equal(per_sample[0][0][:, lay.block_columns("LAT")], want), (robot, arm, "latent mismatch")
    # G columns identical across arms (default sample) and equal to stock's
    Js = np.stack([stock[f"{robot}/0/{j}"] for j in envs["urma"][0].joint_names])
    g_stock = Js[:, stock_layout.block_columns("G")]
    for arm in ARMS:
        if arm in ("urma", "empty"):
            continue
        lat_dim = mats[arm][0].shape[1] - lc.D_WIDTH - lc.G_WIDTH if lc.is_latent_arm(arm) else 0
        assert np.array_equal(mats[arm][0][:, lc.joint_layout(arm, lat_dim).block_columns("G")], g_stock), (robot, arm, "G differs")
    # no two arms identical
    for i, a in enumerate(ARMS):
        for b in ARMS[i + 1:]:
            assert not (mats[a][0].shape == mats[b][0].shape and np.array_equal(mats[a][0], mats[b][0])), (robot, a, b, "identical")
    n = len(envs["dk"][0].joint_names)
    print(f"{robot:14s} OK  limbs={n:2d}  widths " + " ".join(f"{a}={mats[a][0].shape[1]}" for a in ARMS), flush=True)
    if robot in ("unitree_a1", "atlas"):
        print_table(robot, envs["dk"][0], mats)
    for env, _ in envs.values():
        env.close()

for arm, w in widths.items():
    assert len(w) == 1, (arm, "width differs across robots", w)
fam = {widths[a].copy().pop()[0] for a in lc.NON_LATENT_ARMS}
assert len(fam) == 1, ("non-latent family widths differ", fam)
fam = {widths[a].copy().pop()[0] for a in LAT_ARMS}
assert len(fam) == 1, ("latent family widths differ", fam)
print("\nALL CHECKS PASSED; joint widths:", {a: widths[a].copy().pop()[0] for a in ARMS})
if out:
    np.savez_compressed(out, **dump)
    print("dumped", out)
