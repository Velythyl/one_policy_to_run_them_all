"""Plot each limb's fitted capsule (wireframe) over its geometry points, world frame at the nominal pose.
    python plot_capsules.py <out_dir> robot [robot ...]"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from envs import build_env
from one_policy_to_run_them_all.context import limb_context as lc

out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
for robot in sys.argv[2:]:
    env = build_env(robot, seed=0, context_arm="dk")
    m, st = env.model, env._limb_context_static
    d = mujoco.MjData(m); mujoco.mj_forward(m, d)
    fig = plt.figure(figsize=(14, 7))
    for k, (elev, azim) in enumerate(((15, -60), (0, 0))):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        for i, b in enumerate(st["bodies"]):
            R, x = d.xmat[b].reshape(3, 3), d.xpos[b]
            pts = lc.body_capsule_points(m, b)
            if len(pts):
                w = pts @ R.T + x
                sel = w[:: max(1, len(w) // 300)]
                ax.scatter(*sel.T, s=1, c="0.6", alpha=0.4)
            if not st["has_geom"][i]:
                continue
            r, h = st["capsule_shape"][i]
            c = st["capsule_pos"][i]
            M = np.zeros(9); mujoco.mju_quat2Mat(M, st["capsule_quat"][i]); a = M.reshape(3, 3)[:, 2]
            a = R @ a; c = R @ c + x
            t = np.linspace(0, 2 * np.pi, 16)
            perp1 = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0]); perp1 /= np.linalg.norm(perp1); perp2 = np.cross(a, perp1)
            col = plt.cm.tab20(i % 20)
            for z in np.linspace(-h, h, 4):
                ring = c + z * a + r * (np.outer(np.cos(t), perp1) + np.outer(np.sin(t), perp2))
                ax.plot(*ring.T, color=col, lw=0.8)
            for ph in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
                off = r * (np.cos(ph) * perp1 + np.sin(ph) * perp2)
                ax.plot(*np.stack([c - (h + r) * a * 0 - h * a + off, c + h * a + off]).T, color=col, lw=0.8)
        allp = d.xpos[1:]
        mid, span = allp.mean(0), (allp.max(0) - allp.min(0)).max() / 2 + 0.1
        ax.set_xlim(mid[0] - span, mid[0] + span); ax.set_ylim(mid[1] - span, mid[1] + span); ax.set_zlim(mid[2] - span, mid[2] + span)
        ax.view_init(elev, azim); ax.set_title(f"{robot} limb capsules (grey = geometry points)")
    fig.tight_layout(); fig.savefig(out / f"{robot}_capsules.png", dpi=110); plt.close(fig)
    print("wrote", out / f"{robot}_capsules.png", flush=True)
    env.close()
