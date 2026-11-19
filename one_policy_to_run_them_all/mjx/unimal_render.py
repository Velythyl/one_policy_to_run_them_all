"""Render converted Unimals standing (MuJoCo C, PD hold at the nominal pose for 3 s from the drop pose) as one PNG grid,
and optionally replay recorded qpos trajectories (npz with key qpos (T, nq)) as GIFs.
    MUJOCO_GL=egl python unimal_render.py stand <out.png> robot [robot ...]
    MUJOCO_GL=egl python unimal_render.py replay <traj.npz> <robot> <out.gif>
"""
import json
import sys

import mujoco
import numpy as np
from PIL import Image, ImageDraw

from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.unimal_convert import settle

W, H = 320, 240


def camera(m, d):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = d.qpos[0:3] * [1, 1, 0.5]
    cam.distance, cam.azimuth, cam.elevation = 2.6, 135, -20
    return cam


def stand(out, robots):
    tiles = []
    for r in robots:
        c = load_spec(r).cfg
        m = mujoco.MjModel.from_xml_path(c["xml_path"])
        nom = np.asarray(c["nominal_joint_positions"])
        _, _, d = settle(m, nom, c["params"]["control"]["p_gain"], c["params"]["control"]["d_gain"], c["initial_drop_height"])
        with mujoco.Renderer(m, H, W) as ren:
            ren.update_scene(d, camera(m, d))
            img = Image.fromarray(ren.render())
        ImageDraw.Draw(img).text((5, 5), f"{r[7:]}\nn={len(nom)} feet={len(c['foot_names'])} z={d.qpos[2]:.2f}", fill=(255, 255, 255))
        tiles.append(img)
    cols = min(5, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    grid = Image.new("RGB", (cols * W, rows * H))
    for i, t in enumerate(tiles):
        grid.paste(t, ((i % cols) * W, (i // cols) * H))
    grid.save(out)


def replay(traj, r, out, every=2):
    c = load_spec(r).cfg
    m = mujoco.MjModel.from_xml_path(c["xml_path"])
    d = mujoco.MjData(m)
    q = np.load(traj)["qpos"]
    frames = []
    with mujoco.Renderer(m, H, W) as ren:
        for t in range(0, len(q), every):
            d.qpos[:] = q[t]
            mujoco.mj_forward(m, d)
            ren.update_scene(d, camera(m, d))
            frames.append(Image.fromarray(ren.render()))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=40 * every, loop=0)


if __name__ == "__main__":
    if sys.argv[1] == "stand":
        stand(sys.argv[2], sys.argv[3:])
    else:
        replay(sys.argv[2], sys.argv[3], sys.argv[4])
