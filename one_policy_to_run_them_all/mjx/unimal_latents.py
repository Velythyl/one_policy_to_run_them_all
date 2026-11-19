"""Campaign 6 latents: URMA robots (copied unchanged) + Unimals (encoded with the SAME v4 models, z-scored with the
SAME stats as the URMA latents). Writes <out>/<arm>/<robot>.json, normalization_config.json, meta.json.

    LD_LIBRARY_PATH=/network/scratch/c/charlie.gauthier/urma_context/lib \
    /home/mila/c/charlie.gauthier/voxvae/venv/bin/python unimal_latents.py <arm> <out_dir>      (arm: vme_sa | vmestar_sa)

Unimal scans (<unimal>-parsed.json next to each XML) are segmented PER JOINT (limbx/N, limby/N; ">NO_ACTUATORS<" =
torso). The output therefore has
  "joint:<joint name>" -> latent of that joint's scan part (per-joint lookup: 2 hinges on one body get their own latents),
  "<body name>"        -> latent of the body's first joint (load_latents needs one entry per body; trunk = >NO_ACTUATORS<).
A joint without its own scan part takes the other hinge of its body, else its nearest ancestor body's (listed in meta).
Check: the URMA robots' raw latents are recomputed with the same model/step/extent and compared to
latents_aux/<arm>_raw_latents.json (must agree), so the Unimal latents come from exactly the URMA latents' encoder.
"""
import glob
import json
import os
import random
import shutil
import sys
import types
from pathlib import Path

sys.path.insert(0, "/network/scratch/c/charlie.gauthier/urma_context/voxvae-urma")
sys.modules.setdefault("wandb", types.ModuleType("wandb"))
import mujoco
import numpy as np
import torch

S = Path("/network/scratch/c/charlie.gauthier/urma_context")
URMA_LAT = S / "mjx/frozen/9f019a1/latents"  # = the latents campaigns 3/4 trained with
UNIMALS = Path("/network/scratch/c/charlie.gauthier/unimals_100")
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"  # the URMA v4 latents were encoded on GPU (code/produce_v4.sbatch)
MODELS = {"vme_sa": (S / "models/vme_sa_v4", 3150), "vmestar_sa": (S / "models/vmestar_sa_v4_lr1e-4", 10700)}


def robot_name(xml_path):
    import re
    return "unimal_" + re.sub(r"[^0-9a-zA-Z]+", "_", Path(xml_path).stem.replace("floor-", ""))


def encode(model, cfg, extent, parsed):
    from voxvae.dataloading.dataloader import load_for_inference
    np.random.seed(0); random.seed(0); torch.manual_seed(0)  # same as produce_latents.py
    jp, rc, vox = load_for_inference([str(parsed)], cfg, global_bounds=extent)
    with torch.no_grad():
        lat = model.get_latent(vox.to(DEV).float()).cpu()
    return {c: l for c, l in zip(rc, lat.reshape(lat.shape[0], -1).numpy())}


def main():
    arm, out = sys.argv[1], Path(sys.argv[2]) / sys.argv[1]
    from voxvae.load import load_vma
    from voxvae.vma import load_wandbconfig_as_hydraconfig
    ck, step = MODELS[arm]
    cfg = load_wandbconfig_as_hydraconfig(str(ck))
    assert not getattr(cfg.dataloader, "normalize_01", True)
    extent = float(cfg.dataloader.global_extent)
    assert abs(extent - json.load(open(S / "global_extent_v3.json"))["global_extent"]) < 1e-9
    meta_urma = json.load(open(URMA_LAT / arm / "meta.json"))
    assert meta_urma["checkpoint"] == str(ck) and meta_urma["step"] == step, meta_urma
    model = load_vma(str(ck), step, cfg).to(DEV).eval()
    print("device", DEV, torch.cuda.get_device_name(0) if DEV != "cpu" else "", flush=True)
    norm = json.load(open(URMA_LAT / arm / "normalization_config.json"))
    mean, std = (np.asarray(v) for v in norm["normalization_vals"])

    # 1) same-encoder check on URMA robots
    raw_ref = json.load(open(S / f"latents_aux/{arm}_raw_latents.json"))
    maxdiff = {}
    for r in ("unitree_go1", "hexapod", "unitree_h1", "atlas"):
        got = encode(model, cfg, extent, S / f"scans/{r}/vme_input/{r}-parsed.json")
        maxdiff[r] = max(float(np.abs(got[b] - np.asarray(v)).max() / np.abs(np.asarray(v)).mean()) for b, v in raw_ref[r].items())
        # z-scored file must equal (raw - mean) / (std + 1e-8)
        z = json.load(open(URMA_LAT / arm / f"{r}.json"))
        for b, v in raw_ref[r].items():
            assert np.allclose(z[b], (np.asarray(v) - mean) / (std + 1e-8), atol=1e-6), (r, b)
    print("URMA re-encode max |diff| / mean |latent| vs stored raw:", maxdiff, flush=True)
    assert max(maxdiff.values()) < 0.02, maxdiff  # GPU TF32 convs: tiny run-to-run / GPU-type differences

    out.mkdir(parents=True, exist_ok=True)
    for f in sorted((URMA_LAT / arm).glob("*.json")):
        shutil.copy2(f, out / f.name)  # 20 URMA robots + normalization_config.json + meta.json (overwritten below)
    fills, raw_all, n = {}, {}, 0
    for split in ("train", "test"):
        for xml in sorted(glob.glob(str(UNIMALS / split / "xml" / "*.xml"))):
            name = robot_name(xml)
            if (out / f"{name}.json").is_file() and name in raw_all:
                continue  # train/test duplicate (byte-identical)
            raw = encode(model, cfg, extent, xml.replace(".xml", "-parsed.json"))
            m = mujoco.MjModel.from_xml_path(xml)
            bname = lambda b: "trunk" if b == 1 else mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            res, fill = {}, {}
            jl = {}
            for j in range(m.njnt):
                if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                    continue
                jn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                src = jn
                if src not in raw:  # other hinge on the same body, else nearest ancestor body with a scanned joint
                    b = m.jnt_bodyid[j]
                    while True:
                        cand = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k) for k in range(m.njnt)
                                if m.jnt_bodyid[k] == b and mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k) in raw]
                        if cand or b <= 1:
                            src = cand[0] if cand else ">NO_ACTUATORS<"
                            break
                        b = m.body_parentid[b]
                    fill[jn] = src
                jl[jn] = raw[src]
                res["joint:" + jn] = ((raw[src] - mean) / (std + 1e-8)).astype(float).tolist()
            for b in range(1, m.nbody):
                js = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, k) for k in range(m.njnt) if m.jnt_bodyid[k] == b and m.jnt_type[k] == mujoco.mjtJoint.mjJNT_HINGE]
                v = raw[">NO_ACTUATORS<"] if b == 1 else jl[js[0]]
                res[bname(b)] = ((v - mean) / (std + 1e-8)).astype(float).tolist()
            extra = set(raw) - set(jl) - {">NO_ACTUATORS<"}
            assert not extra, (name, extra)
            (out / f"{name}.json").write_text(json.dumps(res))
            raw_all[name] = {k: v.astype(float).tolist() for k, v in raw.items()}
            if fill:
                fills[name] = fill
            n += 1
            if n % 20 == 0:
                print(n, name, flush=True)
    allz = np.stack([np.asarray(v) for r in raw_all for k, v in json.load(open(out / f"{r}.json")).items() if k.startswith("joint:")])
    meta = dict(meta_urma, applied_to="20 URMA robots (copied from " + str(URMA_LAT / arm) + ") + %d Unimals (unimals_100 train+test)" % n,
                unimal_lookup="per joint: keys 'joint:<joint name>' (mjx_env uses them when present); body keys = first joint's latent",
                unimal_filled_joints=fills, unimal_zscore_stats={"mean_abs_mean": float(np.abs(allz.mean(0)).mean()), "std_mean": float(allz.std(0).mean())},
                urma_reencode_maxdiff=maxdiff)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out.parent / f"{arm}_unimal_raw_latents.json").write_text(json.dumps(raw_all))
    print(arm, "done:", n, "Unimals; filled joints in", len(fills), "Unimals; z-scored Unimal joint latents: |mean| avg %.3f, std avg %.3f"
          % (meta["unimal_zscore_stats"]["mean_abs_mean"], meta["unimal_zscore_stats"]["std_mean"]), flush=True)


if __name__ == "__main__":
    main()
