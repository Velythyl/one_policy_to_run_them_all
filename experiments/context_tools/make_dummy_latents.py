"""Write DUMMY per-body latents (random, deterministic) for testing vme / vmestar / matme wiring.
Output: <out_root>/{vme,vmestar,matme}/<robot_dir_name>.json. Never point this at the real latents dir."""
import json, sys, zlib
from pathlib import Path
import mujoco
import numpy as np
from envs import build_env, ALL_ROBOTS
from one_policy_to_run_them_all.context.limb_context import robot_dir_name

out_root = Path(sys.argv[1]); dim = int(sys.argv[2]) if len(sys.argv) > 2 else 16
assert "dummy" in out_root.as_posix(), "dummy latents must go to a path containing 'dummy'"
for arm, salt in (("vme", 1), ("matme", 2), ("vmestar", 3)):
    (out_root / arm).mkdir(parents=True, exist_ok=True)
    for name in ALL_ROBOTS:
        env = build_env(name)
        m = env.model
        lat = {}
        for b in range(1, m.nbody):
            bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
            rng = np.random.default_rng(zlib.crc32(f"{arm}/{name}/{bn}".encode()) + salt)
            lat[bn] = np.round(rng.uniform(-1, 1, dim), 6).tolist()
        (out_root / arm / f"{robot_dir_name(env)}.json").write_text(json.dumps(lat))
        env.close()
    print(arm, "done")
