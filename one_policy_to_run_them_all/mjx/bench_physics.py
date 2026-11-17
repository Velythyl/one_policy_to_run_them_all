"""Physics-only throughput: vmapped control steps (nr_substeps mjx.step each) per robot / impl / nr_envs.
    python bench_physics.py <impl jax|warp> <robot> <nr_envs> [variant]
variant: native | feet_only (contacts restricted to floor-vs-(feet,trunk) via contype/conaffinity)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp
import mujoco
from mujoco import mjx

ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
CFG_ROOT = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs")


def load_model(robot, variant="native"):
    cfg = json.loads((CFG_ROOT / f"{robot}.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(ENV_ROOT / robot / "data" / cfg["xml_file"]))
    m.opt.timestep = cfg["timestep"]
    if variant == "feet_only":
        keep = set(cfg["collision_groups"]["floor"])
        for k, v in cfg["collision_groups"].items():
            if k.startswith("trunk") or "foot" in k or k == "feet" or k.startswith("hip"):
                keep |= set(v)
        for g in range(m.ngeom):
            if g not in keep:
                m.geom_contype[g] = 0; m.geom_conaffinity[g] = 0
    return m, cfg


def main():
    impl, robot, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    variant = sys.argv[4] if len(sys.argv) > 4 else "native"
    m, cfg = load_model(robot, variant)
    t0 = time.time()
    mx = mjx.put_model(m, impl=impl)
    d0 = mjx.make_data(m, impl=impl)
    qpos = np.array(m.qpos0); qpos[2] = cfg["initial_drop_height"]
    qi = np.array(cfg["qpos_joint_idx"], dtype=int); qpos[qi] = cfg["nominal_joint_positions"]
    d0 = d0.replace(qpos=jnp.asarray(qpos))
    keys = jax.random.split(jax.random.PRNGKey(0), n)
    nsub = cfg["nr_substeps"]

    def one(d, k):
        ctrl = 0.1 * jax.random.normal(k, (m.nu,))
        d = d.replace(ctrl=ctrl)
        d = jax.lax.fori_loop(0, nsub, lambda i, dd: mjx.step(mx, dd), d)
        return d

    @jax.jit
    def run(ds, keys):
        def body(carry, _):
            ds, keys = carry
            keys = jax.vmap(lambda k: jax.random.split(k)[0])(keys)
            ds = jax.vmap(one)(ds, keys)
            return (ds, keys), ds.qpos[:, 2].mean()
        (ds, keys), h = jax.lax.scan(body, (ds, keys), None, length=100)
        return ds, keys, h

    ds = jax.vmap(lambda _: d0)(jnp.arange(n))
    ds, keys, h = run(ds, keys); jax.block_until_ready(h)
    tc = time.time() - t0
    t1 = time.time()
    for _ in range(3):
        ds, keys, h = run(ds, keys)
    jax.block_until_ready(h)
    dt = time.time() - t1
    sps = 300 * n / dt
    ncon = int(np.asarray(getattr(ds, "ncon", jnp.zeros(1))).max()) if hasattr(ds, "ncon") else -1
    print(json.dumps({"impl": impl, "robot": robot, "variant": variant, "nr_envs": n, "compile_s": round(tc, 1),
                      "control_steps_per_s": round(sps), "mean_height_end": float(h[-1]),
                      "gpu": str(jax.devices()[0].device_kind)}), flush=True)


if __name__ == "__main__":
    main()
