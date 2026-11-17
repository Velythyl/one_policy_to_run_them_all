"""Side-by-side learning curves: MJX/Warp runs (log.jsonl) vs the CPU URMA baseline (TensorBoard).
Run with the CPU venv (has tensorboard + matplotlib):
    python compare_runs.py <out.png> <cpu_tb_dir or -> <mjx_run_dir>[=label] ...
x axis = env steps per robot. Prints a table at the CPU baseline's final per-robot step budget.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SHORT = {"unitree_go1": "go1", "unitree_h1": "h1", "hexapod": "hx"}


def load_mjx(d):
    rows = [json.loads(l) for l in open(Path(d) / "log.jsonl")]
    robots = sorted({k.split("/")[0] for k in rows[0] if k.endswith("/ep_return")})
    out = {}
    for r in robots:
        x = np.array([row["steps_per_robot"] for row in rows], float)
        def series(m):
            y = np.array([row.get(f"{r}/{m}", np.nan) for row in rows], float)
            return x, y
        out[r] = {"ep_return": series("ep_return"), "eps_track": series("eps_track_perf"), "ep_length": series("ep_length"),
                  "unstable": series("sim_unstable")}
    return out


def load_cpu(tb_dir, nr_robots=3):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    files = sorted(Path(tb_dir).rglob("events.out.tfevents*"))
    out = {}
    tags_seen = set()
    for f in files:
        ea = EventAccumulator(str(f), size_guidance={"scalars": 0}); ea.Reload()
        for tag in ea.Tags()["scalars"]:
            tags_seen.add(tag)
            for long, short in SHORT.items():
                if not tag.endswith("/" + short):
                    continue
                kind = "ep_return" if "episode_return" in tag else "eps_track" if "eps_track_perf_perc" in tag else "ep_length" if "episode_length" in tag else None
                if kind is None:
                    continue
                ev = ea.Scalars(tag)
                x = np.array([e.step for e in ev], float) / nr_robots  # global step -> per robot
                y = np.array([e.value for e in ev], float)
                out.setdefault(long, {})[kind] = (x, y)
    if not out:
        print("CPU tags available:", sorted(tags_seen)[:60])
    return out


def smooth(y, k=5):
    if len(y) < k:
        return y
    yy = np.convolve(np.nan_to_num(y, nan=np.nanmean(y)), np.ones(k) / k, mode="same")
    return yy


def main():
    png = sys.argv[1]
    cpu = load_cpu(sys.argv[2]) if sys.argv[2] != "-" else {}
    runs = []
    for a in sys.argv[3:]:
        d, _, label = a.partition("=")
        runs.append((label or Path(d).name, load_mjx(d)))
    robots = sorted(set(cpu) | {r for _, rr in runs for r in rr})
    fig, axes = plt.subplots(2, len(robots), figsize=(5 * len(robots), 7), squeeze=False)
    budget = {r: cpu[r]["ep_return"][0][-1] for r in cpu if "ep_return" in cpu[r]}
    table = []
    for j, r in enumerate(robots):
        for i, kind in enumerate(("ep_return", "eps_track")):
            ax = axes[i, j]
            if r in cpu and kind in cpu[r]:
                x, y = cpu[r][kind]
                ax.plot(x / 1e6, smooth(y, 3), label="CPU URMA (MuJoCo C)", color="k")
            for label, rr in runs:
                if r in rr:
                    x, y = rr[r][kind]
                    ax.plot(x / 1e6, smooth(y), label=f"GPU {label}")
            ax.set_title(f"{r}: {'episode return' if kind == 'ep_return' else 'episode tracking perf.'}")
            ax.set_xlabel("env steps per robot (M)")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.legend(fontsize=7)
        b = budget.get(r)
        row = {"robot": r, "cpu_budget_M": None if b is None else round(b / 1e6, 1)}
        if b is not None:
            for kind in ("ep_return", "eps_track"):
                x, y = cpu[r][kind]
                row[f"cpu_{kind}"] = round(float(np.nanmean(y[-3:])), 3)
        for label, rr in runs:
            if r not in rr:
                continue
            for kind in ("ep_return", "eps_track"):
                x, y = rr[r][kind]
                lim = b if b is not None else x[-1]
                m = (x <= lim) & (x > lim * 0.9)
                row[f"{label}_{kind}@budget"] = round(float(np.nanmean(y[m])), 3) if m.any() else None
                row[f"{label}_{kind}@end"] = round(float(np.nanmean(y[-5:])), 3)
            row[f"{label}_unstable_total"] = int(np.nansum(rr[r]["unstable"][1]))
        table.append(row)
    fig.tight_layout()
    fig.savefig(png, dpi=110)
    for row in table:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
