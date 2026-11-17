"""wandb logging for train_mjx.py, compatible with ModuMorph's wandbutils (selection + plotting).

Config: the keys ModuMorph's wandbutils select on (model.model_shorthand, vma.add_gt_bodies, vma.add_vma_latents,
vma.add_nonbodyinfo, vma.delete_context, fixargs.fixargs, fixargs.ablation, task.task_shorthand, meta.config_name)
carry the SAME value types/strings as ModuMorph's runs (Python bools / None, the same fixargs strings), so the
ALL_POSSIBLE_SETTINGS dicts of /home/mila/c/charlie.gauthier/ModuMorph/wandbutils/refactor_plots.py work with only
"model.model_shorthand": "URMA" (PRETTY_NAME "D (URMA)", ...). The authoritative arm key is urma.context_arm.

History: ModuMorph key names where an equivalent exists (x axis "train/env_steps_done", "train_stats__env__/reward",
"train_stats__env__reward/__reward__<term>", ...), see TrainMetrics.

Robustness: WANDB_MODE (online | offline | disabled) is honoured; the run id is persisted in <out>/wandb_run_id so every
chained job resumes the SAME run (wandb.init(id=..., resume="allow")); any wandb failure prints a warning and disables
wandb for the rest of the job -- training and log.jsonl are never affected.
Env: URMA_WANDB=0 disables; URMA_WANDB_ENTITY (default velythyl = ModuMorph's entity), URMA_WANDB_PROJECT
(default urma_context); WANDB_API_KEY or ~/.netrc for the key (never stored in the repo).
"""
import json
import math
import os
import secrets
import socket
import string
import subprocess
import sys
import traceback
from collections import deque
from pathlib import Path

import numpy as np

DEFAULT_ENTITY = "velythyl"
DEFAULT_PROJECT = "urma_context"

# ModuMorph's fixargs strings (exactly as in ALL_POSSIBLE_SETTINGS after the downloader's "/" -> "-"), used as markers
_FIX_PLAIN = "MODEL.VMA.LATENT_DIR /tmp/vmalatentdir"
_FIX_VMESTAR = "MODEL.VMA.CHECK_PATH resnet_new_decoder-L16-WTrue-notest_TargetOnly MODEL.VMA.LATENT_DIR /tmp/vmalatentdir"
_FIX_MATME = "MODEL.VMA.CHECK_PATH 1dcnn_autoencoder_BodyOnly  MODEL.VMA.LATENT_DIR /tmp/vmalatentdir"  # (double space as in ModuMorph)

# arm -> (vma.add_gt_bodies, vma.add_vma_latents, vma.add_nonbodyinfo, vma.delete_context, fixargs.fixargs, fixargs.ablation)
# mirrors the ModuMorph PRETTY_NAME it corresponds to ("D (ModuM)" -> "D (URMA)", ...)
ARM_COMPAT = {
    "dk":         (True,  False, None,  None, _FIX_PLAIN,   "Segmented"),       # D+K
    "d":          (False, False, None,  None, _FIX_PLAIN,   "Segmented"),       # D
    "k":          (True,  False, False, None, _FIX_PLAIN,   "Segmented"),       # K
    "empty":      (True,  False, None,  True, _FIX_PLAIN,   "Segmented"),       # Ø
    "vme_sa":     (False, True,  None,  None, _FIX_PLAIN,   "Segmented"),       # VME (size-aware latents)
    "vmestar_sa": (False, True,  None,  None, _FIX_VMESTAR, "TargetOnly"),      # VME* (size-aware latents)
    "matme":      (False, True,  None,  None, _FIX_MATME,   "1DCNN_BodyOnly"),  # MatME
    # per-robot [0,1]-scaled latents: same vma keys, fixargs marked so they never collide with VME / VME*
    "vme":        (False, True,  None,  None, _FIX_PLAIN + " URMA.LATENT_SCALING per_robot01",   "Segmented_PerRobot01"),
    "vmestar":    (False, True,  None,  None, _FIX_VMESTAR + " URMA.LATENT_SCALING per_robot01", "TargetOnly_PerRobot01"),
    # stock URMA description: vma.* None (matches no ModuMorph setting), urma.stock_context True
    "urma":       (None,  None,  None,  None, "URMA_STOCK_CONTEXT", "URMA_STOCK"),
}
PRETTY = {"dk": "D+K", "d": "D", "k": "K", "empty": "Ø", "vme_sa": "VME", "vmestar_sa": "VME*", "matme": "MatME",
          "vme": "VME-pr01", "vmestar": "VME*-pr01", "urma": "URMA-stock"}

# reward terms, in the order mjx_env._reward returns them; names = the upstream reward_functions logging names
REWARD_TERMS = ("track_xy_vel_cmd", "track_yaw_vel_cmd", "linear_velocity", "angular_velocity", "angular_position",
                "joint_nominal_diff", "joint_position_limit", "velocity", "acceleration", "torque", "action_rate",
                "collision", "base_height", "air_time", "symmetry_air")


def _warn(msg):
    print(f"[wandb_logger] WARNING: {msg}", file=sys.stderr, flush=True)


def git_info(code_dir):
    """(commit, dirty) of the code tree; frozen trees (git archive) carry the commit in <frozen>/FROZEN_OK."""
    code_dir = Path(code_dir)
    try:
        c = subprocess.run(["git", "-C", str(code_dir), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        if c.returncode == 0:
            d = subprocess.run(["git", "-C", str(code_dir), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=20)
            return c.stdout.strip(), bool(d.stdout.strip())
    except Exception:
        pass
    for p in (code_dir / "FROZEN_OK", code_dir.parent / "FROZEN_OK"):
        if p.is_file():
            return p.read_text().split()[0], False
    return "unknown", None


def latent_meta(latent_dir, arm):
    p = Path(latent_dir) / arm / "meta.json" if latent_dir else None
    if p is None or not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": repr(e)[:200]}


def build_config(a, robots, eval_robots, deviations, warp_ffi, gpu, steps_per_iter, nr_iters, code_dir, run_name):
    arm = a.context_arm
    gt, lat, nonbody, delete, fix, abl = ARM_COMPAT.get(arm, (None, None, None, None, f"URMA_ARM {arm}", f"URMA_{arm}"))
    commit, dirty = git_info(code_dir)
    task = "URMA16" if len(robots) == 16 else f"URMA{len(robots)}r"
    job = os.environ.get("SLURM_JOB_ID", "local")
    return {
        "model": {"model_name": "urma", "model_shorthand": "URMA"},
        # None = key ABSENT, as in ModuMorph's runs: wandb's {"config.x": null} filter (runselector) matches a missing
        # key but not a stored null; the downloader then writes "<key>:None" like for ModuMorph runs
        "vma": {k: v for k, v in (("add_gt_bodies", gt), ("add_vma_latents", lat), ("add_nonbodyinfo", nonbody),
                                  ("delete_context", delete)) if v is not None},
        "fixargs": {"fixargs": fix, "ablation": abl},
        "task": {"task_shorthand": task, "task_name": f"URMA locomotion, {len(robots)} train robots (MJX/Warp GPU)"},
        "dataset": {"dataset_name": "urma", "train_robots": list(robots), "eval_robots": list(eval_robots)},
        "meta": {"project": os.environ.get("URMA_WANDB_PROJECT", DEFAULT_PROJECT), "run_name": run_name, "seed": a.seed,
                 "config_name": f"urma_mjx_{task}", "HYDRA_SWEEP_DIR": str(Path(a.out).resolve()),
                 "SLURM_JOB_ID": job, "wandb_mode": os.environ.get("WANDB_MODE", "online"), "tags": ["urma_mjx", arm]},
        "urma": {"context_arm": arm, "pretty_arm": PRETTY.get(arm, arm), "stock_context": arm == "urma",
                 "args": vars(a), "git_commit": commit, "git_dirty": dirty, "code_dir": str(code_dir),
                 "cfg_dir": os.environ.get("URMA_MJX_CFG", ""), "latent_dir": a.latent_dir,
                 "latent_meta": latent_meta(a.latent_dir, arm), "recipe_deviations_gpu": deviations,
                 "warp_ffi": list(warp_ffi), "steps_per_iter": steps_per_iter, "nr_iters": nr_iters,
                 "reward_terms": list(REWARD_TERMS), "first_job": {"slurm_job_id": job, "gpu": gpu,
                 "cluster": os.environ.get("SLURM_CLUSTER_NAME", "unknown"), "host": socket.gethostname()}},
    }


class TrainMetrics:
    """Turns one log.jsonl record into ModuMorph-named wandb metrics.

    ModuMorph (metamorph/algos/ppo/ppo.py + utils/meter.py): per robot, the mean return of its last episodes; the
    env value is the mean over robots (only once every robot has finished an episode); train_stats__env__/* is that
    value averaged over the last 10 logged iterations (np.mean(v[-10:])), train/reward the unsmoothed one,
    train/reward_std the std over robots. Here a robot's per-iteration value is the mean over ALL episodes that ended
    in the iteration (thousands), carried forward when none ended.
    """

    def __init__(self, robots, eval_robots):
        self.robots, self.eval_robots = list(robots), list(eval_robots)
        self.last = {}  # robot -> {metric: last value}
        self.hist = {}  # env metric -> deque(maxlen=10)

    def _env(self, key, vals):
        v = float(np.mean(vals))
        self.hist.setdefault(key, deque(maxlen=10)).append(v)
        return v, float(np.mean(self.hist[key]))

    def __call__(self, rec):
        m = {"train/env_steps_done": rec["env_steps"], "train/cur_iter": rec["iter"] - 1, "train/fps": rec["sps"],
             "train/iter_s": rec["iter_s"], "train/wall_s": rec["wall_s"], "train/steps_per_robot": rec["steps_per_robot"]}
        ppo = {"pi_loss": "pg_loss", "val_loss": "critic_loss", "approx_kl": "approx_kl", "clip_frac": "clip_frac",
               "std": "std", "grad_norm": "policy_grad_norm", "explained_var": "explained_var"}
        for k, src in ppo.items():
            if src in rec:
                m[f"train_stats__env__/{k}"] = rec[src]
        per = {"reward": "ep_return", "ep_len": "ep_length", "metric": "eps_track_perf"}
        unstable = 0.0
        for r in self.robots:
            L = self.last.setdefault(r, {})
            for k, src in per.items():
                v = rec.get(f"{r}/{src}")
                if v is not None and math.isfinite(v):
                    L[k] = v
            for t in REWARD_TERMS:
                v = rec.get(f"{r}/term_{t}")
                if v is not None and math.isfinite(v):
                    L[f"__reward__{t}"] = v
            for k in L:
                m[f"train_stats__env__/{r}/{k}" if not k.startswith("__reward__") else f"train_stats__env__reward/{r}/{k}"] = L[k]
            for k in ("step_reward", "track_err_sq", "sim_unstable", "episodes"):
                if f"{r}/{k}" in rec:
                    m[f"train_stats__env__/{r}/{k}"] = rec[f"{r}/{k}"]
            unstable += rec.get(f"{r}/sim_unstable", 0.0)
        m["train/sim_unstable"] = unstable
        if all("reward" in self.last.get(r, {}) for r in self.robots):
            vals = [self.last[r]["reward"] for r in self.robots]
            raw, sm = self._env("reward", vals)
            m["train/reward"], m["train/reward_std"], m["train_stats__env__/reward"] = raw, float(np.std(vals)), sm
            for k in ("ep_len", "metric"):
                m[f"train_stats__env__/{k}"] = self._env(k, [self.last[r][k] for r in self.robots])[1]
            for t in REWARD_TERMS:
                key = f"__reward__{t}"
                if all(key in self.last[r] for r in self.robots):
                    m[f"train_stats__env__reward/{key}"] = self._env(key, [self.last[r][key] for r in self.robots])[1]
        ev = [r for r in self.eval_robots if f"eval/{r}/ep_return" in rec]
        if ev:
            for r in ev:
                for k, src in per.items():
                    m[f"eval__env__/{r}/{k}"] = rec[f"eval/{r}/{src}"]
                m[f"eval__env__/{r}/episodes"] = rec.get(f"eval/{r}/episodes")
            for k, src in per.items():
                m[f"eval__env__/{k}"] = float(np.mean([rec[f"eval/{r}/{src}"] for r in ev]))
            m["eval__env__/eval_s"] = rec.get("eval_s")
        for k, v in rec.items():
            if k.startswith("mem/") and isinstance(v, (int, float)):
                m[k] = v
        return {k: v for k, v in m.items() if v is not None and not (isinstance(v, float) and not math.isfinite(v))}


class WandbLogger:
    """Never raises: on any wandb error it warns and turns itself off (log.jsonl is the source of truth)."""

    def __init__(self, out, config, run_name, robots, eval_robots):
        self.run = None
        self.metrics = TrainMetrics(robots, eval_robots)
        self.out = Path(out)
        if os.environ.get("URMA_WANDB", "1") == "0" or os.environ.get("WANDB_MODE") == "disabled":
            print("[wandb_logger] wandb disabled", flush=True)
            return
        try:
            import wandb
            self.wandb = wandb
            id_file = self.out / "wandb_run_id"
            run_id = id_file.read_text().strip() if id_file.is_file() else "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
            if not id_file.is_file():
                id_file.write_text(run_id + "\n")
            mode = os.environ.get("WANDB_MODE", "online")
            self.run = wandb.init(
                entity=os.environ.get("URMA_WANDB_ENTITY", DEFAULT_ENTITY),
                project=os.environ.get("URMA_WANDB_PROJECT", DEFAULT_PROJECT),
                id=run_id, resume="allow", name=run_name, dir=str(self.out), mode=mode,
                tags=config["meta"]["tags"], group=config["urma"]["context_arm"], save_code=False,
                settings=wandb.Settings(init_timeout=180, x_disable_stats=True, console="off", quiet=True))
            # identical config in every chained job (works offline too): the first job's identity and the list of
            # all jobs are persisted in <out>/wandb_jobs.json
            jobs_file = self.out / "wandb_jobs.json"
            jobs = json.loads(jobs_file.read_text()) if jobs_file.is_file() else {}
            job = os.environ.get("SLURM_JOB_ID", "local")
            first = config["urma"]["first_job"]
            jobs.setdefault(job, {**first, "code_dir": config["urma"]["code_dir"],
                                  "git_commit": config["urma"]["git_commit"], "wandb_mode": mode})
            jobs_file.write_text(json.dumps(jobs, indent=1))
            first_id = next(iter(jobs))
            config = json.loads(json.dumps(config, default=str))
            config["urma"]["first_job"] = {k: jobs[first_id][k] for k in first}
            config["meta"]["SLURM_JOB_ID"] = jobs[first_id]["slurm_job_id"]
            config["meta"]["wandb_mode"] = jobs[first_id]["wandb_mode"]
            config["urma"]["jobs"] = jobs
            self.run.config.update(config, allow_val_change=True)
            print(f"[wandb_logger] run {self.run.entity}/{self.run.project}/{self.run.id} mode={mode} "
                  f"resumed={self.run.resumed}", flush=True)
        except Exception as e:
            _warn(f"wandb init failed, continuing without wandb: {e!r}")
            traceback.print_exc()
            self.run = None

    def log(self, rec):
        if self.run is None:
            return
        try:
            m = self.metrics(rec)
            step = int(rec["iter"])
            if step < (self.run.step or 0):
                return  # iteration already logged by an earlier job (re-run after a crash): keep wandb monotonic
            self.run.log(m, step=step)
        except Exception as e:
            _warn(f"wandb log failed, disabling wandb for this job: {e!r}")
            self._finish_quiet(1)

    def _finish_quiet(self, code):
        try:
            if self.run is not None:
                self.run.finish(exit_code=code)
        except Exception as e:
            _warn(f"wandb finish failed: {e!r}")
        self.run = None

    def finish(self, exit_code=0):
        self._finish_quiet(exit_code)
