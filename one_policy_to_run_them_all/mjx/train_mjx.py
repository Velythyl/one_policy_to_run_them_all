"""Fully on-GPU URMA PPO (uni_ppo semantics) over MJX/MuJoCo-Warp envs.

One jitted function does a whole PPO iteration: rollout (lax.scan over nr_steps; inside, one
vmapped batch_step per robot type), GAE, and nr_epochs x nr_minibatches updates. No host round
trips per env step; the host only logs once per iteration.

PPO details copied from algorithms/uni_ppo/ppo/ppo.py: per-robot-type minibatches of size
minibatch_size / nr_types, per-type advantage normalisation, loss = mean over types, separate
policy / critic Adam with global-norm clipping and the same linear LR schedule, GAE whose
recursion is cut only by terminations (truncated steps bootstrap from the final observation).

    python train_mjx.py --robots unitree_go1,unitree_h1,hexapod --nr_envs 4096 --total_steps 100e6 \
        --out /network/scratch/c/charlie.gauthier/urma_context/mjx/runs/<name>
"""
import argparse
import importlib.util
import json
import os
import pickle
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

import signal
import sys

from one_policy_to_run_them_all.mjx.robot_spec import load_spec, TRAIN_16, EVAL_4
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv, WARP_FFI_CONFIG
from one_policy_to_run_them_all.mjx.wandb_logger import WandbLogger, build_config, REWARD_TERMS

PPO_DIR = Path(__file__).resolve().parents[1] / "algorithms" / "uni_ppo" / "ppo"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


Policy = _load("urma_policy", PPO_DIR / "policy.py").Policy
Critic = _load("urma_critic", PPO_DIR / "critic.py").Critic


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--robots", default=",".join(TRAIN_16))
    p.add_argument("--eval_robots", default=",".join(EVAL_4), help="unseen robots: evaluated only, never trained on ('' = none)")
    p.add_argument("--eval_every", type=int, default=0, help="evaluate every N iterations (0 = every 17233920 env steps, experiments/experiment.sh)")
    p.add_argument("--eval_envs", type=int, default=50, help="envs per eval robot (experiment.sh: evaluation_episodes=50)")
    p.add_argument("--max_wall_s", type=float, default=0, help=">0: checkpoint and exit(3) after this many seconds (chained jobs)")
    p.add_argument("--nr_envs", type=int, default=4096, help="per robot type")
    p.add_argument("--nr_steps", type=int, default=42)
    p.add_argument("--nr_minibatches", type=int, default=16)
    p.add_argument("--nr_epochs", type=int, default=10)
    p.add_argument("--total_steps", type=float, default=100e6, help="env steps summed over all robots")
    p.add_argument("--lr_start", type=float, default=4e-4)
    p.add_argument("--lr_end", type=float, default=0.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.9)
    p.add_argument("--clip_range", type=float, default=0.1)
    p.add_argument("--critic_coef", type=float, default=1.0)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--context_arm", default="urma")
    p.add_argument("--latent_dir", default="")
    p.add_argument("--impl", default="warp")
    p.add_argument("--model_variant", default="native")
    p.add_argument("--contacts_per_env", type=int, default=64)
    p.add_argument("--solimp_dmax_clamp", type=float, default=0.99,
                   help="cap geom solimp dmax (float32 Warp blows up at impedance ~1: go1/anymal/badger feet); 0 = off")
    p.add_argument("--friction_tangential_floor", type=float, default=0.0,
                   help="DR tangential friction sampled in [max(min, floor), max]; 0 = upstream range (default since the "
                        "adaptive-impratio fix; was 0.3)")
    p.add_argument("--physics_overrides", default='{"impratio_adapt_mu0": 0.3}',
                   help="RECIPE DEVIATION (GPU only, user-approved 2026-11-17): JSON GPU physics overrides (MJXRobotEnv.physics_overrides). "
                        "Default: per-world impratio = impratio * min(1, g(mu)/g(0.3)), g = mu^2 (1 + mu^2): identical model for DR "
                        "friction mu >= 0.3, softer friction rows below (float32 Warp blows up there otherwise). '' = none")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--bench_iters", type=int, default=0, help=">0: only time this many iterations")
    p.add_argument("--debug_finite", action="store_true", help="report non-finite leaves of the first rollout")
    # campaign 6 (URMA 16 + 100 Unimals): grouped compilation + per-robot env counts + final full evaluation
    p.add_argument("--group_size", type=int, default=0,
                   help="0 = one jitted PPO iteration over all robot types (campaigns 1-5). >0 = robot types are compiled in "
                        "groups of this size (one jitted rollout+GAE and one jitted minibatch-gradient per group, called "
                        "in sequence by the host; gradients summed, then one optimizer step per minibatch). Same PPO math; "
                        "host RAM bounded by one group's compile; each group's program is cached by JAX's persistent cache")
    p.add_argument("--nr_envs_override", default="",
                   help="comma list pattern=N (fnmatch on the robot name) overriding --nr_envs, e.g. 'unimal_*=196' "
                        "(env steps per robot are proportional to its env count; needs --group_size > 0)")
    p.add_argument("--eval_group_size", type=int, default=4, help="grouped path: eval robots compiled per group of this size")
    p.add_argument("--final_eval_robots", default="",
                   help="@file or comma list: evaluated once on the final policy (same protocol as --eval_robots), written to "
                        "<out>/final_eval.json before DONE (grouped path)")
    p.add_argument("--train_set", default="", help="wandb urma.train_set (campaign tag, e.g. urma16+unimal100)")
    return p.parse_args()


def robot_list(s):
    """'a,b,c' or '@/path/to/file' (one robot per line, '#' comments)."""
    if s.startswith("@"):
        return [l.split("#")[0].strip() for l in open(s[1:]) if l.split("#")[0].strip()]
    return [r for r in s.split(",") if r]


def gpu_mem_diag():
    """GiB: XLA allocator in use / peak, Warp mempool current / high, device free; # Warp-captured graphs."""
    import warp as wp
    from mujoco.mjx.warp import ffi as mffi
    g = 2 ** 30
    out = {}
    try:
        ms = jax.devices()[0].memory_stats() or {}
        out["mem/xla_in_use"] = round(ms.get("bytes_in_use", 0) / g, 2)
        out["mem/xla_peak"] = round(ms.get("peak_bytes_in_use", 0) / g, 2)
        out["mem/warp_pool"] = round(wp.get_mempool_used_mem_current("cuda:0") / g, 2)
        out["mem/warp_pool_high"] = round(wp.get_mempool_used_mem_high("cuda:0") / g, 2)
        dev = wp.get_device("cuda:0")
        out["mem/dev_free"] = round(dev.free_memory / g, 2)
        out["mem/warp_graphs"] = sum(len(getattr(c, "captures", {})) for c in mffi._JAX_CALLABLE_VARIADIC_TUPLE_REGISTRY.values())
        out["mem/warp_callables"] = len(mffi._JAX_CALLABLE_VARIADIC_TUPLE_REGISTRY)
    except Exception as e:  # diagnostics only
        out["mem/error"] = repr(e)[:200]
    return out


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    robots = robot_list(a.robots)
    R = len(robots)
    N, T = a.nr_envs, a.nr_steps
    import fnmatch
    ovr = [(x.split("=")[0], int(x.split("=")[1])) for x in a.nr_envs_override.split(",") if x]
    Ns = [next((n for pat, n in ovr if fnmatch.fnmatch(r, pat)), N) for r in robots]
    GROUPED = a.group_size > 0
    assert GROUPED or len(set(Ns)) == 1, "--nr_envs_override needs --group_size > 0"
    mk_env = lambda r, n, ev=False: MJXRobotEnv(load_spec(r), context_arm=a.context_arm, latent_dir=a.latent_dir, impl=a.impl,
                        nr_envs_per_type=n, model_variant=a.model_variant, eval_mode=ev, contacts_per_env=a.contacts_per_env, solimp_dmax_clamp=a.solimp_dmax_clamp, friction_tangential_floor=a.friction_tangential_floor, physics_overrides=a.physics_overrides)
    envs = [mk_env(r, n) for r, n in zip(robots, Ns)]
    eval_robots = robot_list(a.eval_robots)
    eval_envs = [mk_env(r, a.eval_envs, True) for r in eval_robots]
    final_eval_robots = robot_list(a.final_eval_robots)
    for n_ in Ns:
        assert (n_ * T) % a.nr_minibatches == 0
    batch_per_type = N * T
    mb = batch_per_type // a.nr_minibatches
    mbs = [n_ * T // a.nr_minibatches for n_ in Ns]
    steps_per_iter = sum(Ns) * T
    nr_iters = int(a.total_steps // steps_per_iter)
    nr_updates = max(nr_iters, 1)
    print(json.dumps({"robots": robots, "nr_envs_per_type": N, "nr_envs_per_robot": dict(zip(robots, Ns)), "nr_steps": T,
                      "batch": steps_per_iter, "minibatch_per_type": mb, "iters": nr_iters, "gpu": jax.devices()[0].device_kind,
                      "impl": a.impl, "arm": a.context_arm, "variant": a.model_variant, "group_size": a.group_size,
                      "total_worlds": sum(Ns), "steps_per_robot_total": {r: nr_iters * n_ * T for r, n_ in zip(robots, Ns)}}), flush=True)
    deviations = {"solimp_dmax_clamp": a.solimp_dmax_clamp,  # float32 Warp: cap contact impedance dmax
                  "friction_tangential_floor": a.friction_tangential_floor, "physics_overrides": a.physics_overrides}  # DR tangential mu in [floor, max]
    deviations["applied_impratio_adapt_mu0"] = {e.spec.name: e.impratio_adapt_mu0 for e in envs + eval_envs}
    print(json.dumps({"recipe_deviations_gpu": deviations}), flush=True)
    cfg = {"args": vars(a), "recipe_deviations_gpu": deviations, "warp_ffi": list(WARP_FFI_CONFIG)}
    (out / f"config-{os.environ.get('SLURM_JOB_ID', 'local')}.json").write_text(json.dumps(cfg, indent=1))
    wcfg = build_config(a, robots, eval_robots, deviations, WARP_FFI_CONFIG, jax.devices()[0].device_kind, steps_per_iter,
                        nr_iters, Path(__file__).resolve().parents[2], out.name)
    wlog = WandbLogger(out, wcfg, out.name, robots, eval_robots) if a.bench_iters == 0 and not a.debug_finite else None

    policy = Policy(1.0, 1.0, 0.015, 1e-8, 10.0, 1e-8, 2.0)
    critic = Critic(1.0, 0.015, 1e-8)

    def lr_schedule(count):
        frac = 1.0 - (count // (a.nr_minibatches * a.nr_epochs)) / nr_updates
        return a.lr_end + frac * (a.lr_start - a.lr_end)

    key = jax.random.PRNGKey(a.seed)
    key, kp, kc, *ke = jax.random.split(key, 3 + R)
    t0 = time.time()
    states = [jax.jit(partial(e.batch_reset, nr_envs=n_))(k) for e, k, n_ in zip(envs, ke, Ns)]
    o0 = states[0].obs
    pparams = policy.init(kp, o0["jd"][:2], o0["js"][:2], o0["fd"][:2], o0["fs"][:2], o0["gp"][:2])
    cparams = critic.init(kc, o0["jd"][:2], o0["js"][:2], o0["fd"][:2], o0["fs"][:2], o0["gc"][:2])
    tx = lambda: optax.chain(optax.clip_by_global_norm(a.max_grad_norm), optax.inject_hyperparams(optax.adam)(learning_rate=lr_schedule))
    pstate = TrainState.create(apply_fn=policy.apply, params=pparams, tx=tx())
    cstate = TrainState.create(apply_fn=critic.apply, params=cparams, tx=tx())
    # concrete step counters, so the jitted iteration does not retrace after the first update
    pstate = pstate.replace(step=jnp.array(0, jnp.int32))
    cstate = cstate.replace(step=jnp.array(0, jnp.int32))
    start_iter = 0
    ckpt = out / "checkpoint.pkl"
    if ckpt.is_file():
        with open(ckpt, "rb") as f:
            ck = pickle.load(f)
        pstate = pstate.replace(params=ck["pparams"], opt_state=ck["popt"], step=jnp.array(ck["pstep"], jnp.int32))
        cstate = cstate.replace(params=ck["cparams"], opt_state=ck["copt"], step=jnp.array(ck["cstep"], jnp.int32))
        start_iter = ck["iter"]
        key = ck["key"]
        # env counters drive the reward curriculum and the first-step command sample: continue them
        # (DR / physics state restart from a fresh reset; DR is resampled with prob 0.002 per step anyway)
        states = [s.replace(steps_done=jnp.full_like(s.steps_done, start_iter * T)) for s in states]
        print(f"resumed from iter {start_iter}", flush=True)

    def pol(params, o):
        return policy.apply(params, o["jd"], o["js"], o["fd"], o["fs"], o["gp"])

    def val(params, o):
        return critic.apply(params, o["jd"], o["js"], o["fd"], o["fs"], o["gc"]).squeeze(-1)

    def rollout(pstate, cstate, states, key, racc):
        def body(carry, _):
            states, key, racc = carry
            new_states, trans, new_racc = [], [], []
            for i, e in enumerate(envs):
                key, k = jax.random.split(key)
                o = states[i].obs
                mean, logstd = pol(pstate.params, o)
                std = jnp.exp(logstd)
                act = mean + std * jax.random.normal(k, mean.shape)
                logp = (-0.5 * ((act - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd).sum(-1)
                v = val(cstate.params, o)
                s2, info = e.batch_step(states[i], act)
                nv = val(cstate.params, info["final_obs"])
                new_states.append(s2)
                # logging only: per-episode sums of each reward term (accumulated across iterations in racc)
                acc = racc[i] + info["reward_terms"]
                dn_ = info["done"][:, None]
                new_racc.append(jnp.where(dn_, 0.0, acc))
                trans.append(dict(obs=o, act=act, logp=logp, val=v, next_val=nv, rew=info["reward"],
                                  term=info["terminated"].astype(jnp.float32), done=info["done"],
                                  ep_ret=info["episode_return"], ep_len=info["episode_length"],
                                  eps_track=info["eps_track_perf_perc"], track_err=info["track_err_sq"],
                                  unstable=info["sim_unstable"].astype(jnp.float32), ep_terms=jnp.where(dn_, acc, 0.0),
                                  **{k: info[k].astype(jnp.float32) for k in info if k.startswith("ovf_") or k in ("nacon_frac", "nefc_max")}))
            return (new_states, key, new_racc), trans
        (states, key, racc), traj = jax.lax.scan(body, (states, key, racc), None, length=T)
        return states, key, traj, racc

    def gae(tr):
        delta = tr["rew"] + a.gamma * tr["next_val"] * (1.0 - tr["term"]) - tr["val"]
        def f(adv, x):
            d, t = x
            adv = d + a.gamma * a.gae_lambda * (1.0 - t) * adv
            return adv, adv
        _, adv = jax.lax.scan(f, jnp.zeros_like(delta[-1]), (delta, tr["term"]), reverse=True)
        return adv, adv + tr["val"]

    def update(pstate, cstate, traj, key):
        flat = []
        for tr in traj:
            adv, ret = gae(tr)
            fl = lambda x: x.reshape((T * N,) + x.shape[2:])
            flat.append(dict(obs=jax.tree_util.tree_map(fl, tr["obs"]), act=fl(tr["act"]), logp=fl(tr["logp"]),
                             adv=fl(adv), ret=fl(ret), val=fl(tr["val"])))
        key, *pk = jax.random.split(key, R + 1)
        perms = [jax.vmap(lambda k: jax.random.permutation(k, T * N))(jax.random.split(k, a.nr_epochs)).reshape(-1, mb) for k in pk]
        perms = jnp.stack(perms, axis=1)  # (epochs*minibatches, R, mb)

        def loss_fn(pp, cp, idx):
            losses, mets = [], []
            for i in range(R):
                b = jax.tree_util.tree_map(lambda x: x[idx[i]], flat[i])
                adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
                mean, logstd = pol(pp, b["obs"])
                std = jnp.exp(logstd)
                nlp = (-0.5 * ((b["act"] - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd).sum(-1)
                ent = (logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1)
                lr_ = nlp - b["logp"]
                ratio = jnp.exp(lr_)
                pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - a.clip_range, 1 + a.clip_range))
                v = val(cp, b["obs"])
                vl = 0.5 * (v - b["ret"]) ** 2
                loss = pg - a.entropy_coef * ent + a.critic_coef * vl
                losses.append(loss.mean())
                mets.append({"pg_loss": pg.mean(), "critic_loss": vl.mean(), "approx_kl": ((ratio - 1) - lr_).mean(),
                             "clip_frac": (jnp.abs(ratio - 1) > a.clip_range).mean(), "std": std.mean()})
            return jnp.mean(jnp.stack(losses)), jax.tree_util.tree_map(lambda *x: jnp.mean(jnp.stack(x)), *mets)

        grad_fn = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)

        def mb_step(carry, idx):
            ps, cs = carry
            (loss, met), (gp, gc) = grad_fn(ps.params, cs.params, idx)
            ps, cs = ps.apply_gradients(grads=gp), cs.apply_gradients(grads=gc)
            met["policy_grad_norm"] = optax.global_norm(gp)
            return (ps, cs), met
        (pstate, cstate), met = jax.lax.scan(mb_step, (pstate, cstate), perms)
        met = jax.tree_util.tree_map(jnp.mean, met)
        allret = jnp.concatenate([f["ret"] for f in flat]); allval = jnp.concatenate([f["val"] for f in flat])
        met["explained_var"] = 1 - jnp.var(allret - allval) / (jnp.var(allret) + 1e-8)
        return pstate, cstate, met, key

    # Donating the carried state keeps XLA's buffer addresses stationary across iterations, so Warp's
    # address-keyed CUDA graphs (GraphMode.WARP) are re-used instead of re-captured every iteration.
    @partial(jax.jit, donate_argnums=(0, 1, 2, 4) if os.environ.get("URMA_DONATE", "1") == "1" else ())
    def iteration(pstate, cstate, states, key, racc):
        states, key, traj, racc = rollout(pstate, cstate, states, key, racc)
        pstate, cstate, met, key = update(pstate, cstate, traj, key)
        stats = {}
        for r, tr in zip(robots, traj):
            dn = tr["done"].astype(jnp.float32)
            cnt = dn.sum()
            stats[r] = {
                "episodes": cnt,
                "ep_return": jnp.where(cnt > 0, (tr["ep_ret"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
                "ep_length": jnp.where(cnt > 0, (tr["ep_len"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
                "eps_track_perf": jnp.where(cnt > 0, (tr["eps_track"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
                "step_reward": tr["rew"].mean(),
                "track_err_sq": tr["track_err"].mean(),
                "sim_unstable": tr["unstable"].sum(),
                **{k: tr[k][-1].sum() for k in tr if k.startswith("ovf_")},  # worlds ever flagged (sticky)
                **{k: tr[k].max() for k in ("nacon_frac", "nefc_max") if k in tr},
                **{f"term_{t}": jnp.where(cnt > 0, tr["ep_terms"][..., j].sum() / jnp.maximum(cnt, 1), jnp.nan)
                   for j, t in enumerate(REWARD_TERMS)},
            }
        return pstate, cstate, states, key, met, stats, racc

    horizon = max([e.horizon for e in eval_envs] or [1])
    eval_every = a.eval_every or -(-17233920 // steps_per_iter)

    @jax.jit
    def evaluate(pparams, key):
        """CPU uni_ppo evaluate_callback: eval robots (env.eval -> curriculum 1), stochastic policy
        actions, statistics of the first finished episode of every env within one horizon."""
        stats = {}
        for r, e in zip(eval_robots, eval_envs):
            key, kr, kl = jax.random.split(key, 3)
            s0 = e.batch_reset(kr, a.eval_envs)

            def body(c, _):
                s, k, seen, ret, ln, trk = c
                k, ka = jax.random.split(k)
                mean, logstd = pol(pparams, s.obs)
                act = mean + jnp.exp(logstd) * jax.random.normal(ka, mean.shape)
                s, info = e.batch_step(s, act)
                first = info["done"] & ~seen
                ret = jnp.where(first, info["episode_return"], ret)
                ln = jnp.where(first, info["episode_length"], ln)
                trk = jnp.where(first, info["eps_track_perf_perc"], trk)
                return (s, k, seen | info["done"], ret, ln, trk), None
            z = jnp.zeros(a.eval_envs)
            (s, _, seen, ret, ln, trk), _ = jax.lax.scan(body, (s0, kl, jnp.zeros(a.eval_envs, bool), z, z, z), None, length=e.horizon)
            cnt = jnp.maximum(seen.sum(), 1)
            stats[r] = {"ep_return": (ret * seen).sum() / cnt, "ep_length": (ln * seen).sum() / cnt,
                        "eps_track_perf": (trk * seen).sum() / cnt, "episodes": seen.sum()}
        return stats

    # ------------------------------------------------------------------ grouped compilation (--group_size > 0)
    # Same math as rollout/update/iteration above, but each group of robot types is its own jitted program:
    #   rollout_g:  lax.scan over nr_steps of the group's batch_steps + GAE + flatten + per-robot stats
    #   grad_g:     gradient of sum_{robots in group} loss_r on minibatch j, added into a donated accumulator
    #   apply:      grads / R (= gradient of the old mean over robot types), one Adam step for policy and critic
    # Per minibatch the host calls grad_g for every group, then apply; 10 epochs x 16 minibatches as before.
    def robot_stats(tr):
        dn = tr["done"].astype(jnp.float32)
        cnt = dn.sum()
        return {
            "episodes": cnt,
            "ep_return": jnp.where(cnt > 0, (tr["ep_ret"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
            "ep_length": jnp.where(cnt > 0, (tr["ep_len"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
            "eps_track_perf": jnp.where(cnt > 0, (tr["eps_track"] * dn).sum() / jnp.maximum(cnt, 1), jnp.nan),
            "step_reward": tr["rew"].mean(),
            "track_err_sq": tr["track_err"].mean(),
            "sim_unstable": tr["unstable"].sum(),
            **{k: tr[k][-1].sum() for k in tr if k.startswith("ovf_")},
            **{k: tr[k].max() for k in ("nacon_frac", "nefc_max") if k in tr},
            **{f"term_{t}": jnp.where(cnt > 0, tr["ep_terms"][..., j].sum() / jnp.maximum(cnt, 1), jnp.nan)
               for j, t in enumerate(REWARD_TERMS)},
        }

    def make_group(idx):
        genvs, gN, gnames = [envs[i] for i in idx], [Ns[i] for i in idx], [robots[i] for i in idx]

        def rollout_g(pparams, cparams, states, key, racc):
            def body(carry, _):
                states, key, racc = carry
                new_states, trans, new_racc = [], [], []
                for i, e in enumerate(genvs):
                    key, k = jax.random.split(key)
                    o = states[i].obs
                    mean, logstd = pol(pparams, o)
                    std = jnp.exp(logstd)
                    act = mean + std * jax.random.normal(k, mean.shape)
                    logp = (-0.5 * ((act - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd).sum(-1)
                    v = val(cparams, o)
                    s2, info = e.batch_step(states[i], act)
                    nv = val(cparams, info["final_obs"])
                    new_states.append(s2)
                    acc = racc[i] + info["reward_terms"]
                    dn_ = info["done"][:, None]
                    new_racc.append(jnp.where(dn_, 0.0, acc))
                    trans.append(dict(obs=o, act=act, logp=logp, val=v, next_val=nv, rew=info["reward"],
                                      term=info["terminated"].astype(jnp.float32), done=info["done"],
                                      ep_ret=info["episode_return"], ep_len=info["episode_length"],
                                      eps_track=info["eps_track_perf_perc"], track_err=info["track_err_sq"],
                                      unstable=info["sim_unstable"].astype(jnp.float32), ep_terms=jnp.where(dn_, acc, 0.0),
                                      **{k: info[k].astype(jnp.float32) for k in info if k.startswith("ovf_") or k in ("nacon_frac", "nefc_max")}))
                return (new_states, key, new_racc), trans
            (states, key, racc), traj = jax.lax.scan(body, (states, key, racc), None, length=T)
            flat, stats = [], {}
            for r, n_, tr in zip(gnames, gN, traj):
                adv, ret = gae(tr)
                fl = lambda x, n_=n_: x.reshape((T * n_,) + x.shape[2:])
                flat.append(dict(obs=jax.tree_util.tree_map(fl, tr["obs"]), act=fl(tr["act"]), logp=fl(tr["logp"]),
                                 adv=fl(adv), ret=fl(ret), val=fl(tr["val"])))
                stats[r] = robot_stats(tr)
            return states, key, racc, flat, stats

        def grad_g(pp, cp, flat, perms, j, acc):
            def loss_fn(pp, cp):
                tot, mets = 0.0, []
                for i in range(len(idx)):
                    b = jax.tree_util.tree_map(lambda x: x[perms[i][j]], flat[i])
                    adv = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
                    mean, logstd = pol(pp, b["obs"])
                    std = jnp.exp(logstd)
                    nlp = (-0.5 * ((b["act"] - mean) / std) ** 2 - 0.5 * jnp.log(2.0 * jnp.pi) - logstd).sum(-1)
                    ent = (logstd + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e)).sum(-1)
                    lr_ = nlp - b["logp"]
                    ratio = jnp.exp(lr_)
                    pg = jnp.maximum(-adv * ratio, -adv * jnp.clip(ratio, 1 - a.clip_range, 1 + a.clip_range))
                    v = val(cp, b["obs"])
                    vl = 0.5 * (v - b["ret"]) ** 2
                    loss = pg - a.entropy_coef * ent + a.critic_coef * vl
                    tot = tot + loss.mean()
                    mets.append({"pg_loss": pg.mean(), "critic_loss": vl.mean(), "approx_kl": ((ratio - 1) - lr_).mean(),
                                 "clip_frac": (jnp.abs(ratio - 1) > a.clip_range).mean(), "std": std.mean()})
                return tot, jax.tree_util.tree_map(lambda *x: jnp.sum(jnp.stack(x)), *mets)
            (_, met), (gp, gc) = jax.value_and_grad(loss_fn, argnums=(0, 1), has_aux=True)(pp, cp)
            add = lambda x, y: jax.tree_util.tree_map(jnp.add, x, y)
            return add(acc[0], gp), add(acc[1], gc), add(acc[2], met)

        return dict(idx=idx, rollout=jax.jit(rollout_g, donate_argnums=(2, 4)), grad=jax.jit(grad_g, donate_argnums=(5,)))

    if GROUPED:
        groups = [make_group(list(range(i, min(i + a.group_size, R)))) for i in range(0, R, a.group_size)]

        @jax.jit
        def make_perms(key):  # identical draws to update(): key, *pk = split(key, R + 1)
            key, *pk = jax.random.split(key, R + 1)
            return key, [jax.vmap(lambda k: jax.random.permutation(k, T * n_))(jax.random.split(k, a.nr_epochs)).reshape(-1, m_)
                         for k, n_, m_ in zip(pk, Ns, mbs)]

        MET_KEYS = ("pg_loss", "critic_loss", "approx_kl", "clip_frac", "std")

        @partial(jax.jit, donate_argnums=(0, 1, 2, 3))
        def apply_grads(pstate, cstate, acc, macc):
            gp, gc, met = jax.tree_util.tree_map(lambda x: x / R, acc)
            met = dict(met, policy_grad_norm=optax.global_norm(gp))
            pstate, cstate = pstate.apply_gradients(grads=gp), cstate.apply_gradients(grads=gc)
            zero = jax.tree_util.tree_map(jnp.zeros_like, acc)
            return pstate, cstate, zero, jax.tree_util.tree_map(jnp.add, macc, met)

        @jax.jit
        def explained_var(rets, vals):
            allret, allval = jnp.concatenate(rets), jnp.concatenate(vals)
            return 1 - jnp.var(allret - allval) / (jnp.var(allret) + 1e-8)

        TIMING = os.environ.get("URMA_TIMING", "0") == "1"

        def grouped_iteration(pstate, cstate, states, key, racc):
            t_i = time.time()
            # one group: the key stream of iteration() exactly (equivalence test); several: one key per group
            gk = [key] if len(groups) == 1 else None
            if gk is None:
                key, *gk = jax.random.split(key, len(groups) + 1)
            flats, stats = [None] * R, {}
            states, racc = list(states), list(racc)
            for g, k in zip(groups, gk):
                ix = g["idx"]
                st_g, k_out, ra_g, fl_g, sts = g["rollout"](pstate.params, cstate.params, [states[i] for i in ix], k, [racc[i] for i in ix])
                if len(groups) == 1:
                    key = k_out
                for n_, i in enumerate(ix):
                    states[i], racc[i], flats[i] = st_g[n_], ra_g[n_], fl_g[n_]
                stats.update(sts)
            if TIMING:
                jax.block_until_ready(flats); t_r = time.time()
            key, perms = make_perms(key)
            z = lambda t: jax.tree_util.tree_map(jnp.zeros_like, t)
            acc = (z(pstate.params), z(cstate.params), {k_: jnp.zeros(()) for k_ in MET_KEYS})
            macc = dict({k_: jnp.zeros(()) for k_ in MET_KEYS}, policy_grad_norm=jnp.zeros(()))
            nmb = a.nr_epochs * a.nr_minibatches
            for j in range(nmb):
                jj = jnp.int32(j)
                for g in groups:
                    ix = g["idx"]
                    acc = g["grad"](pstate.params, cstate.params, [flats[i] for i in ix], [perms[i] for i in ix], jj, acc)
                pstate, cstate, acc, macc = apply_grads(pstate, cstate, acc, macc)
            met = {k_: v / nmb for k_, v in macc.items()}
            met["explained_var"] = explained_var([f["ret"] for f in flats], [f["val"] for f in flats])
            if TIMING:  # logging only: wall time of the rollout phase vs the update phase
                jax.block_until_ready(met); met["time/update_s"] = time.time() - t_r; met["time/rollout_s"] = t_r - t_i
            return pstate, cstate, states, key, met, stats, racc

        eval_groups = []
        def make_eval_group(names, evs):
            def evaluate_g(pparams, key):
                st = {}
                for r, e in zip(names, evs):
                    key, kr, kl = jax.random.split(key, 3)
                    s0 = e.batch_reset(kr, a.eval_envs)

                    def body(c, _, e=e):
                        s, k, seen, ret, ln, trk = c
                        k, ka = jax.random.split(k)
                        mean, logstd = pol(pparams, s.obs)
                        act = mean + jnp.exp(logstd) * jax.random.normal(ka, mean.shape)
                        s, info = e.batch_step(s, act)
                        first = info["done"] & ~seen
                        ret = jnp.where(first, info["episode_return"], ret)
                        ln = jnp.where(first, info["episode_length"], ln)
                        trk = jnp.where(first, info["eps_track_perf_perc"], trk)
                        return (s, k, seen | info["done"], ret, ln, trk), None
                    zz = jnp.zeros(a.eval_envs)
                    (s, _, seen, ret, ln, trk), _ = jax.lax.scan(body, (s0, kl, jnp.zeros(a.eval_envs, bool), zz, zz, zz), None, length=e.horizon)
                    cnt = jnp.maximum(seen.sum(), 1)
                    st[r] = {"ep_return": (ret * seen).sum() / cnt, "ep_length": (ln * seen).sum() / cnt,
                             "eps_track_perf": (trk * seen).sum() / cnt, "episodes": seen.sum()}
                return st
            return jax.jit(evaluate_g)

        for i in range(0, len(eval_robots), a.eval_group_size):
            eval_groups.append(make_eval_group(eval_robots[i:i + a.eval_group_size], eval_envs[i:i + a.eval_group_size]))

        def grouped_evaluate(pparams, key, egroups=None):
            out_ = {}
            for fn in (egroups or eval_groups):
                key, k = jax.random.split(key)
                out_.update(jax.device_get(fn(pparams, k)))
            return out_

    stop = {"flag": False}
    signal.signal(signal.SIGUSR1, lambda *_: stop.update(flag=True))  # Slurm --signal=USR1@<s>: checkpoint + exit(3)

    def save_ckpt(it_done):
        tmp = out / "checkpoint.pkl.tmp"
        with open(tmp, "wb") as f:
            pickle.dump({"pparams": jax.device_get(pstate.params), "popt": jax.device_get(pstate.opt_state),
                         "pstep": int(pstate.step), "cparams": jax.device_get(cstate.params),
                         "copt": jax.device_get(cstate.opt_state), "cstep": int(cstate.step),
                         "iter": it_done, "key": jax.device_get(key), "args": vars(a)}, f)
        os.replace(tmp, ckpt)

    racc = [jnp.zeros((n_, len(REWARD_TERMS)), jnp.float32) for n_ in Ns]  # fresh episodes after (re)start
    if a.debug_finite:
        st_, k_, traj_, _ = jax.jit(rollout)(pstate, cstate, states, key, racc)
        for r, tr in zip(robots, traj_):
            for path, x in jax.tree_util.tree_leaves_with_path(tr):
                if jnp.issubdtype(x.dtype, jnp.floating):
                    bad = ~jnp.isfinite(x)
                    if bool(bad.any()):
                        t_first = int(jnp.argmax(bad.reshape(bad.shape[0], -1).any(-1)))
                        print("NONFINITE", r, jax.tree_util.keystr(path), "first t", t_first, "frac", float(bad.mean()), flush=True)
        print("debug_finite done", flush=True)
        return
    log = open(out / "log.jsonl", "a")
    t_compile = time.time()
    n_run = a.bench_iters if a.bench_iters > 0 else nr_iters
    for it in range(start_iter, n_run):
        ti = time.time()
        step_fn = grouped_iteration if GROUPED else iteration
        pstate, cstate, states, key, met, stats, racc = step_fn(pstate, cstate, states, key, racc)
        jax.block_until_ready(met)
        dt = time.time() - ti
        if it == start_iter:
            print(json.dumps({"compile_plus_first_iter_s": round(time.time() - t_compile, 1), "setup_s": round(t_compile - t0, 1)}), flush=True)
        rec = {"iter": it + 1, "env_steps": (it + 1) * steps_per_iter, "steps_per_robot": (it + 1) * N * T,
               **({"steps_per_robot_min": (it + 1) * min(Ns) * T} if GROUPED else {}),
               "sps": round(steps_per_iter / dt), "iter_s": round(dt, 2), "wall_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in met.items()},
               **{f"{r}/{k}": float(v) for r, s in stats.items() for k, v in s.items()}}
        if eval_envs and a.bench_iters == 0 and (it + 1) % eval_every == 0:
            te = time.time()
            key, ke = jax.random.split(key)
            ev = grouped_evaluate(pstate.params, ke) if GROUPED else jax.device_get(evaluate(pstate.params, ke))
            rec.update({f"eval/{r}/{k}": float(v) for r, s_ in ev.items() for k, v in s_.items()})
            rec["eval_s"] = round(time.time() - te, 1)
        rec.update(gpu_mem_diag())
        log.write(json.dumps(rec) + "\n"); log.flush()
        if wlog is not None:
            wlog.log(rec)
        if (it + 1) % 5 == 0 or it == start_iter + 1:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        timeout = a.max_wall_s > 0 and time.time() - t0 > a.max_wall_s
        if a.bench_iters == 0 and ((it + 1) % a.save_every == 0 or it + 1 == nr_iters or stop["flag"] or timeout):
            save_ckpt(it + 1)
        if a.bench_iters == 0 and (stop["flag"] or timeout) and it + 1 < nr_iters:
            print(f"CHECKPOINTED at iter {it + 1}/{nr_iters}; exiting for the next chained job", flush=True)
            if wlog is not None:
                wlog.finish(0)
            sys.exit(3)
    if a.bench_iters == 0 and final_eval_robots and not (out / "final_eval.json").is_file():
        assert GROUPED, "--final_eval_robots needs --group_size > 0"
        tf = time.time()
        fe = {}
        for i in range(0, len(final_eval_robots), a.eval_group_size):
            names = final_eval_robots[i:i + a.eval_group_size]
            fn = make_eval_group(names, [mk_env(r, a.eval_envs, True) for r in names])
            kf = jax.random.fold_in(jax.random.PRNGKey(a.seed + 7919), i)  # fixed keys: independent of the training key stream
            fe.update({r: {k: float(v) for k, v in d_.items()} for r, d_ in jax.device_get(fn(pstate.params, kf)).items()})
            print(json.dumps({"final_eval_done": len(fe), "of": len(final_eval_robots), "s": round(time.time() - tf, 1)}), flush=True)
            if stop["flag"] or (a.max_wall_s > 0 and time.time() - t0 > a.max_wall_s):
                print("final eval interrupted (time); the next chained job redoes it", flush=True)
                if wlog is not None:
                    wlog.finish(0)
                sys.exit(3)
        (out / "final_eval.json").write_text(json.dumps({"iter": nr_iters, "eval_envs": a.eval_envs, "robots": fe,
                                                         "wall_s": round(time.time() - tf, 1)}, indent=1))
        if wlog is not None:
            wlog.log_final_eval(fe)
    if a.bench_iters == 0:
        (out / "DONE").write_text(str(nr_iters))
    if wlog is not None:
        wlog.finish(0)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
