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

from one_policy_to_run_them_all.mjx.robot_spec import load_spec
from one_policy_to_run_them_all.mjx.mjx_env import MJXRobotEnv

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
    p.add_argument("--robots", default="unitree_go1,unitree_h1,hexapod")
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
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--save_every", type=int, default=50)
    p.add_argument("--bench_iters", type=int, default=0, help=">0: only time this many iterations")
    return p.parse_args()


def main():
    a = parse()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    robots = a.robots.split(",")
    R = len(robots)
    N, T = a.nr_envs, a.nr_steps
    envs = [MJXRobotEnv(load_spec(r), context_arm=a.context_arm, latent_dir=a.latent_dir, impl=a.impl,
                        nr_envs_per_type=N, model_variant=a.model_variant) for r in robots]
    batch_per_type = N * T
    assert batch_per_type % a.nr_minibatches == 0
    mb = batch_per_type // a.nr_minibatches
    steps_per_iter = R * N * T
    nr_iters = int(a.total_steps // steps_per_iter)
    nr_updates = max(nr_iters, 1)
    print(json.dumps({"robots": robots, "nr_envs_per_type": N, "nr_steps": T, "batch": steps_per_iter,
                      "minibatch_per_type": mb, "iters": nr_iters, "gpu": jax.devices()[0].device_kind,
                      "impl": a.impl, "arm": a.context_arm, "variant": a.model_variant}), flush=True)

    policy = Policy(1.0, 1.0, 0.015, 1e-8, 10.0, 1e-8, 2.0)
    critic = Critic(1.0, 0.015, 1e-8)

    def lr_schedule(count):
        frac = 1.0 - (count // (a.nr_minibatches * a.nr_epochs)) / nr_updates
        return a.lr_end + frac * (a.lr_start - a.lr_end)

    key = jax.random.PRNGKey(a.seed)
    key, kp, kc, *ke = jax.random.split(key, 3 + R)
    t0 = time.time()
    states = [jax.jit(partial(e.batch_reset, nr_envs=N))(k) for e, k in zip(envs, ke)]
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
        print(f"resumed from iter {start_iter}", flush=True)

    def pol(params, o):
        return policy.apply(params, o["jd"], o["js"], o["fd"], o["fs"], o["gp"])

    def val(params, o):
        return critic.apply(params, o["jd"], o["js"], o["fd"], o["fs"], o["gc"]).squeeze(-1)

    def rollout(pstate, cstate, states, key):
        def body(carry, _):
            states, key = carry
            new_states, trans = [], []
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
                trans.append(dict(obs=o, act=act, logp=logp, val=v, next_val=nv, rew=info["reward"],
                                  term=info["terminated"].astype(jnp.float32), done=info["done"],
                                  ep_ret=info["episode_return"], ep_len=info["episode_length"],
                                  eps_track=info["eps_track_perf_perc"], track_err=info["track_err_sq"]))
            return (new_states, key), trans
        (states, key), traj = jax.lax.scan(body, (states, key), None, length=T)
        return states, key, traj

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

    @jax.jit
    def iteration(pstate, cstate, states, key):
        states, key, traj = rollout(pstate, cstate, states, key)
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
            }
        return pstate, cstate, states, key, met, stats

    log = open(out / "log.jsonl", "a")
    t_compile = time.time()
    n_run = a.bench_iters if a.bench_iters > 0 else nr_iters
    for it in range(start_iter, n_run):
        ti = time.time()
        pstate, cstate, states, key, met, stats = iteration(pstate, cstate, states, key)
        jax.block_until_ready(met)
        dt = time.time() - ti
        if it == start_iter:
            print(json.dumps({"compile_plus_first_iter_s": round(time.time() - t_compile, 1), "setup_s": round(t_compile - t0, 1)}), flush=True)
        rec = {"iter": it + 1, "env_steps": (it + 1) * steps_per_iter, "steps_per_robot": (it + 1) * N * T,
               "sps": round(steps_per_iter / dt), "iter_s": round(dt, 2), "wall_s": round(time.time() - t0, 1),
               **{k: float(v) for k, v in met.items()},
               **{f"{r}/{k}": float(v) for r, s in stats.items() for k, v in s.items()}}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if (it + 1) % 5 == 0 or it == start_iter + 1:
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if a.bench_iters == 0 and ((it + 1) % a.save_every == 0 or it + 1 == nr_iters):
            tmp = out / "checkpoint.pkl.tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"pparams": jax.device_get(pstate.params), "popt": jax.device_get(pstate.opt_state),
                             "pstep": int(pstate.step), "cparams": jax.device_get(cstate.params),
                             "copt": jax.device_get(cstate.opt_state), "cstep": int(cstate.step),
                             "iter": it + 1, "key": jax.device_get(key)}, f)
            os.replace(tmp, ckpt)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
