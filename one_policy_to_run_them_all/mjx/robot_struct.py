"""Structural per-robot facts the MJX port needs that live only in each robot's Python code.

Parsed from the robot's own CPU sources (reward / termination / DR functions) and, where parsing
would be fragile, probed by executing the CPU env (observation timer order, foot-observation mode).
Every parser asserts that the source matches one of the known templates, so a robot whose code
deviates fails loudly instead of being ported silently wrong.  Run in the CPU venv
(called by extract_robot_config.py).
"""
import inspect
import re

import numpy as np

NAME_LIST = re.compile(r'"([A-Za-z0-9_]+)"')


def _src(obj):
    return inspect.getsource(type(obj))


def _names(s):
    return NAME_LIST.findall(s)


def _index_expr(expr, n):
    """Evaluate a numpy index expression like '12:' / ':14' / '[0, 1, 14]' on arange(n)."""
    ar = np.arange(n)
    return [int(i) for i in np.atleast_1d(eval(f"ar[{expr}]", {"ar": ar}))]  # noqa: S307 (repo source)


def parse_reward(env):
    src = _src(env.reward_function)
    n, feet = len(env.joint_names), list(env.foot_names)
    out = {}
    # collisions counted in the reward
    m = re.search(r"check_any_collision_for_all\(\[(.*?)\]\)", src, re.S)
    out["reward_collision_geoms"] = _names(m.group(1)) if m else []
    m = re.search(r"trunk_collision = 1 if self\.env\.check_any_collision\(\[(.*?)\]\)", src, re.S)
    out["reward_trunk_groups"] = _names(m.group(1)) if m else []
    if re.search(r"^\s*nr_collisions = 0\s*$", src, re.M):
        assert not out["reward_collision_geoms"] and not out["reward_trunk_groups"]
    else:
        assert re.search(r"nr_collisions = sum\(collisions\.values\(\)\) \+ trunk_collision", src), env.LONG_NAME
    # timers: attribute -> foot name (reward step())
    timer_foot = dict(re.findall(r'self\.(time_since_last_touchdown_\w+) = 0 if self\.env\.check_collision\("floor", "(\w+)"\)', src))
    on_ground = dict(re.findall(r'(foot_\w+_on_ground) = self\.env\.check_collision\("floor", "(\w+)"\)', src))
    air = re.findall(r"if (foot_\w+_on_ground)(?: == 1)?:\s*\n\s*air_time_reward \+= self\.(time_since_last_touchdown_\w+) - self\.air_time_max", src)
    sym = re.findall(r"if not (foot_\w+_on_ground) and not (foot_\w+_on_ground):\s*\n\s*symmetry_air_violations \+= 1", src)
    fidx = {f: i for i, f in enumerate(feet)}
    out["timer_attr_to_foot"] = {a: fidx[f] for a, f in timer_foot.items()}
    out["air_pairs"] = [[fidx[on_ground[v]], fidx[timer_foot[a]]] for v, a in air]
    out["symmetry_pairs"] = [[fidx[on_ground[a]], fidx[on_ground[b]]] for a, b in sym]
    out["timer_feet"] = sorted(set(out["timer_attr_to_foot"].values()))
    if timer_foot:
        assert len(air) == len(on_ground) == len(timer_foot), (env.LONG_NAME, air, on_ground, timer_foot)
    # joint nominal difference (joint indices into joint_names / seen nominal)
    if re.search(r"joint_nominal_diff_norm = 0\.0", src):
        nd = []
    else:
        m = re.search(r"joint_nominal_diff_norm = np\.sum\(np\.square\(self\.env\.data\.qpos\[(\d+):(\d*)\] - "
                      r"self\.env\.domain_randomization_seen_robot_function\.seen_joint_nominal_position\[(\d*):(\d*)\]\)\)", src)
        if m:
            q0, q1, n0, n1 = m.groups()
            nd = _index_expr(f"{n0}:{n1}", n)
            q = list(range(int(q0), int(q1) if q1 else 7 + n))
            assert [7 + j for j in nd] == q, (env.LONG_NAME, nd, q)
        else:
            m = re.search(r"joint_nominal_diff_norm = np\.sum\(np\.square\(self\.env\.data\.qpos\[self\.env\.joint_mask_qpos\]\[(.*?)\] - "
                          r"self\.env\.domain_randomization_seen_robot_function\.seen_joint_nominal_position\[(.*?)\]\)\)", src)
            assert m, env.LONG_NAME
            nd = _index_expr(m.group(1), n)
            assert nd == _index_expr(m.group(2), n), env.LONG_NAME
    out["nominal_diff_joints"] = nd
    # joint quantities: masked robots index qpos/qvel/qfrc with joint masks, others use [7:] / [6:]
    masked = "joint_mask_qvel" in src
    nv6 = env.model.nv - 6
    assert masked or nv6 == n, env.LONG_NAME
    return out


def parse_termination(env):
    t = env.termination_function
    name = type(t).__name__
    src = _src(t)
    if name == "Rudin2022Termination":
        assert 'ids = self.env.collision_groups["trunk"]' in src
        return {"kind": "contact", "groups": ["trunk"]}
    if name == "TrunkAndHipCollisionTermination":
        assert 'hip_ids = self.env.collision_groups["hip"]' in src and 'trunk_ids = self.env.collision_groups["trunk"]' in src
        return {"kind": "contact", "groups": ["trunk", "hip"]}
    if name == "HeightTermination":
        m = re.search(r"return self\.env\.data\.qpos\[2\] < ([0-9.]+)\s*$", src, re.M)
        if m:
            return {"kind": "height", "threshold": float(m.group(1))}
        m = re.search(r'return self\.env\.data\.geom\("(\w+)"\)\.xpos\[2\] < ([0-9.]+)\s*$', src, re.M)
        assert m, (env.LONG_NAME, src)
        import mujoco
        gid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, m.group(1))
        assert gid >= 0
        return {"kind": "geom_height", "geom": int(gid), "geom_name": m.group(1), "threshold": float(m.group(2))}
    if name == "NoneTerminationFunction":
        assert "return False" in src
        return {"kind": "none"}
    raise NotImplementedError(f"{env.LONG_NAME}: termination {name}")


def parse_seen_dr(env):
    src = _src(env.domain_randomization_seen_robot_function)
    out = {}
    m = re.search(r"self\.env\.model\.geom_size\[\[([\d, ]+)\],0\] = self\.env\.np_rng\.uniform\(self\.foot_size_min, self\.foot_size_max\)((?: \* self\.foot_size_noise_factor)+)\s*$", src, re.M)
    if m:
        out["foot_size_geoms"] = [int(x) for x in m.group(1).split(",")]
        out["foot_size_noise_power"] = m.group(2).count("foot_size_noise_factor")
    else:
        assert "geom_size" not in src.split("def sample")[1], env.LONG_NAME
        out["foot_size_geoms"], out["foot_size_noise_power"] = [], 0
    for q, attr in (("damping", "joint_damping"), ("frictionloss", "joint_frictionloss")):
        lo_hi = "joint_damping" if q == "damping" else "joint_friction_loss"
        if re.search(rf"self\.seen_{attr} = np\.maximum\(0, self\.default_{attr} \+ self\.env\.np_rng\.uniform\(self\.add_{lo_hi}_min, self\.add_{lo_hi}_max", src):
            out[f"{q}_mode"] = "add_clamp"
        elif re.search(rf"self\.seen_{attr} = self\.default_{attr} \* \(1 \+ self\.env\.np_rng\.uniform\(-self\.{lo_hi}_factor, self\.{lo_hi}_factor", src):
            out[f"{q}_mode"] = "factor"
        else:
            assert re.search(rf"self\.seen_{attr} = self\.env\.np_rng\.uniform\(self\.{lo_hi}_min, self\.{lo_hi}_max", src), (env.LONG_NAME, q)
            out[f"{q}_mode"] = "uniform"
    for q in ("armature", "stiffness"):
        lo_hi = f"joint_{q}"
        assert re.search(rf"self\.seen_joint_{q} = self\.env\.np_rng\.uniform\(self\.{lo_hi}_min, self\.{lo_hi}_max", src), (env.LONG_NAME, q)
    # indices that keep their default value even when not staying at default
    keep = {}
    for q, idx in re.findall(r"self\.seen_joint_(\w+)\[\[([\d, ]+)\]\] = self\.default_joint_\1\[\[\2\]\]", src):
        keep[q] = [int(x) for x in idx.split(",")]
    out["keep_default"] = keep
    # upstream typo: the stay branch assigns `self.seen_armature` (armature keeps its last sample)
    out["armature_stay_typo"] = "self.seen_armature = self.default_joint_armature" in src
    return out


def parse_model_dr(env):
    src = _src(env.domain_randomization_mujoco_model_function)
    m = re.search(r"self\.env\.model\.geom_friction\[\[([\d, ]+)\]\] = \[self\.sampled_friction_tangential, self\.sampled_friction_torsional_feet, self\.sampled_friction_rolling_feet\]", src)
    assert "self.env.model.geom_solref[:, 0] = -self.sampled_stiffness" in src and "self.env.model.geom_solref[:, 1] = -self.sampled_damping" in src
    assert "self.env.model.opt.gravity[2] = -self.sampled_gravity" in src
    if m:
        assert "self.env.model.geom_friction[0] = [self.sampled_friction_tangential, self.sampled_friction_torsional_ground, self.sampled_friction_rolling_ground]" in src
        return {"friction_mode": "ground_feet", "friction_feet_geoms": [int(x) for x in m.group(1).split(",")]}
    assert "self.env.model.geom_friction[:,] = [self.sampled_friction_tangential, self.sampled_friction_torsional, self.sampled_friction_rolling]" in src, env.LONG_NAME
    return {"friction_mode": "all", "friction_feet_geoms": []}


def parse_unseen_dr(env):
    src = _src(env.domain_randomization_unseen_robot_function)
    return {"has_foot_noise": "foot_size_noise_factor" in src}


def probe_foot_obs(env):
    """Execute the CPU env's observation code with sentinel timer values to find (a) which timer
    attribute fills each foot's timer slot and (b) whether foot contact/timer obs are live (updated
    every step), frozen (only refreshed when the domain is re-randomised: badger family) or missing
    (always missing_value: sea_snake)."""
    env.observation_noise_function.modify_observation = lambda o: o
    env.observation_dropout_function.modify_observation = lambda o: o
    rf = env.reward_function
    attrs = sorted(a for a in vars(rf) if a.startswith("time_since_last_touchdown_"))
    for k, a in enumerate(attrs):
        setattr(rf, a, 0.1 * (k + 1))
    idx = env.foot_time_since_last_ground_contact_update_obs_idx

    def slots(obs):
        v = (np.asarray(obs)[idx] + 1.0) * 2.5
        return [next((a for k, a in enumerate(attrs) if abs(v[i] - 0.1 * (k + 1)) < 1e-4), None) for i in range(len(idx))]

    live = slots(env.get_observation())
    if all(s is not None for s in live) and attrs:
        mode, order = "live", live
    else:
        env.initial_observation = env.get_initial_observation(getattr(env, "multi_robot_max_observation_size", -1))
        fr = slots(env.get_observation())
        if attrs and all(s is not None for s in fr):
            mode, order = "frozen", fr
        else:
            mode, order = "missing", []
    return mode, order


def extract_struct(env):
    s = {}
    s.update(parse_reward(env))
    s["termination"] = parse_termination(env)
    s["seen_dr"] = parse_seen_dr(env)
    s["model_dr"] = parse_model_dr(env)
    s["unseen_dr"] = parse_unseen_dr(env)
    mode, order = probe_foot_obs(env)
    s["foot_obs_mode"] = mode
    t2f = s["timer_attr_to_foot"]
    # obs slot i shows the timer of foot obs_timer_feet[i]
    s["obs_timer_feet"] = [t2f[a] for a in order]
    s["masked"] = hasattr(env, "joint_mask_qpos")
    return s
