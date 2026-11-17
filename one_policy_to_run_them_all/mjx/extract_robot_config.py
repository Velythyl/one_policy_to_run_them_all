"""Run in the CPU venv (/network/scratch/c/charlie.gauthier/urma_context/venv): instantiate the real
CPU robot env and dump every scalar/array parameter the MJX port needs into a JSON per robot.
This avoids hand-transcribing ~100 per-robot constants.

    python extract_robot_config.py <out_dir> robot [robot ...]
"""
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "experiments" / "context_tools"))
from envs import build_env  # noqa: E402
from one_policy_to_run_them_all.context import limb_context as lc  # noqa: E402
from one_policy_to_run_them_all.mjx.robot_struct import extract_struct  # noqa: E402
import inspect, re  # noqa: E402

LATENTS_DUMMY = "/network/scratch/c/charlie.gauthier/urma_context/latents_dummy"


def scalars(obj):
    out = {}
    for k, v in vars(obj).items():
        if k == "env" or k.startswith("_"):
            continue
        if isinstance(v, (bool, int, float, np.floating, np.integer)):
            out[k] = float(v)
    return out


def tolist(x):
    return np.asarray(x, dtype=np.float64).tolist()


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    for robot in sys.argv[2:]:
        env = build_env(robot, seed=0, context_arm="dk")
        m = env.model
        s = env.domain_randomization_seen_robot_function
        # description pieces at the nominal pose (default values, before any DR sample)
        n = len(env.joint_names)
        dof_idx = getattr(env, "joint_mask_qvel", np.arange(6, 6 + n)) - 6
        jnt_idx = getattr(env, "joint_mask_jnt_vals", np.arange(1, 1 + n)) - 1
        qpos0, qvel0 = env.data.qpos.copy(), env.data.qvel.copy()
        from importlib import import_module
        init_mod = import_module(f"one_policy_to_run_them_all.environments.{robot}.initial_state_functions.handler")
        qp, qv = init_mod.get_initial_state_function("default", env).setup()
        env.data.qpos[:] = qp; env.data.qvel[:] = qv
        mujoco.mj_forward(m, env.data)
        m_suf = re.search(r'foot_geom_suffix="(\w*)"', inspect.getsource(type(env)))
        suffix = m_suf.group(1) if m_suf else ""
        g, feet, joints_geo = lc._robot_geometry(env, [f + suffix for f in env.foot_names])
        fields, st = lc.limb_fields(env, dof_idx, jnt_idx, joints_geo)
        env.data.qpos[:] = qpos0; env.data.qvel[:] = qvel0
        mujoco.mj_forward(m, env.data)
        cfg = {
            "robot": robot,
            "xml": str(Path(type(env).__module__.replace(".", "/")).parent),
            "xml_file": env.terrain_function.xml_file_name,
            "joint_names": env.joint_names, "foot_names": env.foot_names,
            "joint_nr_direct_child_joints": list(map(float, env.joint_nr_direct_child_joints)),
            "nominal_joint_positions": tolist(env.nominal_joint_positions),
            "max_joint_velocities": tolist(env.max_joint_velocities),
            "initial_drop_height": float(env.initial_drop_height),
            "dt": float(env.dt), "nr_substeps": int(env.nr_substeps), "horizon": int(env.horizon),
            "timestep": float(m.opt.timestep),
            "qpos_joint_idx": tolist(getattr(env, "joint_mask_qpos", np.arange(7, 7 + n))),
            "qvel_joint_idx": tolist(getattr(env, "joint_mask_qvel", np.arange(6, 6 + n))),
            "jnt_idx": tolist(jnt_idx + 1),
            "complete_nominal_joint_position": tolist(getattr(env, "complete_nominal_joint_position", [])),
            "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
            "collision_groups": {k: sorted(int(i) for i in v) for k, v in env.collision_groups.items()},
            "params": {
                "control": scalars(env.control_function),
                "reward": scalars(env.reward_function),
                "seen": scalars(s),
                "unseen": scalars(env.domain_randomization_unseen_robot_function),
                "model_dr": scalars(env.domain_randomization_mujoco_model_function),
                "action_delay": scalars(env.domain_randomization_action_delay_function),
                "perturbation": scalars(env.domain_randomization_perturbation_function),
                "init": scalars(env.initial_state_function),
                "command": scalars(env.command_function),
                "noise": scalars(env.observation_noise_function),
                "dropout": scalars(env.observation_dropout_function),
                "dr_sampling_prob": float(env.domain_randomization_sampling_function.probability),
                "perturb_sampling_prob": float(env.domain_randomization_perturbation_sampling_function.probability),
                "command_sampling_prob": float(env.command_sampling_function.probability),
                "termination_type": type(env.termination_function).__name__,
                "termination": scalars(env.termination_function),
            },
            "defaults": {
                "body_mass": tolist(m.body_mass), "body_inertia": tolist(m.body_inertia), "body_ipos": tolist(m.body_ipos),
                "actuator_ctrlrange": tolist(m.actuator_ctrlrange), "jnt_range": tolist(m.jnt_range),
                "dof_damping": tolist(m.dof_damping), "dof_armature": tolist(m.dof_armature),
                "jnt_stiffness": tolist(m.jnt_stiffness), "dof_frictionloss": tolist(m.dof_frictionloss),
                "geom_friction": tolist(m.geom_friction), "geom_size": tolist(m.geom_size),
                "geom_solref": tolist(m.geom_solref), "gravity": tolist(m.opt.gravity),
            },
            # limb_context pieces at the nominal pose; the MJX env recomputes the DR-dependent ones
            "context": {
                "fields": {k: tolist(v) for k, v in fields.items()},
                "g": {k: tolist(v) for k, v in g.items()},
                "feet": {k: tolist(v) for k, v in feet.items()},
                "body_names": st["body_names"],
                "default_body_mass_limbs": tolist(st["default_body_mass"]),
                "default_body_inertia_limbs": tolist(st["default_body_inertia"]),
                "limb_bodies": tolist(st["bodies"]),
                "robot_dimensions": tolist(env.robot_dimensions),
                "dof_idx": tolist(dof_idx), "jnt_idx": tolist(jnt_idx),
            },
            # full CPU description per arm at default values (for verification of the JAX port)
            "cpu_descriptions": {},
            "foot_geom_suffix": suffix,
            "eq": {"neq": int(m.neq), "types": [int(t) for t in m.eq_type], "active": [int(a) for a in m.eq_active0]},
        }
        for arm in ("urma", "dk", "k", "d", "empty", "vme", "vmestar", "matme"):
            try:
                e2 = build_env(robot, seed=0, context_arm=arm, latent_dir=LATENTS_DUMMY if arm in lc.LATENT_ARMS else "")
            except Exception as ex:  # latent file missing for this robot
                print(robot, arm, "SKIP", repr(ex)[:200], flush=True)
                continue
            cfg["cpu_descriptions"][arm] = {
                "joints": tolist(np.stack([e2.name_to_description_vector[j] for j in e2.joint_names])),
                "feet": tolist(np.stack([e2.name_to_description_vector[f] for f in e2.foot_names])),
            }
            e2.close()
        cfg["struct"] = extract_struct(build_env(robot, seed=0, context_arm="urma"))
        (out_dir / f"{robot}.json").write_text(json.dumps(cfg))
        print(robot, "ok", env.termination_function.__class__.__name__, flush=True)
        env.close()


if __name__ == "__main__":
    main()
