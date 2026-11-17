"""Per-robot static spec for the MJX port: parameters come from the JSON that
extract_robot_config.py dumps from the real CPU env; the few structural facts that live only in
the robots' Python code (reward collision lists, termination rule, symmetry pairs, DR geom ids)
are listed in STRUCT below, copied from the CPU sources (file:line given)."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CFG_ROOT = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs")
ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"

STRUCT = {
    # environments/unitree_go1/reward_functions/rudin_own_var.py:122-125, termination rudin2022 (trunk group)
    # DR foot geoms from seen_robot_functions/default.py:94 + mujoco_model_functions/default.py
    # NOTE upstream uses geom 19 (not FR_foot=20) for FR foot size/friction DR; kept as-is.
    "unitree_go1": dict(
        foot_groups=["FR_foot", "FL_foot", "RR_foot", "RL_foot"],
        reward_collision_geoms=["FR_calf1", "FL_calf1", "RR_calf1", "RL_calf1"],
        reward_trunk_groups=["trunk_1", "trunk_2", "trunk_3"],
        termination=("contact", ["trunk"]),
        symmetry_pairs=[(0, 1), (2, 3)],
        nominal_diff_joints=[],
        dr_foot_geoms=[19, 31, 43, 55], foot_size_noise_power=1,
    ),
    # environments/unitree_h1/reward_functions/rudin_own_var.py (nr_collisions = 0; nominal diff qpos[17:])
    # termination_functions/height.py: qpos[2] < 0.545; seen_robot default.py applies foot noise twice
    "unitree_h1": dict(
        foot_groups=["left_foot", "right_foot"],
        reward_collision_geoms=[],
        reward_trunk_groups=[],
        termination=("height", 0.545),
        symmetry_pairs=[(0, 1)],
        nominal_diff_joints=list(range(10, 19)),
        dr_foot_geoms=[14, 15, 16, 29, 30, 31], foot_size_noise_power=2,
    ),
    # environments/hexapod/reward_functions/rudin_own_var.py:128-133; termination trunk_and_hip_collision.py
    "hexapod": dict(
        foot_groups=["foot_0", "foot_1", "foot_2", "foot_3", "foot_4", "foot_5"],
        reward_collision_geoms=[f"hip_{i}" for i in range(6)] + [f"knee_{i}" for i in range(6)],
        reward_trunk_groups=[f"trunk_{i}" for i in range(1, 8)],
        termination=("contact", ["trunk", "hip"]),
        symmetry_pairs=[(0, 1), (2, 3), (4, 5)],
        nominal_diff_joints=[],
        dr_foot_geoms=[24, 34, 44, 54, 64, 74], foot_size_noise_power=1,
    ),
}


@dataclass
class RobotSpec:
    name: str
    cfg: dict
    struct: dict

    @property
    def xml_path(self):
        return ENV_ROOT / self.name / "data" / self.cfg["xml_file"]

    def group_geoms(self, names):
        ids = set()
        for n in names:
            ids |= set(self.cfg["collision_groups"][n])
        return sorted(ids)

    def arr(self, *path):
        x = self.cfg
        for p in path:
            x = x[p]
        return np.asarray(x, dtype=np.float64)


def load_spec(name):
    cfg = json.loads((CFG_ROOT / f"{name}.json").read_text())
    return RobotSpec(name, cfg, STRUCT[name])
