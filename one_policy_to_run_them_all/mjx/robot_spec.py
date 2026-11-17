"""Per-robot static spec for the MJX port.

Everything comes from the JSON that extract_robot_config.py dumps from the real CPU env (CPU venv):
numeric parameters are read off the instantiated env objects, and the structural facts that live
only in each robot's Python code (reward collision lists, termination rule, symmetry pairs,
nominal-diff joints, DR geom ids / variants, foot-observation mode) are parsed from the robot's
sources by robot_struct.py (every parser asserts the source matches a known template).
No per-robot table is hand-written here.

The XML used for a robot can be overridden by the model manifest (robot_models_final.json).
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CFG_ROOT = Path(os.environ.get("URMA_MJX_CFG", "/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2"))
ENV_ROOT = Path(__file__).resolve().parents[1] / "environments"
MANIFEST = Path("/network/scratch/c/charlie.gauthier/urma_context/robot_models_final.json")

TRAIN_16 = ("unitree_a1", "unitree_go1", "unitree_go2", "anymal_b", "anymal_c", "barkour_v0", "barkour_vb", "badger",
            "bittle", "unitree_h1", "unitree_g1", "talos", "robotis_op3", "nao_v5", "cassie", "hexapod")
EVAL_4 = ("honey_badger", "badger_locked", "atlas", "sea_snake")


@dataclass
class RobotSpec:
    name: str
    cfg: dict

    @property
    def struct(self):
        return self.cfg["struct"]

    @property
    def xml_path(self):
        if MANIFEST.is_file():
            ent = json.loads(MANIFEST.read_text()).get(self.name)
            if ent:
                return Path(ent["xml"])
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
    return RobotSpec(name, cfg)
