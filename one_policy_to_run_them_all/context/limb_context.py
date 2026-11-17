"""Per-limb context arms for URMA joint / foot description tokens.

A limb is a joint plus its child body (the body the joint is attached to in MuJoCo,
``model.jnt_bodyid``; 1:1 for all 20 robots, checked at startup). Every robot's
``environment.py::get_name_to_description_vector`` calls :func:`compute_name_to_description_vector`;
nothing here is robot specific.

Arms (``environment.context_arm``)
----------------------------------
``urma``   stock URMA description, bit-identical to upstream 7c18194 (default; 23 / 10 wide).
``dk``     [K | D | G]                     non-latent family, 53 wide
``k``      [K | 0 | G]                     non-latent family
``d``      [0 | D | G]                     non-latent family
``empty``  [0 | 0 | 0]                     non-latent family: the whole token is zeros
``vme``    [VME latent | D | G]            latent family, L + 17 wide
``vmestar`` [VME* (fine-tuned VME) latent | D | G]  latent family
``matme``  [MatME latent | D | G]          latent family

Within a family every arm has the same width, so the policy / critic networks are identical
(same parameter shapes); masked blocks are zeros.

G = the robot-global copies URMA appends to every token (p_gain, d_gain, action_scaling_factor,
total mass, length/width/height). They stay in the tokens of every arm EXCEPT ``empty`` (user
decision 2026-11-16): ``empty`` joint and foot descriptions are literally all zeros, so all tokens of
a robot are identical (the honest no-context baseline, no joint IDs). State observations and the
robot-global fields in ``general_state`` are untouched in every arm.

Foot tokens always keep URMA's shape: [foot position (3) | G (7)]. Foot position is kinematics,
so it is zeroed in every arm without K (``empty``, ``d``, ``vme``, ``vmestar``, ``matme``); G is zeroed in ``empty``.

D values use URMA's domain-randomised "seen" values (``seen_body_mass``, ``seen_joint_*``) and are
recomputed whenever ``handle_domain_randomization`` resamples the robot (it calls
get_name_to_description_vector). Model-geometry K fields (body pos/ipos/iquat, geom quat/type/size)
are read from the nominal model cached at the first call, so the hidden "unseen" noise DR writes into
``model`` (e.g. foot geom size, trunk COM) never leaks into the context.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


NON_LATENT_ARMS = ("dk", "k", "d", "empty")
LATENT_ARMS = ("vme", "vmestar", "matme", "vme_sa", "vmestar_sa")
ALL_ARMS = ("urma",) + NON_LATENT_ARMS + LATENT_ARMS


def is_latent_arm(arm):
    """Latent arms are generic: any arm name other than urma / dk / k / d / empty is read from
    <latent_dir>/<arm>/<robot>.json (layout [latent | D | G])."""
    return arm not in ("urma",) + NON_LATENT_ARMS and bool(arm) and arm.replace("_", "").isalnum()


def latent_arms_available(latent_dir, robot):
    """Every latent arm with a file for this robot under latent_dir (for the startup guard)."""
    if not latent_dir or not Path(latent_dir).is_dir():
        return []
    return sorted(p.name for p in Path(latent_dir).iterdir() if p.is_dir() and is_latent_arm(p.name) and (p / f"{robot}.json").is_file())

# Geom types kept for the one-hot (plane / hfield / sdf never appear on robot limbs)
GEOM_TYPE_SLOTS = ("none", "sphere", "capsule", "ellipsoid", "cylinder", "box", "mesh")
_GEOM_TYPE_TO_SLOT = {
    int(mujoco.mjtGeom.mjGEOM_SPHERE): 1, int(mujoco.mjtGeom.mjGEOM_CAPSULE): 2,
    int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): 3, int(mujoco.mjtGeom.mjGEOM_CYLINDER): 4,
    int(mujoco.mjtGeom.mjGEOM_BOX): 5, int(mujoco.mjtGeom.mjGEOM_MESH): 6,
}

# (name, width) in order. Fields marked URMA keep URMA's normalisation verbatim.
K_FIELDS = (
    ("joint_pos", 3),          # URMA: joint anchor in trunk frame, bbox-normalised to [-1, 1]
    ("joint_axis", 3),         # URMA: joint axis in trunk frame
    ("joint_range", 2),        # URMA: seen joint range / 4.6
    ("nominal_pos", 1),        # URMA: seen nominal position / 4.6
    ("nr_child_joints", 1),    # URMA: nr direct child joints - 1
    ("body_pos", 3),           # child body pos in parent body frame / 0.5 m
    ("body_ipos", 3),          # child body COM offset in body frame / 0.5 m
    ("body_iquat", 4),         # child body inertial-frame quaternion
    ("geom_quat", 4),          # primary geom quaternion in body frame (zeros if no geom)
    ("geom_type", len(GEOM_TYPE_SLOTS)),  # primary geom type one-hot (slot 0 = no geom)
    ("geom_size", 3),          # primary geom size / 0.5 m (zeros if no geom)
    ("parent_limb", 1),        # parent limb index, ModuMorph-style over [0, 32]: (idx + 1) / 16 - 1; trunk -> -1
    ("depth", 1),              # nr ancestor limbs / 5
)
D_FIELDS = (
    ("link_mass", 1),          # (log10(seen child body mass) + 1) / 2
    ("link_inertia", 3),       # (log10(seen diag inertia) + 3) / 2
    ("torque_limit", 1),       # URMA
    ("max_joint_vel", 1),      # URMA
    ("damping", 1),            # URMA
    ("armature", 1),           # URMA
    ("stiffness", 1),          # URMA
    ("frictionloss", 1),       # URMA
)
G_FIELDS = (("p_d_gain_scaling", 3), ("robot_mass", 1), ("robot_dims", 3))

# URMA stock joint token order, expressed in the fields above
URMA_JOINT_ORDER = (
    "joint_pos", "joint_axis", "nr_child_joints", "nominal_pos",
    "torque_limit", "max_joint_vel", "damping", "armature", "stiffness", "frictionloss",
    "joint_range", "p_d_gain_scaling", "robot_mass", "robot_dims",
)


def _width(fields):
    return sum(w for _, w in fields)


K_WIDTH, D_WIDTH, G_WIDTH = _width(K_FIELDS), _width(D_FIELDS), _width(G_FIELDS)
_BLOCK_OF = {f: "K" for f, _ in K_FIELDS} | {f: "D" for f, _ in D_FIELDS} | {f: "G" for f, _ in G_FIELDS}
_FIELD_WIDTH = dict(K_FIELDS + D_FIELDS + G_FIELDS)

class _ArmBlocks(dict):
    def __missing__(self, arm):
        if is_latent_arm(arm):
            return {"LAT", "D", "G"}
        raise KeyError(arm)


ARM_BLOCKS_ON = _ArmBlocks({
    "urma": {"K", "D", "G"}, "dk": {"K", "D", "G"}, "k": {"K", "G"}, "d": {"D", "G"}, "empty": set(),
})


@dataclass(frozen=True)
class Layout:
    """Column layout of one arm's token: ordered (block, field, start, stop)."""
    arm: str
    columns: tuple
    width: int

    def block_columns(self, block):
        idx = [np.arange(s, e) for b, _, s, e in self.columns if b == block]
        return np.concatenate(idx) if idx else np.zeros(0, dtype=int)

    def field_slice(self, field):
        for _, f, s, e in self.columns:
            if f == field:
                return slice(s, e)
        raise KeyError(field)


def joint_layout(arm, latent_dim=0):
    if arm == "urma":
        blocks = [(_BLOCK_OF[f], ((f, _FIELD_WIDTH[f]),)) for f in URMA_JOINT_ORDER]
    elif arm in NON_LATENT_ARMS:
        blocks = [("K", K_FIELDS), ("D", D_FIELDS), ("G", G_FIELDS)]
    elif is_latent_arm(arm):
        if latent_dim <= 0:
            raise ValueError(f"arm {arm} needs a latent dim > 0")
        blocks = [("LAT", ((f"{arm}_latent", latent_dim),)), ("D", D_FIELDS), ("G", G_FIELDS)]
    else:
        raise ValueError(f"unknown context_arm {arm!r}; expected one of {ALL_ARMS}")
    columns, start = [], 0
    for block, fields in blocks:
        for f, w in fields:
            columns.append((block, f, start, start + w))
            start += w
    return Layout(arm, tuple(columns), start)


FOOT_LAYOUT = Layout("foot", (("K", "foot_pos", 0, 3), ("G", "p_d_gain_scaling", 3, 6),
                              ("G", "robot_mass", 6, 7), ("G", "robot_dims", 7, 10)), 10)


# --------------------------------------------------------------------------- latents

_LATENT_CACHE = {}


def robot_dir_name(env):
    """Folder name under environments/ (the latent file key), from the env class's module path."""
    return type(env).__module__.split(".")[-2]


def latent_path(latent_dir, arm, robot):
    return Path(latent_dir) / arm / f"{robot}.json"


def load_latents(latent_dir, arm, robot, model):
    """{body name: 1-D array} covering every body of the robot (world excluded). Fails loudly."""
    if not latent_dir:
        raise ValueError(f"context_arm={arm} needs environment.latent_dir")
    path = latent_path(latent_dir, arm, robot)
    key = path.as_posix()
    if key not in _LATENT_CACHE:
        if not path.is_file():
            raise FileNotFoundError(f"missing {arm} latent file for {robot}: {path}")
        with open(path) as f:
            raw = json.load(f)
        latents = {k: np.asarray(v, dtype=np.float64) for k, v in raw.items()}
        shapes = {v.shape for v in latents.values()}
        if len(shapes) != 1 or len(next(iter(shapes))) != 1 or next(iter(shapes))[0] == 0:
            raise ValueError(f"{path}: latents must be non-empty 1-D vectors of one common length, got {shapes}")
        body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(1, model.nbody)]
        missing = [b for b in body_names if b not in latents]
        if missing:
            raise KeyError(f"{path}: no latent for bodies {missing}")
        bad = [k for k, v in latents.items() if not np.all(np.isfinite(v))]
        if bad:
            raise ValueError(f"{path}: non-finite latents for bodies {bad}")
        _LATENT_CACHE[key] = latents
    return _LATENT_CACHE[key]


# --------------------------------------------------------------------------- static per-robot data

def _primary_geom(model, body):
    """First collision geom of the body (contype|conaffinity != 0), else its first geom, else -1."""
    adr, num = model.body_geomadr[body], model.body_geomnum[body]
    if num == 0:
        return -1
    for g in range(adr, adr + num):
        if model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0:
            return g
    return adr


def _static_limb_data(env):
    """Nominal-model kinematic data, cached on the env at the first call (before any DR sample)."""
    cached = getattr(env, "_limb_context_static", None)
    if cached is not None:
        return cached
    m = env.model
    joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in env.joint_names]
    bodies = np.array([m.jnt_bodyid[j] for j in joint_ids])
    if len(set(bodies.tolist())) != len(bodies):
        raise ValueError(f"{robot_dir_name(env)}: limbs are not 1:1 joint/body")
    body_to_limb = {int(b): i for i, b in enumerate(bodies)}
    trunk = m.body("trunk").id
    parent, depth = np.full(len(bodies), -1.0), np.zeros(len(bodies))
    for i, b in enumerate(bodies):
        p = m.body_parentid[b]
        while p != trunk and p != 0:
            if int(p) in body_to_limb:
                if parent[i] == -1:
                    parent[i] = body_to_limb[int(p)]
                depth[i] += 1
            p = m.body_parentid[p]
    geom_type = np.zeros((len(bodies), len(GEOM_TYPE_SLOTS)))
    geom_quat, geom_size = np.zeros((len(bodies), 4)), np.zeros((len(bodies), 3))
    for i, b in enumerate(bodies):
        g = _primary_geom(m, b)
        if g < 0:
            geom_type[i, 0] = 1.0
            continue
        gt = int(m.geom_type[g])
        if gt not in _GEOM_TYPE_TO_SLOT:
            raise ValueError(f"{robot_dir_name(env)}: unsupported geom type {gt} on limb {env.joint_names[i]}")
        geom_type[i, _GEOM_TYPE_TO_SLOT[gt]] = 1.0
        geom_quat[i] = m.geom_quat[g]
        geom_size[i] = m.geom_size[g]
    static = dict(
        bodies=bodies,
        body_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
        body_pos=m.body_pos[bodies].copy(), body_ipos=m.body_ipos[bodies].copy(),
        body_iquat=m.body_iquat[bodies].copy(),
        default_body_mass=m.body_mass[bodies].copy(), default_body_inertia=m.body_inertia[bodies].copy(),
        geom_quat=geom_quat, geom_type=geom_type, geom_size=geom_size,
        parent=parent, depth=depth,
    )
    env._limb_context_static = static
    return static


# --------------------------------------------------------------------------- field computation

def _robot_geometry(env, foot_geom_names):
    """URMA's bbox / trunk-frame computation, verbatim. Sets env.robot_* / gains / mass attributes.

    Returns (global fields G, foot positions, per-joint (pos, axis)) normalised exactly like URMA.
    """
    trunk_pos = env.data.body("trunk")
    trunk_rot_t = trunk_pos.xmat.reshape(3, 3).T

    relative_geom_positions = env.data.geom_xpos - trunk_pos.xpos
    for i in range(len(relative_geom_positions)):
        relative_geom_positions[i] = np.matmul(trunk_rot_t, relative_geom_positions[i])

    relative_joint_positions = np.zeros((len(env.joint_names), 3))
    for i, joint in enumerate(env.joint_names):
        relative_joint_positions[i] = env.data.joint(joint).xanchor - trunk_pos.xpos
        relative_joint_positions[i] = np.matmul(trunk_rot_t, relative_joint_positions[i])

    rotated_geom_sizes = np.zeros_like(env.model.geom_size)
    for i in range(len(rotated_geom_sizes)):
        global_frame_geom_size = np.matmul(env.data.geom_xmat[i].reshape(3, 3), env.model.geom_size[i])
        trunk_frame_geom_size = np.matmul(trunk_rot_t, global_frame_geom_size)
        rotated_geom_sizes[i] = np.abs(trunk_frame_geom_size)

    minus = relative_geom_positions - rotated_geom_sizes
    plus = relative_geom_positions + rotated_geom_sizes
    # Ignore the first geom (floor)
    mins = np.array([min(min(minus[1:, a]), min(plus[1:, a]), min(relative_joint_positions[:, a])) for a in range(3)])
    maxs = np.array([max(max(minus[1:, a]), max(plus[1:, a]), max(relative_joint_positions[:, a])) for a in range(3)])
    env.robot_length = maxs[0] - mins[0]
    env.robot_width = maxs[1] - mins[1]
    env.robot_height = maxs[2] - mins[2]
    env.robot_dimensions = np.array([env.robot_length, env.robot_width, env.robot_height])
    env.gains_and_action_scaling_factor = np.array([env.control_function.seen_p_gain, env.control_function.seen_d_gain, env.control_function.seen_scaling_factor])
    env.mass = np.array([np.sum(env.domain_randomization_seen_robot_function.seen_body_mass)])

    g = {
        "p_d_gain_scaling": (env.gains_and_action_scaling_factor / [100.0 / 2, 2.0 / 2, 0.8 / 2]) - 1.0,
        "robot_mass": (env.mass / (170.0 / 2)) - 1.0,
        "robot_dims": (env.robot_dimensions / (2.0 / 2)) - 1.0,
    }

    def to_trunk_normalised(world_pos):
        rel = np.matmul(trunk_rot_t, world_pos - trunk_pos.xpos)
        return (((rel - mins) / env.robot_dimensions) / 0.5) - 1.0

    feet = {name: to_trunk_normalised(env.data.geom(geom).xpos) for name, geom in zip(env.foot_names, foot_geom_names)}
    joints = []
    for joint_name in env.joint_names:
        joint = env.data.joint(joint_name)
        joints.append((to_trunk_normalised(joint.xanchor), np.matmul(trunk_rot_t, joint.xaxis)))
    return g, feet, joints


def limb_fields(env, dof_idx, jnt_idx, joints_geo):
    """All per-limb K and D fields (unmasked): {field: (n_limbs, width)}, plus the static data."""
    s = env.domain_randomization_seen_robot_function
    st = _static_limb_data(env)
    n = len(env.joint_names)
    seen_mass = s.seen_body_mass[st["bodies"]]
    seen_inertia = st["default_body_inertia"] * (seen_mass / st["default_body_mass"])[:, None]
    col = lambda x: np.asarray(x, dtype=np.float64).reshape(n, 1)
    f = {
        "joint_pos": np.stack([p for p, _ in joints_geo]),
        "joint_axis": np.stack([a for _, a in joints_geo]),
        "joint_range": s.seen_joint_range[jnt_idx] / 4.6,
        "nominal_pos": col(s.seen_joint_nominal_position / 4.6),
        "nr_child_joints": col(env.joint_nr_direct_child_joints) - 1.0,
        "body_pos": st["body_pos"] / 0.5,
        "body_ipos": st["body_ipos"] / 0.5,
        "body_iquat": st["body_iquat"],
        "geom_quat": st["geom_quat"],
        "geom_type": st["geom_type"],
        "geom_size": st["geom_size"] / 0.5,
        "parent_limb": col((st["parent"] + 1.0) / 16.0 - 1.0),
        "depth": col(st["depth"] / 5.0),
        "link_mass": col((np.log10(seen_mass) + 1.0) / 2.0),
        "link_inertia": (np.log10(np.maximum(seen_inertia, 1e-12)) + 3.0) / 2.0,
        "torque_limit": col((s.seen_torque_limit / (1000.0 / 2)) - 1.0),
        "max_joint_vel": col((s.seen_joint_velocity / (35.0 / 2)) - 1.0),
        "damping": col((s.seen_joint_damping[dof_idx] / (10.0 / 2)) - 1.0),
        "armature": col((s.seen_joint_armature[dof_idx] / (0.2 / 2)) - 1.0),
        "stiffness": col((s.seen_joint_stiffness[jnt_idx] / (30.0 / 2)) - 1.0),
        "frictionloss": col((s.seen_joint_frictionloss[dof_idx] / (1.2 / 2)) - 1.0),
    }
    return f, st


def assemble_joint_matrix(arm, fields, g, latents=None):
    """(n_limbs, width) joint description matrix of one arm, from precomputed fields."""
    n = fields["joint_pos"].shape[0]
    layout = joint_layout(arm, 0 if latents is None else latents.shape[1])
    out = np.zeros((n, layout.width))
    on = ARM_BLOCKS_ON[arm]
    for block, f, s, e in layout.columns:
        if block not in on:
            continue
        if block == "G":
            out[:, s:e] = g[f][None, :]
        elif block == "LAT":
            out[:, s:e] = latents
        else:
            out[:, s:e] = fields[f]
    return out, layout


def limb_latents(latent_dir, arm, env, body_names):
    lat = load_latents(latent_dir, arm, robot_dir_name(env), env.model)
    return np.stack([lat[b] for b in body_names])


def check_arms_distinct(env, fields, g, st, arm, latent_dir):
    """Startup guard (cf. ModuMorph's silent arm collapse): fail loudly if two arms give the same
    joint inputs for this robot, or if a block an arm switches on is all zeros."""
    mats = {a: assemble_joint_matrix(a, fields, g) for a in NON_LATENT_ARMS}
    for la in sorted(set(latent_arms_available(latent_dir, robot_dir_name(env))) | ({arm} if is_latent_arm(arm) else set())):
        if True:
            mats[la] = assemble_joint_matrix(la, fields, g, limb_latents(latent_dir, la, env, st["body_names"]))
    names = list(mats)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if mats[a][0].shape == mats[b][0].shape and np.array_equal(mats[a][0], mats[b][0]):
                raise RuntimeError(f"{robot_dir_name(env)}: context arms {a} and {b} give identical joint descriptions")
    if np.any(mats["empty"][0]):
        raise RuntimeError(f"{robot_dir_name(env)}: empty arm joint description is not all zeros")
    for a in names:
        mat, lay = mats[a]
        for block in ARM_BLOCKS_ON[a] - {"G"}:
            if not np.any(mat[:, lay.block_columns(block)]):
                raise RuntimeError(f"{robot_dir_name(env)}: arm {a} block {block} is all zeros")


def compute_name_to_description_vector(env, default_qpos_qvel, foot_geom_suffix="", dof_idx=None, jnt_idx=None):
    """Drop-in body of every robot's get_name_to_description_vector.

    default_qpos_qvel: the robot's default initial state (description is computed in that pose).
    dof_idx / jnt_idx map env.joint_names[i] into the seen_joint_* arrays indexed by dof (damping,
    armature, frictionloss) and by joint (stiffness, range). Default arange(n) (every joint is
    actuated); cassie / nao_v5 / unitree_g1 pass their passive-joint masks.
    """
    arm = getattr(env, "context_arm", "urma")
    latent_dir = getattr(env, "latent_dir", "")
    n = len(env.joint_names)
    dof_idx = np.arange(n) if dof_idx is None else np.asarray(dof_idx)
    jnt_idx = np.arange(n) if jnt_idx is None else np.asarray(jnt_idx)
    foot_geom_names = [f + foot_geom_suffix for f in env.foot_names]

    # Save initial state, move to the default pose
    qpos_old, qvel_old = env.data.qpos.copy(), env.data.qvel.copy()
    qpos, qvel = default_qpos_qvel
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    mujoco.mj_forward(env.model, env.data)

    g, feet, joints_geo = _robot_geometry(env, foot_geom_names)
    fields, st = limb_fields(env, dof_idx, jnt_idx, joints_geo)
    latents = limb_latents(latent_dir, arm, env, st["body_names"]) if is_latent_arm(arm) else None
    joint_matrix, _ = assemble_joint_matrix(arm, fields, g, latents)

    if not getattr(env, "_limb_context_checked", False):
        check_arms_distinct(env, fields, g, st, arm, latent_dir)
        env._limb_context_checked = True

    out = {}
    foot_k_on, foot_g_on = "K" in ARM_BLOCKS_ON[arm], "G" in ARM_BLOCKS_ON[arm]
    for name in env.foot_names:
        foot_g = np.concatenate([g["p_d_gain_scaling"], g["robot_mass"], g["robot_dims"]])
        out[name] = np.concatenate([feet[name] if foot_k_on else np.zeros(3),
                                    foot_g if foot_g_on else np.zeros(7)])
    for i, name in enumerate(env.joint_names):
        out[name] = joint_matrix[i]

    env.dynamic_joint_description_size = out[env.joint_names[0]].shape[0]
    env.dynamic_foot_description_size = out[env.foot_names[0]].shape[0]

    # Restore initial state
    env.data.qpos[:] = qpos_old
    env.data.qvel[:] = qvel_old
    mujoco.mj_forward(env.model, env.data)
    return out
