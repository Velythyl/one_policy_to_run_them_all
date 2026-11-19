"""Unimal (ModuMorph / MetaMorph) -> URMA GPU robot: cleaned MJCF + a configs_v2-compatible robot config.

    python unimal_convert.py <out_dir> <unimal.xml> [<unimal.xml> ...]      (venv_mjx: mujoco + numpy only)

Writes <out_dir>/xml/<robot>.xml and <out_dir>/configs/<robot>.json, robot = "unimal_" + the Unimal id
(e.g. floor-1409-0-12-01-12-30-07 -> unimal_1409_0_12_01_12_30_07). The config has every key the generic
MJX env (mjx_env.py / robot_spec.py) reads, with the same layout extract_robot_config.py produces for the
20 URMA robots, plus "xml_path" (absolute) and a "unimal" block with the conversion provenance.

Conversion rules (see UNIMAL_PROBE.md for the reasoning):
  MJCF   - torso body "torso/0" renamed "trunk" (URMA convention; body 1); floor plane "floor" added as geom 0;
           sensors removed (URMA reads no sensors); option cone=pyramidal impratio=100 (URMA a1 template);
           default geom solref "-1000 -80" (= the URMA model-DR midpoint, written for every geom by the DR anyway);
           motors: gear g / ctrlrange [-1, 1] -> gear 1 / ctrlrange [-g, g] (URMA PD control writes torques;
           same torque limit as the Unimal). Everything else (timestep 0.005, filterparent disabled, joint
           damping / stiffness / armature / solimplimit, capsule geometry, densities, contact excludes) unchanged.
  joints - every hinge (all are actuated), MuJoCo order. Nominal = 0 (the Unimal's designed pose) clipped into
           the 90 % soft joint range URMA's limit penalty uses.
  feet   - leaf limbs (bodies without child bodies) whose lowest point at the nominal pose is within FOOT_TOL of the
           robot's lowest point; if fewer than 2 qualify, every leaf. Foot geom = the leaf's capsule.
  gains  - one p/d/action-scale per robot (URMA convention): see choose_gains().
  vel    - max joint velocity: MAX_JOINT_VEL for every joint (Unimals define none).
  heights- initial_drop_height = trunk z with the lowest point 1 cm above the floor at the nominal pose;
           nominal_trunk_z (base-height reward) = trunk z after settling 3 s under PD at nominal (MuJoCo C);
           termination = height < 0.5 x nominal_trunk_z (ModuMorph TerminateOnFalling, STAND_HEIGHT_RATIO 0.5).
  DR/rew - parameters copied from unitree_a1's config (quadruped template, rudin_own_var reward, a1 model DR),
           except the robot-specific ranges re-centred on the Unimal's values (see unimal_params()).
  context- limb_context._robot_geometry / limb_fields / assemble_joint_matrix on a light stand-in env. Unimal limbs can
           carry 2 hinges (limbx/N, limby/N) on one body, which limb_context's 1:1 joint/body check rejects; the
           per-limb static data is therefore built here with joint-level limbs sharing the body (see static_limb_data).
"""
import copy
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from one_policy_to_run_them_all.context import limb_context as lc

TEMPLATE_CFG = Path("/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2/unitree_a1.json")
FOOT_TOL = 0.05
MAX_JOINT_VEL = 20.0
LATENTS_DUMMY = "/network/scratch/c/charlie.gauthier/urma_context/latents_dummy"


def robot_name(xml_path):
    stem = Path(xml_path).stem
    return "unimal_" + re.sub(r"[^0-9a-zA-Z]+", "_", stem.replace("floor-", ""))


# ----------------------------------------------------------------------------- MJCF

def convert_xml(src):
    tree = ET.parse(src)
    root = tree.getroot()
    for s in root.findall("sensor"):
        root.remove(s)
    opt = root.find("option")
    opt.set("cone", "pyramidal")
    opt.set("impratio", "100")
    dflt = root.find("default")
    g = dflt.find("geom")
    g.set("solref", "-1000 -80")
    wb = root.find("worldbody")
    floor = ET.Element("geom", {"name": "floor", "type": "plane", "pos": "0 0 0", "size": "100 100 0.125",
                                "condim": "3", "material": "grid"})
    wb.insert(0, floor)
    torso = wb.find("body[@name='torso/0']")
    assert torso is not None
    torso.set("name", "trunk")
    for cam in torso.findall("camera"):  # trackcom camera kept for rendering; harmless
        pass
    act = root.find("actuator")
    gears = {}
    for mot in act.findall("motor"):
        gear = float(mot.get("gear", "1"))
        gears[mot.get("joint")] = gear
        mot.set("gear", "1")
        mot.set("ctrllimited", "true")
        mot.set("ctrlrange", f"{-gear:g} {gear:g}")
    return ET.tostring(root, encoding="unicode"), gears


# ----------------------------------------------------------------------------- kinematics helpers

def lowest_points(m, d):
    """Per geom (excluding the floor): lowest world z of the geom surface (capsules / spheres only)."""
    out = np.full(m.ngeom, np.inf)
    for g in range(1, m.ngeom):
        t, r = m.geom_type[g], m.geom_size[g, 0]
        if t == mujoco.mjtGeom.mjGEOM_SPHERE:
            out[g] = d.geom_xpos[g, 2] - r
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            ax = d.geom_xmat[g].reshape(3, 3)[:, 2] * m.geom_size[g, 1]
            out[g] = min(d.geom_xpos[g, 2] + ax[2], d.geom_xpos[g, 2] - ax[2]) - r
        else:
            raise ValueError(f"unexpected geom type {t}")
    return out


def pose(m, nominal, z=1.0):
    d = mujoco.MjData(m)
    d.qpos[:] = m.qpos0
    d.qpos[0:3] = [0, 0, z]
    d.qpos[3:7] = [1, 0, 0, 0]
    d.qpos[7:] = nominal
    mujoco.mj_forward(m, d)
    return d


def settle(m, nominal, p, dgain, z0, seconds=3.0, nsub=4):
    """MuJoCo C: hold the nominal pose with URMA's PD law (ctrl = torque, clipped by ctrlrange) from z0."""
    d = pose(m, nominal, z0)
    n_ctrl = int(seconds / (m.opt.timestep * nsub))
    zs, tilts = [], []
    for _ in range(n_ctrl):
        tau = p * (nominal - d.qpos[7:]) - dgain * d.qvel[6:]
        d.ctrl[:] = tau
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        R = d.xmat[1].reshape(3, 3)
        zs.append(d.qpos[2]); tilts.append(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))
    return np.asarray(zs), np.asarray(tilts), d


def choose_gains(m, nominal):
    """One (p, d, action scale) per robot, URMA-style.
    p: URMA's largest p gain (100, atlas) for every Unimal: Unimals carry 12-41 kg on capsule legs with 150-300 Nm
       motors and armature 1, i.e. URMA's heavy-robot regime; 100 is the top of URMA's G-context normalisation
       (p / 50 - 1 = 1). d: 2 x sqrt-scaled critical-ish damping is not reachable inside URMA's d range (<= 2);
       we use d = D_GAIN (see probe results) and rely on the Unimal's joint damping 1 + armature 1.
    scale: median joint half-range clipped to URMA's [0.25, 0.75] action-scale range."""
    half = (m.jnt_range[1:, 1] - m.jnt_range[1:, 0]) / 2
    return P_GAIN, D_GAIN, float(np.clip(np.median(half), 0.25, 0.75))


P_GAIN, D_GAIN = 100.0, 2.0


# ----------------------------------------------------------------------------- context (limb_context on a stand-in env)

def static_limb_data(m, joint_ids):
    """limb_context._static_limb_data with joint-level limbs: a body carrying 2 hinges gives 2 limbs that share
    the body's K fields / mass / latent. Parent of a limb = the previous hinge on the same body if any, else the
    last hinge of the nearest ancestor body (kinematic-chain semantics)."""
    bodies = np.array([m.jnt_bodyid[j] for j in joint_ids])
    trunk = m.body("trunk").id
    jpos = {int(j): i for i, j in enumerate(joint_ids)}
    body_joints = {}
    for i, b in enumerate(bodies):
        body_joints.setdefault(int(b), []).append(i)
    parent, depth = np.full(len(bodies), -1.0), np.zeros(len(bodies))
    for i, b in enumerate(bodies):
        same = body_joints[int(b)]
        k = same.index(i)
        if k > 0:
            parent[i] = same[k - 1]
        else:
            p = m.body_parentid[b]
            while p != trunk and p != 0:
                if int(p) in body_joints:
                    parent[i] = body_joints[int(p)][-1]
                    break
                p = m.body_parentid[p]
    for i in range(len(bodies)):  # depth = number of limb ancestors
        j = i
        while parent[j] >= 0:
            depth[i] += 1
            j = int(parent[j])
    ub = sorted(set(bodies.tolist()))
    caps = {b: lc.capsule_features(m, b) for b in ub}
    st = dict(
        bodies=bodies,
        body_names=[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in bodies],
        body_pos=m.body_pos[bodies].copy(), body_ipos=m.body_ipos[bodies].copy(), body_iquat=m.body_iquat[bodies].copy(),
        default_body_mass=m.body_mass[bodies].copy(), default_body_inertia=m.body_inertia[bodies].copy(),
        capsule_quat=np.stack([caps[b][0] for b in bodies]), capsule_shape=np.stack([caps[b][1] for b in bodies]),
        capsule_pos=np.stack([caps[b][2] for b in bodies]), has_geom=np.array([caps[b][3] for b in bodies]),
        jnt_pos=m.jnt_pos[joint_ids].copy(), jnt_axis=m.jnt_axis[joint_ids].copy(), parent=parent, depth=depth,
    )
    return st


def child_joint_counts(m, joint_ids):
    """URMA joint_nr_direct_child_joints with chain semantics: a hinge followed by another hinge on the same body
    has that one as its only direct child; the last hinge of a body has the first hinge of each child body."""
    jb = [int(m.jnt_bodyid[j]) for j in joint_ids]
    out = []
    for i, b in enumerate(jb):
        same = [k for k, bb in enumerate(jb) if bb == b]
        if same.index(i) < len(same) - 1:
            out.append(1.0)
            continue
        kids = [c for c in range(m.nbody) if m.body_parentid[c] == b and c != b]
        out.append(float(sum(1 for c in kids if c in jb)))
    return out


# ----------------------------------------------------------------------------- params from the a1 template

def unimal_params(tpl, m, p, dg, scale, trunk_mass, foot_radius, nominal_trunk_z, gears, nominal):
    P = copy.deepcopy(tpl["params"])
    C = P["control"]
    C.update(p_gain=p, d_gain=dg, scaling_factor=scale, seen_p_gain=p, seen_d_gain=dg, seen_scaling_factor=scale)
    S = P["seen"]
    # a1 relative ranges: trunk mass +-0.8 / 4.713 kg (17 %), p +-3/20 (15 %), d +-0.1/0.5 (20 %), scale +-0.03/0.25 (12 %),
    # foot size +-2.5 % of the foot radius
    S.update(add_trunk_mass_min=-0.17 * trunk_mass, add_trunk_mass_max=0.17 * trunk_mass, default_trunk_mass=trunk_mass,
             foot_size_min=0.975 * foot_radius, foot_size_max=1.025 * foot_radius,
             add_p_gain_min=-0.15 * p, add_p_gain_max=0.15 * p, add_d_gain_min=-0.2 * dg, add_d_gain_max=0.2 * dg,
             add_scaling_factor_min=-0.12 * scale, add_scaling_factor_max=0.12 * scale)
    # joint dynamics DR re-centred on the Unimal defaults (damping 1, armature 1, stiffness 1, frictionloss 0);
    # ModuMorph's own dynamics variants span armature 1-1.5 and damping 0.01-1
    S.update(joint_damping_min=0.5, joint_damping_max=1.5, joint_armature_min=0.75, joint_armature_max=1.25,
             joint_stiffness_min=0.5, joint_stiffness_max=1.5, joint_friction_loss_min=0.0, joint_friction_loss_max=0.65)
    # torque_limit_factor (0.5), velocity factor, range / nominal offsets, unseen factors, model DR, delay, push,
    # init, command, noise, dropout, sampling probs: a1 values unchanged
    R = P["reward"]
    R["nominal_trunk_z"] = nominal_trunk_z
    for k in [k for k in R if k.startswith("time_since_last_touchdown_")]:
        del R[k]
    P["termination_type"] = "HeightTermination"
    P["termination"] = {}
    return P


# ----------------------------------------------------------------------------- main conversion

def convert(src, out_dir, p_gain=None, d_gain=None):
    tpl = json.loads(TEMPLATE_CFG.read_text())
    name = robot_name(src)
    xml_text, gears = convert_xml(src)
    xml_path = out_dir / "xml" / f"{name}.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_text)
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    assert m.geom(0).name == "floor" and m.body(1).name == "trunk" and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    assert all(m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE)
    n = m.njnt - 1
    joint_ids = list(range(1, m.njnt))
    joint_names = [m.joint(j).name for j in joint_ids]
    assert m.nu == n and sorted(m.actuator_trnid[:, 0].tolist()) == joint_ids, "every hinge must have exactly one motor"
    # actuator order may differ from joint order: URMA's env writes ctrl in joint order -> reorder motors in the XML
    act_j = m.actuator_trnid[:, 0].tolist()
    if act_j != joint_ids:
        root = ET.fromstring(xml_text)
        act = root.find("actuator")
        mots = {mm.get("joint"): mm for mm in act.findall("motor")}
        for mm in list(act):
            act.remove(mm)
        for jn in joint_names:
            act.append(mots[jn])
        xml_text = ET.tostring(root, encoding="unicode")
        xml_path.write_text(xml_text)
        m = mujoco.MjModel.from_xml_path(str(xml_path))
        assert m.actuator_trnid[:, 0].tolist() == joint_ids
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    w = hi - lo
    nominal = np.clip(0.0, lo + 0.05 * w, hi - 0.05 * w)
    # feet
    d = pose(m, nominal, 1.0)
    low = lowest_points(m, d)
    zmin = low[1:].min()
    kids = {b: [c for c in range(1, m.nbody) if m.body_parentid[c] == b] for b in range(m.nbody)}
    leaves = [b for b in range(2, m.nbody) if not kids[b]]
    leaf_geom = {}
    for b in leaves:
        gs = [g for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b])]
        assert len(gs) == 1, (name, b, gs)
        leaf_geom[b] = gs[0]
    drop = 1.0 - zmin + 0.01
    p, dg, scale = choose_gains(m, nominal)
    if p_gain is not None:
        p = p_gain
    if d_gain is not None:
        dg = d_gain
    zs, tilts, ds = settle(m, nominal, p, dg, drop)
    # feet: leaves touching the floor after settling under PD at nominal; fallback: leaves near the lowest point
    touching = {int(m.geom_bodyid[c.geom2 if c.geom1 == 0 else c.geom1]) for c in ds.contact[:ds.ncon] if 0 in (c.geom1, c.geom2)}
    feet_rule = "settled_contact"
    feet_b = [b for b in leaves if b in touching]
    if len(feet_b) < 2:
        feet_rule = "lowest_point"
        feet_b = [b for b in leaves if low[leaf_geom[b]] - zmin < FOOT_TOL]
    if len(feet_b) < 2:
        feet_rule = "all_leaves"
        feet_b = leaves
    foot_geoms = [leaf_geom[b] for b in feet_b]
    foot_names = [m.geom(g).name for g in foot_geoms]  # "limb/N"
    radii = m.geom_size[foot_geoms, 0]
    assert np.ptp(radii) < 1e-9, (name, radii)  # URMA samples one foot size for all feet
    nominal_trunk_z = float(zs[-25:].mean())
    trunk_mass = float(m.body_mass[1])
    P = unimal_params(tpl, m, p, dg, scale, trunk_mass, float(radii[0]), nominal_trunk_z, gears, nominal)
    # symmetry pairs: feet mirrored in y at the nominal pose (x equal, y opposite)
    fp = np.array([d.geom_xpos[g] for g in foot_geoms]) - d.xpos[1]
    sym = []
    used = set()
    for i in range(len(fp)):
        for j in range(i + 1, len(fp)):
            if i in used or j in used:
                continue
            if abs(fp[i, 0] - fp[j, 0]) < 0.01 and abs(fp[i, 1] + fp[j, 1]) < 0.01 and abs(fp[i, 1]) > 0.01:
                sym.append([i, j]); used |= {i, j}
    nf = len(foot_names)
    groups = {"floor": [0], "feet": sorted(foot_geoms), "trunk": [m.body_geomadr[1] + k for k in range(m.body_geomnum[1])]}
    for fn, g in zip(foot_names, foot_geoms):
        groups[fn] = [g]
    struct = {
        "reward_collision_geoms": [], "reward_trunk_groups": ["trunk"],
        "timer_attr_to_foot": {f"time_since_last_touchdown_f{i}": i for i in range(nf)},
        "air_pairs": [[i, i] for i in range(nf)], "symmetry_pairs": sym, "timer_feet": list(range(nf)),
        "nominal_diff_joints": [],
        "termination": {"kind": "height", "threshold": 0.5 * nominal_trunk_z},
        "seen_dr": {"foot_size_geoms": foot_geoms, "foot_size_noise_power": 1, "damping_mode": "uniform",
                    "frictionloss_mode": "uniform", "keep_default": {}, "armature_stay_typo": True},
        "model_dr": {"friction_mode": "ground_feet", "friction_feet_geoms": foot_geoms},
        "unseen_dr": {"has_foot_noise": True}, "foot_obs_mode": "live", "obs_timer_feet": list(range(nf)), "masked": False,
    }
    vel = np.full(n, MAX_JOINT_VEL)
    torque = m.actuator_ctrlrange[:, 1].copy()
    # ---- context via limb_context on a stand-in env at the URMA default pose (trunk at drop height, nominal joints)
    st = static_limb_data(m, joint_ids)
    seen = SimpleNamespace(seen_body_mass=m.body_mass.copy(), seen_joint_range=m.jnt_range[1:].copy(),
                           seen_joint_nominal_position=nominal.copy(), seen_torque_limit=torque, seen_joint_velocity=vel,
                           seen_joint_damping=m.dof_damping[6:].copy(), seen_joint_armature=m.dof_armature[6:].copy(),
                           seen_joint_stiffness=m.jnt_stiffness[1:].copy(), seen_joint_frictionloss=m.dof_frictionloss[6:].copy())
    ctrl = SimpleNamespace(seen_p_gain=p, seen_d_gain=dg, seen_scaling_factor=scale)
    data = mujoco.MjData(m)
    env = SimpleNamespace(model=m, data=data, joint_names=joint_names, foot_names=foot_names,
                          joint_nr_direct_child_joints=child_joint_counts(m, joint_ids), control_function=ctrl,
                          domain_randomization_seen_robot_function=seen, _limb_context_static=st,
                          context_arm="urma", latent_dir="")
    data.qpos[:] = m.qpos0; data.qpos[0:3] = [0, 0, drop]; data.qpos[3:7] = [1, 0, 0, 0]; data.qpos[7:] = nominal
    mujoco.mj_forward(m, data)
    idx = np.arange(n)
    g, feet, joints_geo = lc._robot_geometry(env, foot_names)
    fields, _ = lc.limb_fields(env, idx, idx, joints_geo)
    cpu_desc = {}
    foot_g = np.concatenate([g["p_d_gain_scaling"], g["robot_mass"], g["robot_dims"]])
    for arm in ("urma",) + lc.NON_LATENT_ARMS + tuple(lc.latent_arms_available(LATENTS_DUMMY, name)):
        lat = None
        if lc.is_latent_arm(arm):
            raw = json.loads(lc.latent_path(LATENTS_DUMMY, arm, name).read_text())
            lat = np.stack([np.asarray(raw[b]) for b in st["body_names"]])
        jm, _ = lc.assemble_joint_matrix(arm, fields, g, lat)
        on_pos, on_g = lc.foot_blocks_on(arm)
        fm = np.stack([np.concatenate([feet[f] if on_pos else np.zeros(3), foot_g if on_g else np.zeros(7)]) for f in foot_names])
        cpu_desc[arm] = {"joints": jm.tolist(), "feet": fm.tolist()}
    tl = lambda x: np.asarray(x, dtype=np.float64).tolist()
    cfg = {
        "robot": name, "xml": str(xml_path.parent), "xml_file": xml_path.name, "xml_path": str(xml_path),
        "joint_names": joint_names, "foot_names": foot_names,
        "joint_nr_direct_child_joints": env.joint_nr_direct_child_joints,
        "nominal_joint_positions": tl(nominal), "max_joint_velocities": tl(vel),
        "initial_drop_height": float(drop), "dt": 0.02, "nr_substeps": 4, "horizon": 1000, "timestep": float(m.opt.timestep),
        "qpos_joint_idx": tl(np.arange(7, 7 + n)), "qvel_joint_idx": tl(np.arange(6, 6 + n)), "jnt_idx": tl(np.arange(1, 1 + n)),
        "complete_nominal_joint_position": [], "nq": int(m.nq), "nv": int(m.nv), "nu": int(m.nu),
        "collision_groups": {k: [int(x) for x in v] for k, v in groups.items()},
        "params": P,
        "defaults": {
            "body_mass": tl(m.body_mass), "body_inertia": tl(m.body_inertia), "body_ipos": tl(m.body_ipos),
            "actuator_ctrlrange": tl(m.actuator_ctrlrange), "jnt_range": tl(m.jnt_range),
            "dof_damping": tl(m.dof_damping), "dof_armature": tl(m.dof_armature),
            "jnt_stiffness": tl(m.jnt_stiffness), "dof_frictionloss": tl(m.dof_frictionloss),
            "geom_friction": tl(m.geom_friction), "geom_size": tl(m.geom_size),
            "geom_solref": tl(m.geom_solref), "gravity": tl(m.opt.gravity),
        },
        "context": {
            "fields": {k: tl(v) for k, v in fields.items()}, "g": {k: tl(v) for k, v in g.items()},
            "feet": {k: tl(v) for k, v in feet.items()}, "body_names": st["body_names"],
            "default_body_mass_limbs": tl(st["default_body_mass"]), "default_body_inertia_limbs": tl(st["default_body_inertia"]),
            "limb_bodies": tl(st["bodies"]), "robot_dimensions": tl(env.robot_dimensions),
            "dof_idx": tl(idx), "jnt_idx": tl(idx),
        },
        "cpu_descriptions": cpu_desc, "foot_geom_suffix": "", "eq": {"neq": 0, "types": [], "active": []},
        "struct": struct,
        "unimal": {
            "source_xml": str(src), "gears": [gears[j] for j in joint_names], "leaves": [m.body(b).name for b in leaves],
            "feet_bodies": [m.body(b).name for b in feet_b], "feet_rule": feet_rule,
            "nonleaf_floor_contacts": [m.body(b).name for b in sorted(touching) if b not in leaves],
            "self_contacts_settled": int(sum(1 for c in ds.contact[:ds.ncon] if 0 not in (c.geom1, c.geom2))), "n_bodies_with_2_hinges": int(sum(np.bincount(st["bodies"]) == 2)),
            "settle": {"z_last": float(zs[-1]), "z_std_last1s": float(zs[-50:].std()), "z_min": float(zs.min()),
                       "tilt_deg_last": float(tilts[-1]), "drop": float(drop)},
            "mass": float(m.body_mass.sum()), "p_gain": p, "d_gain": dg, "scale": scale,
        },
    }
    out = out_dir / "configs" / f"{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cfg))
    return cfg


def main():
    out_dir = Path(sys.argv[1])
    for src in sys.argv[2:]:
        c = convert(src, out_dir)
        u = c["unimal"]
        print(c["robot"], "n", len(c["joint_names"]), "feet", len(c["foot_names"]), "mass %.1f" % u["mass"],
              "drop %.3f zsettled %.3f std %.4f tilt %.1f" % (u["settle"]["drop"], c["params"]["reward"]["nominal_trunk_z"],
                                                             u["settle"]["z_std_last1s"], u["settle"]["tilt_deg_last"]),
              "sym", c["struct"]["symmetry_pairs"], "2hinge", u["n_bodies_with_2_hinges"], u["feet_rule"], "selfcon", u["self_contacts_settled"], flush=True)


if __name__ == "__main__":
    main()
