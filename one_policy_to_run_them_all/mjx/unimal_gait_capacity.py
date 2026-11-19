# Same open-loop random sinusoid search, three controllers, 40 gaits each, best planar speed (m/s) over seconds 2-4:
#  PD-URMA: PD p,d of the config around nominal, amplitude 2 x action scale (2 sigma of the initial policy)
#  PD-full: PD around the joint-range middle, amplitude = 0.9 x half range (whole joint range)
#  torque : ModuMorph's native control: torque = gear x 0.8 sin(...) (ctrl in [-1,1] x gear), no PD
# For reference the same for unitree_a1 (URMA XML, its PD gains) with PD-URMA.
import sys, json, numpy as np, mujoco
rng = np.random.default_rng(0)
def run(m, ctrl_fn, z0, nom, n=40):
    best = 0
    for trial in range(n):
        f = rng.choice([1.0, 1.5, 2.0, 3.0]); ph = rng.uniform(0, 2 * np.pi, m.nu)
        d = mujoco.MjData(m); d.qpos[0:3] = [0, 0, z0]; d.qpos[3:7] = [1, 0, 0, 0]; d.qpos[7:] = nom; mujoco.mj_forward(m, d)
        for k in range(200):
            s = np.sin(2 * np.pi * f * k * 0.02 + ph)
            d.ctrl[:] = ctrl_fn(d, s)
            for _ in range(4): mujoco.mj_step(m, d)
            if k == 99: x0 = d.qpos[0:2].copy()
        best = max(best, np.linalg.norm(d.qpos[0:2] - x0) / 2.0)
    return best
D = '/network/scratch/c/charlie.gauthier/urma_context/unimal_probe/gen/configs/'
for r in sys.argv[1:]:
    if r == 'unitree_a1':
        c = json.load(open('/network/scratch/c/charlie.gauthier/urma_context/mjx/configs_v2/unitree_a1.json'))
        m = mujoco.MjModel.from_xml_path(json.load(open('/network/scratch/c/charlie.gauthier/urma_context/robot_models_final.json'))['unitree_a1']['xml'])
    else:
        c = json.load(open(D + r + '.json')); m = mujoco.MjModel.from_xml_path(c['xml_path'])
    nom = np.array(c['nominal_joint_positions']); C = c['params']['control']; p, dg, sc = C['p_gain'], C['d_gain'], C['scaling_factor']
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]; mid, half = (lo + hi) / 2, (hi - lo) / 2
    gear = m.actuator_ctrlrange[:, 1]
    z0 = c['initial_drop_height']
    a = run(m, lambda d, s: p * (nom + 2 * sc * s - d.qpos[7:]) - dg * d.qvel[6:], z0, nom)
    if r == 'unitree_a1':
        print(r, 'PD-URMA %.3f' % a, flush=True); continue
    b = run(m, lambda d, s: p * (mid + 0.9 * half * s - d.qpos[7:]) - dg * d.qvel[6:], z0, nom)
    t = run(m, lambda d, s: 0.8 * gear * s, z0, nom)
    print(r, 'PD-URMA %.3f PD-full %.3f torque %.3f' % (a, b, t), flush=True)
