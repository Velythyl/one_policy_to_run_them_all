# Is the converted MJCF slower than the original Unimal physics? Native torque sinusoids (0.8 x gear), 40 gaits,
# best planar speed: (orig) the Unimal XML + a ModuMorph-like floor box only (MuJoCo defaults: solref 0.02 1, impratio 1)
# vs (conv) the converted XML; plus conv with impratio 1, conv with MuJoCo-default solref.
import sys, json, re, numpy as np, mujoco
D = '/network/scratch/c/charlie.gauthier/urma_context/unimal_probe/gen/configs/'
def run(m, z0, n=40, seed=0):
    rng = np.random.default_rng(seed); best = 0; gear = np.abs(m.actuator_gear[:, 0]) * np.abs(m.actuator_ctrlrange[:, 1])
    for trial in range(n):
        f = rng.choice([1.0, 1.5, 2.0, 3.0]); ph = rng.uniform(0, 2 * np.pi, m.nu)
        d = mujoco.MjData(m); d.qpos[0:3] = [0, 0, z0]; d.qpos[3:7] = [1, 0, 0, 0]; mujoco.mj_forward(m, d)
        for k in range(200):
            u = 0.8 * np.sin(2 * np.pi * f * k * 0.02 + ph)
            d.ctrl[:] = u * (m.actuator_ctrlrange[:, 1])  # orig: ctrl in [-1,1] (gear inside); conv: ctrl = torque
            for _ in range(4): mujoco.mj_step(m, d)
            if k == 99: x0 = d.qpos[0:2].copy()
        best = max(best, np.linalg.norm(d.qpos[0:2] - x0) / 2.0)
    return best
for r in sys.argv[1:]:
    c = json.load(open(D + r + '.json')); src = c['unimal']['source_xml']; z0 = c['initial_drop_height']
    txt = open(src).read().replace('<worldbody>', '<worldbody><geom name="floor/0" type="box" size="25 20 1" pos="0 0 -1" material="grid"/>', 1)
    txt = re.sub(r'<sensor>.*?</sensor>', '', txt, flags=re.S)
    mo = mujoco.MjModel.from_xml_string(txt)
    mc = mujoco.MjModel.from_xml_path(c['xml_path'])
    mi = mujoco.MjModel.from_xml_path(c['xml_path']); mi.opt.impratio = 1
    ms = mujoco.MjModel.from_xml_path(c['xml_path']); ms.geom_solref[:] = [0.02, 1]
    print(r, 'orig %.3f conv %.3f conv+impratio1 %.3f conv+solref_default %.3f' % (run(mo, z0), run(mc, z0), run(mi, z0), run(ms, z0)), flush=True)
if __name__ == "__main__" and False:
    pass
