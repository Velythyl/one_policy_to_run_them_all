import json, sys, math
L=[json.loads(l) for l in open(sys.argv[1])]
rs=sorted({k.split('/')[0] for k in L[-1] if '/step_reward' in k})
def avg(key, a, b):
    v=[r[key] for r in L[a:b] if key in r and not math.isnan(r[key])]
    return sum(v)/len(v) if v else float('nan')
n=len(L); W=max(1,n//8)
print('iters', n, 'steps/robot', L[-1]['steps_per_robot'], 'sps', L[-1]['sps'])
for x in rs:
    print(x)
    for a in range(0, n, W):
        print(f"  it {a+1:4d}-{min(a+W,n):4d}: step_rew {avg(x+'/step_reward',a,a+W):.4f} trk_xy {avg(x+'/term_track_xy_vel_cmd',a,a+W):.4f} eps_len {avg(x+'/ep_length',a,a+W):6.0f} eps_ret {avg(x+'/ep_return',a,a+W):.3f} eps_track {avg(x+'/eps_track_perf',a,a+W):.3f} unst {avg(x+'/sim_unstable',a,a+W):.1f} height {avg(x+'/term_base_height',a,a+W):.3f} coll {avg(x+'/term_collision',a,a+W):.3f}")
