import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import sflnid_r as S
J = [(100, 10), (101, 10), (100, None), (101, None)]
def go(a):
    s, d = a; o = S.run_fl(s, epochs=2, lr=0.05, rounds=20, drift_round=d, dp_loss=False)
    return a, [h['probe'] for h in o['hist']], [h['f1'] for h in o['hist']]
if __name__ == '__main__':
    with Pool(2) as p: res = p.map(go, J)
    json.dump(res, open(_os.path.join(ROOT, 'results', 'probe_traj.json'), 'w'))
    for r in res: print(r[0], [round(x, 3) for x in r[1]])
