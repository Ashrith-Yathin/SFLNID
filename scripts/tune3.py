import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import sflnid_r as S
BASE = dict(epochs=2, lr=0.05, drift_round=10, rounds=20)
J = {'drift_fedavg_gate': dict(agg='fedavg', drift_gate=True, dp_loss=True), 'drift_fedavg_nogate': dict(agg='fedavg'),
     'drift_hcta_gate': dict(agg='hcta', drift_gate=True, dp_loss=True), 'drift_hcta_nogate': dict(agg='hcta')}
def go(k):
    o = S.run_fl(100, **BASE, **J[k])
    return k, o['triggers'], [round(h['probe'], 3) for h in o['hist']][6:16], [round(h['f1'], 2) for h in o['hist']][8:16], round(o['final']['f1'], 3)
if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(go, list(J)): print(r, flush=True)
