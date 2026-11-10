import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import sflnid_r as S

BASE = dict(epochs=2, lr=0.05)
JOBS = {
 'clean_fedavg': dict(agg='fedavg'),
 'clean_hcta': dict(agg='hcta'),
 'bd_fedavg': dict(agg='fedavg', attack='backdoor'),
 'bd_hcta': dict(agg='hcta', attack='backdoor'),
 'sf_fedavg': dict(agg='fedavg', attack='signflip'),
 'sf_hcta': dict(agg='hcta', attack='signflip'),
}
def go(k):
    o = S.run_fl(100, **BASE, **JOBS[k])
    f = o['final']
    flags = None
    if 'hcta_log' in o:
        import numpy as np
        fl = np.array([l['flag'] for l in o['hcta_log']]); flags = fl.sum(0).tolist()
    return k, {m: round(f[m], 3) for m in ('acc', 'f1', 'rec', 'fpr', 'asr')}, o['mal'], flags

if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(go, list(JOBS)): print(r, flush=True)
