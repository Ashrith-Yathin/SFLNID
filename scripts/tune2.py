import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, json
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import numpy as np, sflnid_r as S
BASE = dict(epochs=2, lr=0.05, attack='backdoor', agg='hcta')
V = {'zero': dict(), 'partial0.15': dict(flag_mode='partial', flag_w=0.15), 'cwclip0.25': dict(flag_mode='cwclip', flag_w=0.25),
     'cwclip0.5': dict(flag_mode='cwclip', flag_w=0.5), 'cwclip1.0': dict(flag_mode='cwclip', flag_w=1.0), 'partial0.4': dict(flag_mode='partial', flag_w=0.4)}
def go(k):
    o = S.run_fl(100, hcta_kw=V[k], **BASE); f = o['final']
    return k, {m: round(f[m], 3) for m in ('acc', 'f1', 'rec', 'asr')}
if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(go, list(V)): print(r, flush=True)
