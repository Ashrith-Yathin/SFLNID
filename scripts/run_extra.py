import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import sflnid_r as S
BASE = dict(epochs=2, lr=0.05); SEEDS = [0, 1, 2]
OUT = _os.path.join(ROOT, 'results')
jobs = {}
def add(name, **kw):
    for s in SEEDS: jobs[f'{name}__s{s}'] = (s, kw)
add('x_cafat_dpmatched', cafat=True, cafat_eps=0.5, eps_dp=2.0)            # noise x2 -> same (eps,delta) under doubled per-record influence
def go(name):
    f = f'{OUT}/{name}.json'
    if os.path.exists(f): return name, 'cached'
    s, kw = jobs[name]; o = S.run_fl(s, **BASE, **kw); json.dump(o, open(f, 'w'))
    return name, {k: round(v, 3) for k, v in o['final'].items() if k in ('acc', 'f1', 'asr')}
if __name__ == '__main__':
    with Pool(2) as p:
        for r in p.imap_unordered(go, list(jobs)): print(r, flush=True)
    print('EXTRA DONE', flush=True)
