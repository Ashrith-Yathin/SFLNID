import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import sys, os, json
sys.path.insert(0, _os.path.join(ROOT, 'src'))
from multiprocessing import Pool
import sflnid_r as S
BASE = dict(epochs=2, lr=0.05)
SEEDS = [0, 1, 2]
OUT = _os.path.join(ROOT, 'results'); os.makedirs(OUT, exist_ok=True)
jobs = {}
def add(name, **kw):
    for s in SEEDS: jobs[f'{name}__s{s}'] = (s, kw)
# E1/E2 : aggregation under attacks
for att in ['backdoor', 'signflip', 'labelflip', 'none']:
    for agg in ['fedavg', 'hcta', 'median', 'trim', 'krum', 'normclip']:
        add(f'agg_{att}_{agg}', agg=agg, attack=att)
# E3 : evasion
add('adv_cafat0.5', cafat=True, cafat_eps=0.5)
add('adv_naive0.5', cafat=True, cafat_eps=0.5, at_naive=True)
add('adv_cafat0.25', cafat=True, cafat_eps=0.25)
add('adv_cafat1.0', cafat=True, cafat_eps=1.0)
# E4 : drift (no attack)
for agg in ['fedavg', 'hcta']:
    for g in [False, True]:
        add(f'drift_{agg}_gate{int(g)}', agg=agg, drift_round=10, rounds=20, drift_gate=g, dp_loss=g)
add('ctrl_nodrift_gate', agg='fedavg', rounds=20, drift_gate=True, dp_loss=True)   # false-trigger control
add('ctrl_nodrift_nogate', agg='fedavg', rounds=20)
# E5 : combined (backdoor + drift) ablation
D = dict(attack='backdoor', drift_round=10, rounds=20)
add('comb_base', agg='fedavg', **D)
add('comb_A', agg='hcta', **D)
add('comb_B', agg='fedavg', cafat=True, **D)
add('comb_C', agg='fedavg', drift_gate=True, dp_loss=True, **D)
add('comb_AC', agg='hcta', drift_gate=True, dp_loss=True, **D)
add('comb_ABC', agg='hcta', cafat=True, drift_gate=True, dp_loss=True, **D)
add('comb_AB', agg='hcta', cafat=True, **D)

def go(name):
    f = f'{OUT}/{name}.json'
    if os.path.exists(f): return name, 'cached'
    s, kw = jobs[name]
    o = S.run_fl(s, **BASE, **kw)
    json.dump(o, open(f, 'w'))
    return name, {k: round(v, 3) for k, v in o['final'].items() if k in ('acc', 'f1', 'asr')}

if __name__ == '__main__':
    def prio(n):
        if n.startswith('agg_labelflip') or n.startswith('adv_cafat0.25') or n.startswith('adv_cafat1.0') or n.startswith('agg_none_trim') or n.startswith('agg_none_normclip') or n.startswith('agg_signflip_trim'): return 9
        if n.startswith('agg_backdoor'): return 0
        if n.startswith('adv_cafat0.5') or n.startswith('adv_naive'): return 1
        if n.startswith('comb_'): return 2
        if n.startswith('drift_') or n.startswith('ctrl_'): return 3
        if n.startswith('agg_signflip'): return 4
        if n.startswith('agg_none_'): return 5
        if n.startswith('agg_labelflip_') and n.split('_')[2] in ('fedavg', 'hcta', 'median'): return 6
        if n.startswith('adv_'): return 8
        return 9
    names = sorted([n for n in jobs if prio(n) < 9], key=prio); print(len(names), 'jobs', flush=True)
    with Pool(2) as p:
        for r in p.imap_unordered(go, names): print(r, flush=True)
    print('ALL DONE', flush=True)
