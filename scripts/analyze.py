import os as _os
ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import json, glob, os, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
R = _os.path.join(ROOT, 'results'); FIG = _os.path.join(ROOT, 'paper', 'figs'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.titlesize': 13, 'legend.fontsize': 11, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'axes.grid': True, 'grid.alpha': .3, 'figure.dpi': 150})

def load(prefix):
    return [json.load(open(f)) for f in sorted(glob.glob(f'{R}/{prefix}__s*.json'))]
def ms(runs, key, sub='final'):
    v = np.array([r[sub][key] for r in runs]); return v.mean(), v.std(), len(v)
def fmtb(runs, key, d=3):
    if not runs: return '--'
    m, s, n = ms(runs, key)
    return f'$\\mathbf{{{m:.{d}f}\\pm{s:.{d}f}}}$'
def fmt(runs, key, d=3, pct=False):
    if not runs: return '--'
    m, s, n = ms(runs, key);
    return f'{m:.{d}f}$\\pm${s:.{d}f}'
SUM = {}

AGGS = ['fedavg', 'median', 'trim', 'krum', 'normclip', 'hcta']
ANAME = dict(fedavg='FedAvg (SFLNID)', median='Coord.\\ Median', trim='Trimmed Mean', krum='Multi-Krum', normclip='Norm Clipping', hcta='\\textbf{HCTA (ours)}')
ATT = [('none', 'No attack'), ('backdoor', 'Backdoor + model replacement'), ('signflip', 'Scaled sign-flip')]

# ---- Table: aggregation
rows = []
for att, an in ATT:
    rows.append(f'\\multicolumn{{5}}{{l}}{{\\textit{{{an}}}}} \\\\')
    best_f1 = max([ms(load(f'agg_{att}_{a}'), 'f1')[0] for a in AGGS if load(f'agg_{att}_{a}')] or [0])
    for a in AGGS:
        rs = load(f'agg_{att}_{a}')
        if not rs: continue
        SUM[f'agg_{att}_{a}'] = {k: ms(rs, k)[:2] for k in ('acc', 'f1', 'rec', 'fpr', 'asr')}
        asr_c = fmt(rs, 'asr') if att in ('backdoor', 'none') else '--'
        if a == 'hcta':
            asr_b = fmtb(rs, 'asr') if att in ('backdoor', 'none') else '--'
            rows.append(f'{ANAME[a]} & {fmtb(rs,"acc")} & {fmtb(rs,"f1")} & {fmtb(rs,"rec")} & {asr_b} \\\\')
        else:
            rows.append(f'{ANAME[a]} & {fmt(rs,"acc")} & {fmt(rs,"f1")} & {fmt(rs,"rec")} & {asr_c} \\\\')
open(_os.path.join(ROOT, 'paper', 'tab_agg.tex'), 'w').write('\n'.join(rows))

# ---- Fig: bars across defenses (macro-F1 for 3 attacks + ASR for backdoor)
fig, ax = plt.subplots(1, 3, figsize=(11, 3.8))
for j, (att, an) in enumerate(ATT):
    f1m, f1s, names = [], [], []
    for a in AGGS:
        rs = load(f'agg_{att}_{a}')
        if rs: m, s, _ = ms(rs, 'f1'); f1m.append(m); f1s.append(s); names.append(a)
    cols = ['#d62728' if n == 'hcta' else '#9a9a9a' for n in names]
    ax[j].bar(range(len(names)), f1m, yerr=f1s, color=cols, capsize=3)
    for i, v in enumerate(f1m): ax[j].text(i, v + 0.04, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold' if names[i] == 'hcta' else 'normal')
    ax[j].set_xticks(range(len(names))); ax[j].set_xticklabels(['FedAvg' if n == 'fedavg' else 'Med' if n == 'median' else 'Trim' if n == 'trim' else 'Krum' if n == 'krum' else 'Clip' if n == 'normclip' else 'HCTA' for n in names], rotation=30)
    for t in ax[j].get_xticklabels():
        if t.get_text() == 'HCTA': t.set_fontweight('bold'); t.set_color('#d62728')
    ax[j].set_title(an.split(' +')[0]); ax[j].set_ylim(0, 1.05)
    if j == 0: ax[j].set_ylabel('Macro-F1')
plt.tight_layout(); plt.savefig(f'{FIG}/fig_agg.pdf'); plt.close()

# ---- Fig: trust trajectories (backdoor, seed 0)
rs = load('agg_backdoor_hcta')
if rs:
    r = rs[0]; z = np.array([l['z'] for l in r['hcta_log']]); mal = r['mal']
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.3))
    for k in range(z.shape[1]):
        ax.plot(range(1, len(z) + 1), np.clip(z[:, k], -3, 25), color='#d62728' if k in mal else '#1f77b4', alpha=.9 if k in mal else .5, lw=2.6 if k in mal else 1.4, label='malicious' if k == mal[0] else ('benign' if k == [i for i in range(z.shape[1]) if i not in mal][0] else None))
    ax.axvline(4, color='k', ls=':', lw=.8); ax.set_xlabel('Communication round'); ax.set_ylabel('Deviation score $z_i^t$'); ax.legend()
    plt.tight_layout(); plt.savefig(f'{FIG}/fig_trust.pdf'); plt.close()
    # detection stats over all seeds
    tp = fp = tot_m = tot_b = 0; first = []
    for r in load('agg_backdoor_hcta') + load('agg_signflip_hcta') + load('agg_none_hcta'):
        fl = np.array([l['flag'] for l in r['hcta_log']]); mal = r['mal']
        for k in range(fl.shape[1]):
            if k in mal:
                tp += fl[3:, k].sum(); tot_m += fl[3:, k].size
                if fl[:, k].any(): first.append(int(np.argmax(fl[:, k])) + 1 - 4)
            else:
                fp += fl[:, k].sum(); tot_b += fl[:, k].size
    SUM['hcta_detect'] = dict(tpr=tp / max(1, tot_m), fpr=fp / max(1, tot_b), delay=float(np.mean(first)) if first else None, n_first=len(first))

# ---- Evasion table / fig
EPS = ['ev0', 'ev0.25', 'ev0.5', 'ev1.0']
variants = [('agg_none_fedavg', 'SFLNID'), ('adv_naive0.5', 'Naive PGD-AT'), ('adv_cafat0.25', 'CAFAT $\\epsilon_{tr}$=0.25'),
            ('adv_cafat0.5', '\\textbf{CAFAT}'), ('x_cafat_dpmatched', 'CAFAT (DP-matched)')]
PLAIN = {'agg_none_fedavg': 'SFLNID', 'adv_naive0.5': 'Naive PGD-AT', 'adv_cafat0.25': 'CAFAT 0.25', 'adv_cafat0.5': 'CAFAT 0.5', 'adv_cafat1.0': 'CAFAT 1.0', 'x_cafat_dpmatched': 'CAFAT 0.5 (DP-matched)'}
rows = []; fig, ax = plt.subplots(figsize=(8, 4.8))
for key, name in variants:
    rs = load(key)
    if not rs: continue
    cells = [fmt(rs, e) for e in EPS] + [fmt(rs, 'f1')]
    if key.startswith('adv_cafat') or key.startswith('x_cafat'):
        cells = [fmtb(rs, e) for e in EPS] + [fmtb(rs, 'f1')]
        if '\\textbf' not in name: name = '\\textbf{' + name + '}'
    rows.append(f'{name} & ' + ' & '.join(cells) + ' \\\\')
    SUM[key] = {e: ms(rs, e)[:2] for e in EPS + ['f1', 'fpr', 'acc']}
    ax.errorbar([0, .25, .5, 1.], [ms(rs, e)[0] for e in EPS], yerr=[ms(rs, e)[1] for e in EPS], marker='o', ms=7, lw=3 if key.startswith(('adv_cafat', 'x_cafat')) else 1.8, capsize=4, label=PLAIN[key])
ax.set_xlabel('Evasion budget $\\epsilon$ (controllable features)'); ax.set_ylabel('Attack detection rate'); ax.legend(loc='lower left')
plt.tight_layout(); plt.savefig(f'{FIG}/fig_evasion.pdf'); plt.close()
open(_os.path.join(ROOT, 'paper', 'tab_adv.tex'), 'w').write('\n'.join(rows))

# ---- Drift
def rec_stats(runs, drift=10, thr=0.95):
    out = []
    for r in runs:
        f1 = np.array([h['f1'] for h in r['hist']]); pre = f1[drift - 2]  # round drift-1 (index drift-2)
        post = f1[drift - 1:]; mn = post.min(); rec = next((i for i, v in enumerate(post) if v >= thr * pre), len(post))
        out.append((pre, mn, rec, f1[-1], f1[drift - 1:].mean()))
    return np.array(out)
rows = []; fig, ax = plt.subplots(figsize=(8, 4.8))
for key, name, c in [('drift_fedavg_gate0', 'SFLNID', '#7f7f7f'), ('drift_fedavg_gate1', 'SFLNID + C', '#d62728'), ('drift_hcta_gate0', 'SFLNID + A', '#1f77b4'), ('drift_hcta_gate1', 'SFLNID + A + C', '#2ca02c')]:
    rs = load(key)
    if not rs: continue
    a = rec_stats(rs); m, s = a.mean(0), a.std(0)
    trig = [r['triggers'] for r in rs]
    bf = lambda i: f'$\\mathbf{{{m[i]:.3f}\\pm{s[i]:.3f}}}$' if name != 'SFLNID' else f'${m[i]:.3f}\\pm{s[i]:.3f}$'
    nm = name if name == 'SFLNID' else '\\textbf{' + name + '}'
    rows.append(f'{nm} & {bf(0)} & {bf(1)} & {bf(4)} & {bf(3)} \\\\')
    SUM[key] = dict(minf1=(m[1], s[1]), rec=(m[2], s[2]), mean_post=(m[4], s[4]), final=(m[3], s[3]), triggers=trig)
    F = np.array([[h['f1'] for h in r['hist']] for r in rs]); ax.plot(range(1, F.shape[1] + 1), F.mean(0), color=c, label=name, lw=2.6); ax.fill_between(range(1, F.shape[1] + 1), F.mean(0) - F.std(0), F.mean(0) + F.std(0), color=c, alpha=.15)
ax.axvline(10, color='k', ls=':', lw=.8); ax.set_xlabel('Communication round'); ax.set_ylabel('Macro-F1 (global test)'); ax.legend(loc='lower right')
plt.tight_layout(); plt.savefig(f'{FIG}/fig_drift.pdf'); plt.close()
open(_os.path.join(ROOT, 'paper', 'tab_drift.tex'), 'w').write('\n'.join(rows))

# ---- Combined ablation
rows = []
names = [('comb_base', 'SFLNID', '', '', ''), ('comb_A', '+A', '\\checkmark', '', ''), ('comb_B', '+B', '', '\\checkmark', ''), ('comb_C', '+C', '', '', '\\checkmark'),
         ('comb_AB', '+A+B', '\\checkmark', '\\checkmark', ''), ('comb_AC', '+A+C', '\\checkmark', '', '\\checkmark'), ('comb_ABC', '\\textbf{SFLNID-R}', '\\checkmark', '\\checkmark', '\\checkmark')]
for key, name, a, b, c in names:
    rs = load(key)
    if not rs: continue
    cells = [fmt(rs, 'f1'), fmt(rs, 'asr'), fmt(rs, 'ev0.5')]
    SUM[key] = {k: ms(rs, k)[:2] for k in ('acc', 'f1', 'asr', 'ev0.5', 'fpr', 'rec')}
    SUM[key]['t_cli'] = ms([dict(final=dict(t=r['t_cli'])) for r in rs], 't')[:2]; SUM[key]['t_srv'] = ms([dict(final=dict(t=r['t_srv'])) for r in rs], 't')[:2]
    if key != 'comb_base':
        cells = [fmtb(rs, 'f1'), fmtb(rs, 'asr'), fmtb(rs, 'ev0.5')]
        if '\\textbf' not in name: name = '\\textbf{' + name + '}'
    rows.append(f'{name} & {a} & {b} & {c} & ' + ' & '.join(cells) + ' \\\\')
open(_os.path.join(ROOT, 'paper', 'tab_abl.tex'), 'w').write('\n'.join(rows))
fig, ax = plt.subplots(1, 3, figsize=(12, 4.2))
ks = [k for k, *_ in names if k in SUM]
lab = [n.replace('\\textbf{', '').replace('}', '').replace('SFLNID-R (A+B+C)', 'Full') for k, n, *_ in names if k in SUM]
for j, (m, t, good) in enumerate([('f1', 'Macro-F1 ($\\uparrow$)', 1), ('asr', 'Backdoor ASR ($\\downarrow$)', 0), ('ev0.5', 'Detection @ $\\epsilon$=0.5 ($\\uparrow$)', 1)]):
    ax[j].bar(range(len(ks)), [SUM[k][m][0] for k in ks], yerr=[SUM[k][m][1] for k in ks], color=['#d62728' if k == 'comb_ABC' else '#1f77b4' if k != 'comb_base' else '#9a9a9a' for k in ks], capsize=3)
    for i, k in enumerate(ks): ax[j].text(i, SUM[k][m][0] + 0.03, f'{SUM[k][m][0]:.2f}', ha='center', fontsize=9)
    ax[j].set_xticks(range(len(ks))); ax[j].set_xticklabels(lab, rotation=40, fontsize=11); ax[j].set_title(t); ax[j].set_ylim(0, 1.1)
plt.tight_layout(); plt.savefig(f'{FIG}/fig_abl.pdf'); plt.close()

# ---- overhead
oh = {}
for key in ['agg_none_fedavg', 'agg_none_hcta', 'agg_none_krum', 'agg_none_median', 'adv_cafat0.5', 'adv_naive0.5']:
    rs = load(key)
    if rs: oh[key] = dict(cli=float(np.mean([r['t_cli'] for r in rs])), srv=float(np.mean([r['t_srv'] for r in rs])), n=rs[0]['nparams'])
SUM['overhead'] = oh
json.dump(SUM, open(_os.path.join(ROOT, 'results', 'summary.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in SUM.items() if k in ('hcta_detect', 'overhead')}, indent=1, default=float))
print('n result files', len(glob.glob(f'{R}/*.json')))
