"""
SFLNID-R: robustness extension of SFLNID (federated, DP, CNN-GRU NIDS for SDN-enabled IIoT).

Modules
  baseline  : Focal loss + sorted-1D Wasserstein regulariser + Holt adaptive-clip DP + CNN-GRU (re-implementation)
  A  (HCTA) : Holt-CUSUM Trust Aggregation          (poisoning / backdoor robustness under DP)
  B  (CAFAT): Constraint-aware Focal Adversarial Training (evasion robustness, controllable features only)
  C  (DRIFT): CUSUM drift gate (resets Holt clip state, boosts lr, suspends quarantine)

Data: seeded synthetic IIoT-flow generator that reproduces the paper's protocol (10 classes, 19 features,
heavy imbalance, class-disjoint non-IID split over K=7 clients).  `load_real_csv` is a hook for ToN-IoT etc.
"""
import math, time, json
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

torch.set_num_threads(1)
D = 19
CLASSES = ['Normal', 'Backdoor', 'DDoS', 'DoS', 'Injection', 'Mitm', 'Password', 'Ransomware', 'Scanning', 'XSS']
NC = len(CLASSES)
CTRL = list(range(6))            # features an attacker can manipulate (packet size, IAT, payload, flags, ...)
TRIG = [14, 15, 16]              # backdoor trigger features
TRIG_VAL = 4.0
PRIORS = np.array([.55, .04, .10, .06, .06, .015, .06, .035, .05, .03]); PRIORS /= PRIORS.sum()
SEP = [0, 2.3, 3.1, 2.8, 1.5, 1.2, 2.2, 2.4, 2.8, 1.6]
CLIENT_CLASSES = [[1, 2, 5], [3, 4, 6], [7, 8, 9], [2, 6, 8], [1, 3, 9], [4, 7, 5], [2, 8, 4]]
DRIFT_CLASSES = [2, 3, 8]


# ----------------------------------------------------------------------------- data
class Gen:
    def __init__(self, struct_seed=7):
        r = np.random.default_rng(struct_seed)
        self.comp = r.normal(0, 0.9, (3, D))
        self.mu = np.zeros((NC, D))
        for c in range(1, NC):
            idx = r.choice(D, 7, replace=False)
            self.mu[c, idx] = r.normal(0, 1, 7) * SEP[c]
        self.M = np.stack([np.eye(D) + 0.35 * r.normal(0, 1, (D, D)) / math.sqrt(D) for _ in range(NC)])
        self.s = r.uniform(0.8, 1.25, NC)
        v = np.zeros(D); idx = r.choice(D, 8, replace=False); v[idx] = r.normal(0, 1, 8)
        self.drift = 1.6 * v / np.linalg.norm(v) * 2.2

    def sample(self, c, n, rng, drift=False):
        z = rng.standard_t(5, (n, D)) / math.sqrt(5 / 3)
        x = z @ self.M[c] * self.s[c]
        if c == 0:
            x = x + self.comp[rng.integers(0, 3, n)]
        else:
            x = x + self.mu[c]
            if drift and c in DRIFT_CLASSES:
                x = x + self.drift
        return x.astype(np.float32)


def make_fl_data(seed, n_total=42000, K=7):
    rng = np.random.default_rng(1000 + seed)
    G = Gen()
    counts = np.maximum((PRIORS * n_total).astype(int), 400)
    tr_idx = {}; clients = [dict(X=[], y=[]) for _ in range(K)]; clients_d = [dict(X=[], y=[]) for _ in range(K)]
    for c in range(NC):
        n_tr = int(counts[c] * 0.8)
        owners = [k for k in range(K) if c == 0 or c in CLIENT_CLASSES[k]]
        w = rng.dirichlet(np.ones(len(owners)) * 2.0) if c == 0 else rng.dirichlet(np.ones(len(owners)) * 4.0)
        ns = np.maximum((w * n_tr).astype(int), 40)
        for k, nk in zip(owners, ns):
            clients[k]['X'].append(G.sample(c, nk, rng)); clients[k]['y'].append(np.full(nk, c))
            clients_d[k]['X'].append(G.sample(c, nk, rng, drift=True)); clients_d[k]['y'].append(np.full(nk, c))
    out = []
    for cl in (clients, clients_d):
        o = []
        for k in range(K):
            X = np.concatenate(cl[k]['X']); y = np.concatenate(cl[k]['y'])
            o.append((torch.tensor(X), torch.tensor(y, dtype=torch.long)))
        out.append(o)
    # global test (natural imbalance); drifted version
    Xt, yt, Xd = [], [], []
    for c in range(NC):
        n = max(int(counts[c] * 0.25), 150)
        Xt.append(G.sample(c, n, rng)); Xd.append(G.sample(c, n, rng, drift=True)); yt.append(np.full(n, c))
    yt = torch.tensor(np.concatenate(yt), dtype=torch.long)
    return dict(clients=out[0], clients_drift=out[1], Xte=torch.tensor(np.concatenate(Xt)), Xte_drift=torch.tensor(np.concatenate(Xd)), yte=yt)


def load_real_csv(path, label_col, feat_cols, K=7):
    """Hook for ToN-IoT / RT-IoT / Edge-IIoT: returns the same dict structure as make_fl_data (class-disjoint split)."""
    raise NotImplementedError("Provide the CSV preprocessing of [44]; keep the dict format of make_fl_data().")


# ----------------------------------------------------------------------------- model
class CNNGRU(nn.Module):
    def __init__(self, nc=NC, h=32):
        super().__init__()
        self.gru = nn.ModuleList([nn.GRU(D, h, batch_first=True), nn.GRU(h, h, batch_first=True), nn.GRU(h, h, batch_first=True)])
        def blk(i, o, k): return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.GroupNorm(4, o), nn.ReLU(), nn.MaxPool1d(2, ceil_mode=True))
        self.cnn = nn.Sequential(blk(1, 16, 5), blk(16, 32, 3), blk(32, 32, 1))
        self.fc1 = nn.Linear(h + 32 * 3, 64); self.fc2 = nn.Linear(64, 32); self.drop = nn.Dropout(0.2); self.out = nn.Linear(32, nc)

    def forward(self, x):
        g = x.unsqueeze(1)
        for l in self.gru: g, _ = l(g)
        g = g[:, -1]
        c = self.cnn(x.unsqueeze(1)).flatten(1)
        z = F.relu(self.fc1(torch.cat([g, c], 1))); z = F.relu(self.fc2(z))
        return self.out(self.drop(z))


def get_vec(m): return torch.cat([p.data.view(-1) for p in m.parameters()]).clone()
def set_vec(m, v):
    i = 0
    for p in m.parameters():
        n = p.numel(); p.data.copy_(v[i:i + n].view_as(p)); i += n


# ----------------------------------------------------------------------------- losses
ALPHA_VEC = torch.tensor([0.25] + [0.75] * (NC - 1))
def focal(logits, y, gamma=2.0):
    lp = F.log_softmax(logits, 1).gather(1, y[:, None]).squeeze(1)
    pt = lp.exp()
    return (-ALPHA_VEC[y] * (1 - pt) ** gamma * lp).mean()

def wd_reg(model, gsorted):
    """1-D Wasserstein-2^2 between local and global parameter distributions per tensor (exact OT in 1-D = sorted matching)."""
    t = 0.
    for p, gs in zip(model.parameters(), gsorted):
        t = t + ((p.view(-1).sort()[0] - gs) ** 2).mean()
    return t


# ----------------------------------------------------------------------------- module B : CAFAT
def pgd_ctrl(model, x, target, eps, steps=3, to_target=True, all_feats=False):
    """L_inf PGD restricted to attacker-controllable features; minimise CE to `target` (mimicry)."""
    mask = torch.zeros(D); mask[CTRL] = 1
    if all_feats: mask[:] = 1
    was = model.training; model.eval()
    xa = x.clone(); alpha = 2.5 * eps / steps
    for _ in range(steps):
        xa.requires_grad_(True)
        loss = F.cross_entropy(model(xa), target)
        g, = torch.autograd.grad(loss, xa)
        xa = xa.detach() - alpha * g.sign() * mask
        xa = x + (xa - x).clamp(-eps, eps) * mask
    model.train(was)
    return xa.detach()

def naive_augment(model, xb, yb, eps=0.5):
    m = yb > 0
    if m.sum() < 2: return xb, yb
    xa = pgd_ctrl(model, xb[m], torch.zeros_like(yb[m]), eps, all_feats=True)
    return torch.cat([xb, xa]), torch.cat([yb, yb[m]])

def cafat_augment(model, xb, yb, frac=0.5, eps=0.5, gamma=2.0):
    m = yb > 0
    if m.sum() < 2: return xb, yb
    with torch.no_grad():
        pt = F.softmax(model(xb[m]), 1).gather(1, yb[m][:, None]).squeeze(1)
        hard = (1 - pt) ** gamma
    k = max(1, int(frac * m.sum()))
    sel = torch.topk(hard, k)[1]
    xs, ys = xb[m][sel], yb[m][sel]
    e = eps * (0.5 + torch.rand(1).item())
    xa = pgd_ctrl(model, xs, torch.zeros_like(ys), e)
    return torch.cat([xb, xa]), torch.cat([yb, ys])


# ----------------------------------------------------------------------------- Holt
class Holt:
    def __init__(self, xi=0.99, phi=0.01): self.xi, self.phi, self.H, self.T = xi, phi, None, 0.
    def forecast(self): return None if self.H is None else self.H + self.T
    def update(self, v):
        if self.H is None: self.H, self.T = v, 0.; return
        Hn = self.xi * v + (1 - self.xi) * (self.H + self.T)
        self.T = self.phi * (Hn - self.H) + (1 - self.phi) * self.T; self.H = Hn
    def reset(self): self.H, self.T = None, 0.


# ----------------------------------------------------------------------------- aggregators
def _flat_stack(ds): return torch.stack(ds)

def agg_fedavg(ds, w, **k): w = w / w.sum(); return (ds * w[:, None]).sum(0)
def agg_median(ds, w, **k): return ds.median(0)[0]
def agg_trim(ds, w, f=2, **k): s = ds.sort(0)[0]; return s[f:len(ds) - f].mean(0)
def agg_krum(ds, w, f=2, **k):
    n = len(ds); d = torch.cdist(ds, ds) ** 2
    sc = torch.stack([d[i][torch.arange(n) != i].sort()[0][:n - f - 2].sum() for i in range(n)])
    sel = sc.topk(n - f, largest=False)[1]
    return ds[sel].mean(0)                                   # multi-Krum
def agg_normclip(ds, w, **k):
    nr = ds.norm(dim=1); tau = nr.median()
    ds = ds * torch.clamp(tau / (nr + 1e-12), max=1.0)[:, None]
    return agg_fedavg(ds, w)


class HCTA:
    """Holt-CUSUM Trust Aggregation (module A)."""
    def __init__(self, K, k_ref=1.0, h=3.0, kappa=1.5, decay=0.6, lam=1.0, xi=0.6, phi=0.2, flag_mode='zero', flag_w=0.0):
        self.flag_mode, self.flag_w = flag_mode, flag_w
        self.K, self.k_ref, self.h, self.kappa, self.decay, self.lam = K, k_ref, h, kappa, decay, lam
        self.holt = [Holt(xi, phi) for _ in range(K)]; self.S = np.zeros(K); self.flag = np.zeros(K, bool); self.suspended = 0
        self.log = []

    def reset(self, suspend=2):
        for h in self.holt: h.reset()
        self.S[:] = 0; self.suspended = suspend

    def __call__(self, ds, w, **k):
        n = len(ds); m = ds.median(0)[0]
        nr = ds.norm(dim=1); tau = nr.median()
        dev = (ds - m).norm(dim=1); dev = dev / (dev.median() + 1e-12)
        cos = F.cosine_similarity(ds, m[None], dim=1)
        u = dev + (1 - cos)
        mad = (u - u.median()).abs().median() * 1.4826 + 0.15
        z = ((u - u.median()) / mad).numpy()
        L = np.zeros(n)
        for i in range(n):                                    # Holt-smoothed evidence (client-wise level+trend forecast)
            self.holt[i].update(float(z[i])); L[i] = self.holt[i].H + max(0., self.holt[i].T)
        if self.suspended > 0:
            self.suspended -= 1; wt = np.ones(n)               # drift gate: no quarantine while drift is being absorbed
        else:
            self.S = np.maximum(0., self.S + (L - self.k_ref))
            self.flag = self.S > self.h
            self.S[self.flag] *= self.decay                    # probation decay
            wt = np.exp(-self.lam * np.maximum(0., L - 1.0)); wt[self.flag] = 0.
        if wt.sum() < 1e-6: wt = np.ones(n)
        ds = ds * torch.clamp(self.kappa * tau / (nr + 1e-12), max=1.0)[:, None]   # server-side norm bound
        if self.suspended == 0 or True:
            fl = self.flag.copy()
            if fl.any() and (~fl).sum() >= 2 and self.flag_mode != 'zero':
                if self.flag_mode == 'cwclip':                 # salvage: confine flagged updates to the benign coordinate envelope
                    ben = ds[torch.tensor(~fl)]; lo, hi = ben.min(0)[0], ben.max(0)[0]
                    for i in np.where(fl)[0]: ds[i] = torch.maximum(torch.minimum(ds[i], hi), lo)
                wt[fl] = self.flag_w
        ww = w * torch.tensor(wt, dtype=torch.float32)
        self.log.append(dict(z=z.tolist(), flag=self.flag.tolist()))
        return agg_fedavg(ds, ww)


# ----------------------------------------------------------------------------- FL run
def evaluate(model, X, y, extra=None):
    model.eval()
    with torch.no_grad(): p = model(X).argmax(1)
    acc = (p == y).float().mean().item()
    f1s = []
    for c in range(NC):
        tp = ((p == c) & (y == c)).sum().item(); fp = ((p == c) & (y != c)).sum().item(); fn = ((p != c) & (y == c)).sum().item()
        f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    att = y > 0
    rec = (p[att] > 0).float().mean().item(); fpr = (p[~att] > 0).float().mean().item()
    return dict(acc=acc, f1=float(np.mean(f1s)), rec=rec, fpr=fpr)

def asr(model, X, y):
    model.eval(); m = y > 0; xt = X[m].clone(); xt[:, TRIG] = TRIG_VAL
    with torch.no_grad(): p = model(xt).argmax(1)
    return (p == 0).float().mean().item()

def evasion(model, X, y, eps):
    m = y > 0; xs, ys = X[m], y[m]
    xa = pgd_ctrl(model, xs, torch.zeros_like(ys), eps, steps=10) if eps > 0 else xs
    model.eval()
    with torch.no_grad(): p = model(xa).argmax(1)
    return (p > 0).float().mean().item()                      # attack detection rate under evasion

DEFAULT = dict(rounds=16, epochs=1, lr=0.01, mom=0.9, bs=64, eps_dp=4.0, delta=1e-6, lam_wd=0.1, gamma=2.0, kappa_clip=1.5,
               agg='fedavg', attack='none', n_mal=2, attack_start=4, boost=4.0, cafat=False, cafat_eps=0.5,
               drift_round=None, drift_gate=False, drift_k=0.5, drift_h=3.0, drift_sig=0.1, drift_warm=4, holt_clip=True, dp=True, n_total=28000)

def run_fl(seed, **over):
    cfg = dict(DEFAULT); cfg.update(over)
    torch.manual_seed(seed); np.random.seed(seed)
    data = make_fl_data(seed, cfg['n_total']); K = len(data['clients'])
    model = CNNGRU(); gvec = get_vec(model)
    rng = np.random.default_rng(seed + 77)
    mal = set(rng.choice(K, cfg['n_mal'], replace=False).tolist()) if cfg['attack'] != 'none' else set()
    sigma_dp = math.sqrt(2 * math.log(1.25 / cfg['delta'])) / cfg['eps_dp']
    clip_h = [Holt() for _ in range(K)]; Cs = [1.0] * K
    hcta = HCTA(K, **cfg.get('hcta_kw', {})) if cfg['agg'] == 'hcta' else None
    loss_holt = Holt(0.9, 0.1); cus = 0.; res_var = None; lr_boost = 0; trig_rounds = []
    hist = []; t_srv = 0.; t_cli = 0.
    for r in range(1, cfg['rounds'] + 1):
        drifted = cfg['drift_round'] is not None and r >= cfg['drift_round']
        cd = data['clients_drift'] if drifted else data['clients']
        deltas, ws, losses, gns, probes = [], [], [], [], []
        gsorted = None
        t0 = time.time()
        for k in range(K):
            X, y = cd[k]
            set_vec(model, gvec); model.train()
            if cfg['lam_wd'] > 0:
                gm = CNNGRU(); set_vec(gm, gvec); gsorted = [p.data.view(-1).sort()[0] for p in gm.parameters()]
            with torch.no_grad():                                  # drift probe: global model on up to 1500 local samples, before training
                model.eval(); pi = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed * 100003 + r * 101 + k))[:1500]; probes.append(focal(model(X[pi]), y[pi], cfg['gamma']).item()); model.train()
            lr = cfg['lr'] * (2.0 if lr_boost > 0 else 1.0)
            opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=cfg['mom'])
            is_mal = (k in mal) and r >= cfg['attack_start']
            perm = torch.randperm(len(X)); gn_list, ls = [], []
            for ep in range(cfg['epochs']):
                for b in range(0, len(X), cfg['bs']):
                    idx = perm[b:b + cfg['bs']]; xb, yb = X[idx].clone(), y[idx].clone()
                    if len(idx) < 8: continue
                    if is_mal and cfg['attack'] == 'backdoor':
                        a = (yb > 0) & (torch.rand(len(yb)) < 0.5)
                        xb[a.nonzero().flatten()[:, None], torch.tensor(TRIG)[None]] = TRIG_VAL; yb[a] = 0
                    if is_mal and cfg['attack'] == 'labelflip': yb[yb > 0] = 0
                    if cfg['cafat'] and not is_mal: xb, yb = (naive_augment(model, xb, yb, eps=cfg['cafat_eps']) if cfg.get('at_naive') else cafat_augment(model, xb, yb, eps=cfg['cafat_eps']))
                    model.train()
                    loss = focal(model(xb), yb, cfg['gamma'])
                    if cfg['lam_wd'] > 0: loss = loss + cfg['lam_wd'] * wd_reg(model, gsorted)
                    opt.zero_grad(); loss.backward()
                    gn = torch.sqrt(sum((p.grad ** 2).sum() for p in model.parameters())).item()
                    if cfg['dp'] or cfg['holt_clip']:
                        s = 1.0 / max(1.0, gn / Cs[k])
                        for p in model.parameters(): p.grad.mul_(s)
                    gn_list.append(gn); ls.append(loss.item()); opt.step()
            new = get_vec(model); delta = new - gvec
            if is_mal and cfg['attack'] == 'backdoor': delta = delta * cfg['boost']
            if is_mal and cfg['attack'] == 'signflip': delta = -cfg['boost'] * delta
            if cfg['dp'] and not is_mal:
                std = sigma_dp * 2 * lr * Cs[k] / cfg['bs']
                delta = delta + torch.randn_like(delta) * std
            deltas.append(delta); ws.append(float(len(X))); losses.append(float(np.mean(ls))); gns.append(float(np.mean(gn_list)))
            clip_h[k].update(gns[-1]); Cs[k] = max(0.2, cfg['kappa_clip'] * clip_h[k].forecast())
        t_cli += time.time() - t0
        t0 = time.time()
        ds = torch.stack(deltas); w = torch.tensor(ws)
        if cfg['agg'] == 'hcta': upd = hcta(ds, w)
        else: upd = dict(fedavg=agg_fedavg, median=agg_median, trim=agg_trim, krum=agg_krum, normclip=agg_normclip)[cfg['agg']](ds, w)
        gvec = gvec + upd
        # module C : drift gate
        sig = probes
        if cfg.get('dp_loss'):   # release probe loss with clipping (L_max=2) + Laplace(L_max/(n_k*eps_l)), eps_l=0.05/round
            sig = [min(l, 2.0) + float(np.random.laplace(0, 2.0 / (n * 0.05))) for l, n in zip(probes, ws)]
        gl = float(np.median(sig))                                  # robust to a few malicious reporters
        gl_train = float(np.average(losses, weights=ws))
        if cfg['drift_gate']:                                     # module C: Holt forecast of log probe loss + CUSUM on the standardized residual
            lg = math.log(max(gl, 1e-3)); f = loss_holt.forecast()
            if f is not None and r > cfg['drift_warm']:
                z = (lg - f) / cfg['drift_sig']; cus = max(0., cus + z - cfg['drift_k'])
                if cus > cfg['drift_h']:
                    trig_rounds.append(r); cus = 0.; lr_boost = 2
                    for h in clip_h: h.reset()
                    Cs = [max(Cs) for _ in Cs]
                    if hcta: hcta.reset(2)
            loss_holt.update(lg)
        if lr_boost > 0 and not (trig_rounds and trig_rounds[-1] == r): lr_boost -= 1
        t_srv += time.time() - t0
        set_vec(model, gvec)
        Xte = data['Xte_drift'] if drifted else data['Xte']
        ev = evaluate(model, Xte, data['yte']); ev['round'] = r; ev['loss'] = gl_train; ev['probe'] = gl
        hist.append(ev)
    set_vec(model, gvec)
    Xf = data['Xte_drift'] if cfg['drift_round'] else data['Xte']
    final = dict(hist[-1]); final['asr'] = asr(model, Xf, data['yte'])
    for e in (0.25, 0.5, 1.0): final[f'ev{e}'] = evasion(model, Xf, data['yte'], e)
    final['ev0'] = final['rec']
    out = dict(cfg=cfg, seed=seed, final=final, hist=hist, t_cli=t_cli / cfg['rounds'], t_srv=t_srv / cfg['rounds'],
               mal=sorted(mal), triggers=trig_rounds, nparams=int(gvec.numel()))
    if hcta: out['hcta_log'] = hcta.log
    return out
