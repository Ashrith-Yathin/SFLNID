# SFLNID-R

Robustness extension of SFLNID (federated, DP-protected CNN-GRU intrusion detection for SDN-enabled IIoT).
It adds three modules on top of SFLNID and leaves its loss, adaptive DP and CNN-GRU unchanged:

| Module | Targets | Idea |
|---|---|---|
| **A. HCTA** | poisoning | server-side Holt forecast + CUSUM on a median-consensus deviation score, quarantine, trust weights, norm bound |
| **B. CAFAT** | evasion | Focal-factor hardness selection + PGD limited to attacker-controllable features |
| **C. DGA** | concept drift | Holt/CUSUM drift gate on a privately released loss (reported as a negative result) |

> **Data caveat.** The ToN-IoT, RT-IoT and Edge-IIoT files were not reachable when the experiments were run.
> All reported numbers come from a seeded synthetic benchmark that follows the SFLNID protocol
> (10 classes, 19 features, class-disjoint non-IID split over 7 clients). Absolute values are not comparable
> with the base paper. See `docs/IMPROVEMENTS.md` for how to plug in the real CSVs.

## Layout
```
src/sflnid_r.py     data generator, CNN-GRU, Focal + Wasserstein, Holt DP, HCTA, CAFAT, drift gate, FL loop
scripts/run_all.py  main experiment grid (results cached per run in results/*.json)
scripts/run_extra.py extra runs
scripts/analyze.py  builds tables (paper/tab_*.tex), figures (paper/figs) and results/summary.json
scripts/tune*.py    hyperparameter tuning on separate seeds (100, 101)
results/            raw per-run JSON (with per-round history) and summary.json
paper/main.tex      IEEEtran paper source (upload with paper/figs to Overleaf)
docs/IMPROVEMENTS.md findings and next steps
```

## Run
```bash
pip install -r requirements.txt
python scripts/run_all.py      # skips runs that already exist in results/
python scripts/run_extra.py
python scripts/analyze.py      # regenerates tables, figures, summary.json
```
Results for seeds 0, 1, 2 are included, so `analyze.py` works without retraining.
A single run takes about 3 minutes on one CPU core. Delete `results/*.json` to rerun from scratch.

## Real datasets
Implement `load_real_csv` in `src/sflnid_r.py` (it must return the same dict as `make_fl_data`),
set `CTRL` (attacker-controllable feature indices) and `TRIG` (backdoor trigger features), then rerun.

## Key results (synthetic benchmark, 3 seeds)
- HCTA: backdoor ASR 0.979 to 0.520; sign-flip macro-F1 0.051 to 0.732. Does not beat Multi-Krum; costs macro-F1 on clean data.
- CAFAT: attack detection under PGD improves by 4 to 13 points at a small clean cost; DP-matched version is free.
- DGA: negative result.
- Best combination: HCTA + CAFAT.
