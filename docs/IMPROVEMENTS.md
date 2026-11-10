# SFLNID-R: status, findings and what to improve next

## What you have

| File | What it is |
|---|---|
| `overleaf/main.tex` | Full paper in IEEEtran journal format (single file, tables inlined) |
| `overleaf/figs/*.pdf` | Six figures the paper uses |
| `code/sflnid_r.py` | Everything: data generator, CNN-GRU, Focal + Wasserstein, Holt DP, HCTA, CAFAT, drift gate |
| `code/run_all.py`, `run_extra.py`, `analyze.py` | Experiment grid, extra runs, tables and figures |
| `results/*.json` | Raw output of every run (per-round history included) |
| `summary.json` | Aggregated numbers used in the paper |

To compile: create an Overleaf project, upload `main.tex` and the `figs` folder, keep the compiler on pdfLaTeX. Replace the author block.

## Read this first: what is real and what is not

- The three datasets (ToN-IoT, RT-IoT, Edge-IIoT) could not be downloaded from the workspace. Every number comes from a seeded **synthetic** benchmark that copies the base paper's protocol (10 classes, 19 features, heavy imbalance, class-disjoint non-IID split over 7 clients). The paper says this in the abstract, the setup and the limitations.
- Absolute numbers are lower than the base paper (about 0.91 accuracy for plain FL) because the benchmark is deliberately harder. Do not compare them with the base paper's 0.98.
- 3 seeds only. Hyperparameters were tuned on seeds 100 and 101, results are on seeds 0 to 2.
- Before you submit or present this as a finding, rerun on the real CSV files (steps below).

## Findings, including the ones that did not work

1. **HCTA (poisoning).** Backdoor ASR 0.979 to 0.520 (clean floor 0.471). Sign-flip macro-F1 0.051 to 0.732. Flags 94% of malicious client-rounds with a 0.75 round delay.
   - It does **not** beat Multi-Krum (backdoor F1 0.749 vs 0.788, sign-flip 0.732 vs 0.789).
   - It has a clean-data cost: macro-F1 0.764 vs 0.835 for FedAvg, because it falsely quarantined one benign client for 7 and 9 rounds in 2 of 3 clean seeds.
2. **CAFAT (evasion).** Detection under PGD at eps 0.25 / 0.5 / 1.0 goes from 75.2 / 67.8 / 51.8 to 79.4 / 75.0 / 64.8. Naive adversarial training gets about the same robustness, so the real advantage is clean cost (F1 0.829 vs 0.801, FPR 0.012 vs 0.016). Doubling the noise to keep the same privacy guarantee costs nothing. Client time doubles.
3. **DGA (drift).** Negative result. It fires within one round of the drift in 2 of 3 seeds but also fires 2 to 3 extra times, and in no-drift controls it fires in all 3 seeds. It hurt slightly, and in the combined run it made HCTA worse because every firing suspends quarantine.
4. **Best combination:** A + B (ASR 0.323, detection at eps 0.5 of 0.755, F1 0.805).

## Improvements worth trying, in priority order

### 1. Fix the drift gate (most likely quick win)
- The gate was tuned on noise-free loss trajectories and run with Laplace noise (eps_l = 0.05). Retune it **with the noise on**, using several tuning seeds, or raise eps_l to 0.2 and report the extra budget.
- Stop suspending quarantine on every trigger. Instead, only suspend it for clients whose update changed in the direction of the drifted classes, or require two consecutive triggers.
- Make the drift stronger and add a realistic variant (normal traffic shifts too). The current drift is mild: the gap to the no-drift run is gone by round 14, so a gate has little to win.
- Related idea from the literature: cluster model updates with PCA and K-means to find which nodes drifted (Manias et al., arXiv:2109.06088) and combine it with your loss signal.

### 2. Reduce HCTA's false quarantines
- Tune `h` and `k` using seeds with attack-free runs as a hard constraint (zero false quarantine in clean runs).
- Add a probation rule that readmits a client after 2 clean rounds instead of using only the decay factor.
- Use a second signal for the decision, such as the per-class output-layer update pattern, since label-flip style attacks look like a client that simply lacks a class.
- Partial weight and coordinate clipping for flagged clients were tried on the tuning seed: they gave at most +0.02 F1 and let the backdoor back in (ASR 0.62 to 0.97). A better idea is to keep only the backbone layers of a flagged client and drop its classifier head, then test it.

### 3. Make DP and robustness actually interact
- In SFLNID the DP noise std is about `sigma * 2 * lr * C / B`, tiny next to a boosted update. So DP is not a robustness tool here. Try client-level update clipping with larger noise, then let HCTA use the known noise floor to calibrate its score. This is the regime where the DP and Byzantine robustness trade-off (Zhu and Ling, arXiv:2205.00107) really appears.

### 4. Stronger and adaptive attacks
- Slow poisoning that stays under the CUSUM threshold, attacks that mimic the median update, and a malicious fraction of 3 of 7.
- A true adaptive evasion attack that also uses the feature constraints of real flows (integer ports, bounded byte counts).

### 5. CAFAT upgrades
- Curriculum on eps, and a certified or randomized-smoothing variant on the controllable features only.
- Try the hardness fraction beta in {0.25, 0.5, 1.0} and 5 PGD steps. Only beta = 0.5 and 3 steps were run.

## Running on the real datasets

1. Download the three Kaggle CSVs (links are in the base paper's references 65 to 67) and preprocess like Yao et al. (19 features + label), as the base paper does.
2. Implement `load_real_csv` in `code/sflnid_r.py`. It must return the same dict as `make_fl_data`: `clients` (list of `(X, y)` tensors), `clients_drift`, `Xte`, `Xte_drift`, `yte`. Use class-disjoint client assignment as in `CLIENT_CLASSES`.
3. Set `CTRL` to the indices of features an attacker can really change (packet size, inter-arrival time, payload length, flags rate, duration) and `TRIG` to three features you consider non-controllable.
4. Change `NC`, `D`, `CLASSES` per dataset, run `python run_all.py`, then `python analyze.py`. Expect about 3 minutes per run per CPU core with this model size.
5. Update the numbers in the paper from `summary.json`. The prose in section VI quotes specific values, so edit those sentences, do not just swap the tables.

## Honest wording to keep in the paper
- "Synthetic benchmark following the protocol of the base paper" (not "real datasets").
- HCTA "matches" the strongest robust baseline on ASR but is not better; its value is attribution.
- DGA is a negative result.
