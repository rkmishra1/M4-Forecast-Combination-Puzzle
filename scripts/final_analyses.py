"""Post-full-run analyses, computed from cached outputs only.

(i) Diebold-Mariano tests (Harvey small-sample correction) with Romano-Wolf
    step-down familywise control across all rules x frequencies.
(ii) Pool-composition ablations (classical-only, foundation-only) and the
     pool-size curve k=2..8 from cached forecasts (R=20 seeded subsets).
(iii) Weight drift: distance of estimated weights from uniform, and how
      often a rule beats the equal-weight combination per series.
(iv) Runtime appendix parsed from the run log.

All randomness is seeded (42). Outputs -> results/full1/analyses/.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fcc.data import load_m4  # noqa: E402
from fcc.metrics import mase_scale, mse_scale  # noqa: E402

RES = ROOT / "results" / "full1"
OUT = RES / "analyses"
OUT.mkdir(exist_ok=True)
CACHE = RES / "cache"

MODELS = [
    "naive2", "autotheta", "autoets", "autoarima",
    "chronos-t5-small", "chronos-t5-base", "chronos-t5-large",
    "chronos-bolt-base", "timesfm-2.5-200m",
]
CLASSICAL = MODELS[:4]
FOUNDATION = MODELS[4:]
RULES = ["equal", "median", "trim1_equal", "inv_error_global",
         "inv_error_series", "inv_rank_global", "inv_rank_series",
         "nnls_series", "best_single_val"]
FREQS = ["hourly", "daily", "weekly", "monthly", "quarterly", "yearly"]
EPS = 1e-6
SEED = 42
B = 999  # bootstrap replicates for Romano-Wolf


def smape_mat(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    denom = np.abs(y) + np.abs(p)
    num = 2.0 * np.abs(y - p)
    out = np.where(denom > 0, num / np.maximum(denom, 1e-12), 0.0)
    return out.mean(axis=1)


def rmss_mat(y: np.ndarray, p: np.ndarray, s2: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((y - p) ** 2, axis=1) / s2)


def cache_preds(freq: str, model: str, split: str) -> dict[str, np.ndarray]:
    df = pd.read_parquet(CACHE / f"{freq}__{model}__{split}.parquet")
    return {
        str(uid): g.sort_values("step")["y_pred"].to_numpy(dtype=float)
        for uid, g in df.groupby("unique_id", sort=False)
    }


def pooled_owa(per_freq: dict[str, dict[str, float]]) -> dict:
    """per_freq[f] = {'mase': mean_mase, 'smape': ..., 'rmsse': ..., 'n': n}"""
    tot = sum(v["n"] for v in per_freq.values())
    pm = sum(v["mase"] * v["n"] for v in per_freq.values()) / tot
    ps = sum(v["smape"] * v["n"] for v in per_freq.values()) / tot
    pr = sum(v["rmsse"] * v["n"] for v in per_freq.values()) / tot
    return dict(mase=round(pm, 4), smape=round(ps, 4), rmsse=round(pr, 4),
                owa=round(0.5 * (ps / N_BENCH_SMAPE + pm / N_BENCH_MASE), 4))


# ---------------------------------------------------------------- load core
print("loading metrics_per_series ...", flush=True)
met = pd.read_csv(RES / "metrics_per_series.csv").dropna(subset=["mase"])
nb = met[met.method == "naive2"]
N_BENCH_MASE = float(nb["mase"].mean())
N_BENCH_SMAPE = float(nb["smape"].mean())
print(f"  benchmark naive2 pooled: mase={N_BENCH_MASE:.4f} "
      f"smape={N_BENCH_SMAPE:.4f}", flush=True)

pivot = met.pivot_table(index=["freq", "unique_id"], columns="method",
                        values="mase", aggfunc="first")

valerr = pd.read_csv(RES / "val_errors.csv")
ve = valerr.pivot_table(index=["freq", "unique_id"], columns="model",
                        values="val_mase", aggfunc="first")

# ================================================================= (i) DM
print("\n=== (i) Diebold-Mariano + Romano-Wolf ===", flush=True)
rng = np.random.default_rng(SEED)

hyps: list[tuple[str, str, str]] = []  # (freq, rule-with-prefix, target)
D: dict[str, np.ndarray] = {}
best_ind_of: dict[str, str] = {}
for f in FREQS:
    pf = pivot.loc[f]
    models_f = [m for m in MODELS if m in pf.columns]
    rules_f = [f"combo/{r}" for r in RULES if f"combo/{r}" in pf.columns]
    best_ind = min(models_f, key=lambda m: pf[m].mean())
    best_ind_of[f] = best_ind
    cols, names = [], []
    for r in rules_f:
        pairs = [best_ind] if r == "combo/equal" else [best_ind, "combo/equal"]
        for tgt in pairs:
            cols.append(pf[r].to_numpy() - pf[tgt].to_numpy())
            names.append((r, tgt))
    D[f] = np.column_stack(cols)
    for (r, tgt) in names:
        hyps.append((f, r, tgt))
    print(f"  [{f}] n={len(D[f])} best_ind={best_ind} hyps={len(names)}",
          flush=True)

raw: dict[tuple[str, str, str], tuple[float, float, int]] = {}
for f in FREQS:
    Dm, n = D[f], len(D[f])
    fs_hyps = [(r, t) for (ff, r, t) in hyps if ff == f]
    for j, (r, tgt) in enumerate(fs_hyps):
        d = Dm[:, j]
        t = d.mean() / d.std(ddof=1) * np.sqrt(n)
        raw[(f, r, tgt)] = (t * np.sqrt((n - 1) / n), float(d.mean()), n)

print(f"  Romano-Wolf bootstrap B={B} ...", flush=True)
H = len(hyps)
hidx = {h: j for j, h in enumerate(hyps)}
f_slice = {}
for f in FREQS:
    js = [j for j, (ff, _, _) in enumerate(hyps) if ff == f]
    f_slice[f] = (js[0], js[-1] + 1)
Dc = {f: D[f] - D[f].mean(axis=0) for f in FREQS}
boot = np.empty((B, H))
for b in range(B):
    for f in FREQS:
        nf = len(Dc[f])
        idx = rng.integers(0, nf, nf)
        samp = Dc[f][idx]
        sd = samp.std(axis=0)
        se = np.where(sd > 1e-12, sd / np.sqrt(nf), 1.0)  # standard error
        j0, j1 = f_slice[f]
        boot[b, j0:j1] = samp.mean(axis=0) / se
maxstats = np.abs(boot)

order = sorted(hyps, key=lambda h: -abs(raw[h][0]))
abs_t = {h: abs(raw[h][0]) for h in hyps}
padj: dict[tuple[str, str, str], float] = {}
running = 0.0
for s, h in enumerate(order):
    sidx = [hidx[x] for x in order[s:]]
    mx = maxstats[:, sidx].max(axis=1)
    p_star = (1 + int((mx >= abs_t[h]).sum())) / (B + 1)
    running = max(running, min(1.0, p_star))
    padj[h] = running

rows = []
for h in hyps:
    t_hln, dmean, n = raw[h]
    rows.append(dict(freq=h[0], rule=h[1].replace("combo/", ""),
                     target=h[2].replace("combo/", ""), n=n,
                     mean_loss_diff=dmean, dm_hln=t_hln,
                     p_raw=2 * (1 - sps.t.cdf(abs(t_hln), df=n - 1)),
                     p_rwolf=padj[h]))
dm = pd.DataFrame(rows)
dm.to_csv(OUT / "dm_tests.csv", index=False)
print(f"  tests={len(dm)}  significant (RW 5%): "
      f"{int((dm.p_rwolf < 0.05).sum())} / {len(dm)}")
print(dm.groupby("target").agg(n=("p_rwolf", "size"),
                               pmin=("p_rwolf", "min"),
                               pmax=("p_rwolf", "max")))

# ============================================================ (ii) ablations
print("\n=== (ii) pool ablations + size curve ===", flush=True)
rng2 = np.random.default_rng(SEED)
RULES_ABL = ["equal", "median", "trim1_equal", "inv_error_series",
             "inv_rank_series"]
acc: dict[tuple[str, str], dict[str, list]] = {}


def score(freq: str, ids: list[str], y: np.ndarray, s_t: np.ndarray,
          s2_t: np.ndarray, P: np.ndarray, key: tuple[str, str]) -> None:
    a = acc.setdefault(key, {"mase": [], "smape": [], "rmsse": []})
    a["mase"].append((freq, len(ids), float((np.mean(np.abs(y - P), axis=1) / s_t).mean())))
    a["smape"].append((freq, len(ids), float(smape_mat(y, P).mean())))
    a["rmsse"].append((freq, len(ids), float(rmss_mat(y, P, s2_t).mean())))


def combo_pred(rule: str, ts: np.ndarray, eS: np.ndarray) -> np.ndarray:
    n, k, _ = ts.shape
    if rule == "equal":
        return ts.mean(axis=1)
    if rule == "median":
        return np.median(ts, axis=1)
    if rule == "trim1_equal":
        worst = np.nanargmax(eS, axis=1)
        mask = np.ones((n, k))
        mask[np.arange(n), worst] = 0.0
        w = mask / mask.sum(axis=1, keepdims=True)
    elif rule == "inv_error_series":
        w = 1.0 / (eS + EPS)
        w /= w.sum(axis=1, keepdims=True)
    elif rule == "inv_rank_series":
        r = np.argsort(np.argsort(eS, axis=1), axis=1) + 1.0
        w = 1.0 / r
        w /= w.sum(axis=1, keepdims=True)
    elif rule == "best_single_val":
        best = np.nanargmin(eS, axis=1)
        w = np.zeros((n, k))
        w[np.arange(n), best] = 1.0
    else:
        raise ValueError(rule)
    return np.einsum("nk,nkh->nh", w, ts)


for f in FREQS:
    print(f"  [{f}] loading caches ...", flush=True)
    data = load_m4(f, str(ROOT / "data"), None, SEED)
    tp = {m: cache_preds(f, m, "test") for m in MODELS}
    ids = list(tp["naive2"])
    h, mper = data.h, data.m
    n = len(ids)
    y = np.zeros((n, h)); s_t = np.zeros(n); s2_t = np.zeros(n)
    for i, u in enumerate(ids):
        sp = data.splits(u)
        y[i] = sp["test_target"]
        s_t[i] = mase_scale(sp["train"], mper)
        s2_t[i] = mse_scale(sp["train"], mper)
    ts_all = np.stack([np.array([tp[m][u] for u in ids])
                       for m in MODELS], axis=1)  # [n, 9, h]
    eS_all = ve.loc[f].reindex(ids)[MODELS].to_numpy()

    def evaluate(tag: str, sidx: list[int], rules: list[str]) -> None:
        ts = ts_all[:, sidx]
        eS = eS_all[:, sidx]
        for rule in rules:
            P = combo_pred(rule, ts, eS)
            score(f, ids, y, s_t, s2_t, P, (tag, rule))

    evaluate("classical(k=4)", [MODELS.index(m) for m in CLASSICAL], RULES_ABL)
    evaluate("foundation(k=5)", [MODELS.index(m) for m in FOUNDATION], RULES_ABL)
    for k in range(2, 9):
        for rep in range(20):
            sidx = sorted(rng2.choice(9, size=k, replace=False).tolist())
            evaluate(f"rand_k{k}#{rep}", sidx, ["equal", "inv_error_series"])
    del data, tp, ts_all
    print(f"  [{f}] done", flush=True)

abl_rows, curve_rows = [], []
for (tag, rule), a in acc.items():
    pf = {fr: {"mase": mv, "smape": sv, "rmsse": rv, "n": nn}
          for fr, nn, mv in a["mase"]
          for sv in [next(x for fq, _, x in a["smape"] if fq == fr)]
          for rv in [next(x for fq, _, x in a["rmsse"] if fq == fr)]}
    o = pooled_owa(pf)
    if tag.startswith("rand_"):
        k = int(tag.split("#")[0].split("k")[1])
        curve_rows.append(dict(k=k, rep=tag.split("#")[1], rule=rule, **o))
    else:
        abl_rows.append(dict(pool=tag, rule=rule, **o))
abl = pd.DataFrame(abl_rows).sort_values(["pool", "owa"])
abl.to_csv(OUT / "ablations.csv", index=False)
curve = pd.DataFrame(curve_rows)
curve.to_csv(OUT / "poolsize_curve.csv", index=False)
print(abl.to_string(index=False))
cs = curve.groupby(["k", "rule"]).owa.agg(["mean", "min", "max"]).round(4)
print(cs)

# ========================================================= (iii) weight drift
print("\n=== (iii) weight drift ===", flush=True)
k9 = len(MODELS)
w = weights[weights.rule != "equal"].copy()
w["dev"] = (w.weight - 1.0 / k9).abs()
tv = w.groupby(["rule", "freq", "unique_id"], sort=False)["dev"].sum() * 0.5
tv = tv.rename("tv").reset_index()
ent = w.groupby(["rule", "freq", "unique_id"], sort=False)["weight"].sum()  # =1
pW = w.copy()
pW["plogp"] = np.where(w.weight > 0, w.weight * np.log(np.maximum(w.weight, 1e-300)), 0.0)
neff = (pW.groupby(["rule", "freq", "unique_id"])["plogp"].sum() * -1).rename("h")
neff = np.exp(neff).rename("neff").reset_index()
drift = tv.merge(neff, on=["rule", "freq", "unique_id"])
drift_sum = drift.groupby(["rule", "freq"]).agg(
    mean_tv=("tv", "mean"), mean_neff=("neff", "mean")).round(4).reset_index()
drift_sum.to_csv(OUT / "weight_drift_by_freq.csv", index=False)
drift_pooled = drift.groupby("rule").agg(mean_tv=("tv", "mean"),
                                         mean_neff=("neff", "mean")).round(4)
drift_pooled.to_csv(OUT / "weight_drift_pooled.csv")

# % of series where each rule beats the equal-weight combination
meth = met.pivot_table(index=["freq", "unique_id"], columns="method",
                       values="mase", aggfunc="first")
eq = meth["combo/equal"]
beat_rows = []
for r in RULES:
    col = f"combo/{r}"
    if col not in meth.columns:
        continue
    diff = meth[col] - eq
    beat_rows.append(dict(rule=r,
                          pct_beats_equal=round(float((diff < 0).mean() * 100), 1),
                          mean_mase_ratio=round(float((meth[col] / eq).mean()), 4)))
beat = pd.DataFrame(beat_rows)
beat.to_csv(OUT / "beats_equal.csv", index=False)
print(drift_pooled)
print(beat)

# ============================================================ (iv) runtimes
print("\n=== (iv) runtimes from log ===", flush=True)
pat = re.compile(r"\[(\w+)\] (\S+) (val|test): ([\d.]+)s")
rt = pd.DataFrame(
    [(f, m, s, float(v)) for f, m, s, v in pat.findall((ROOT / "full.log").read_text())],
    columns=["freq", "model", "split", "seconds"])
rt = rt.drop_duplicates(subset=["freq", "model", "split"])
rt.to_csv(OUT / "runtimes_full.csv", index=False)
print(f"  cells: {len(rt)} (expect 108)")
tot = rt.groupby("model").seconds.sum().sort_values().round(0)
print((tot / 3600).round(2))

print("\nAll analyses written to", OUT)
