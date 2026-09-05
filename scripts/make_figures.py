"""Three extra manuscript figures from cached full-run outputs."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results" / "full1"
CACHE = RES / "cache"
FIGS = ROOT / "paper" / "figs"
FIGS.mkdir(exist_ok=True)

MODELS = ["naive2","autotheta","autoets","autoarima","chronos-t5-small",
          "chronos-t5-base","chronos-t5-large","chronos-bolt-base","timesfm-2.5-200m"]
FREQS = ["hourly","daily","weekly","monthly","quarterly","yearly"]
SEED = 42

def cache_preds(freq, model, split):
    df = pd.read_parquet(CACHE / f"{freq}__{model}__{split}.parquet")
    return {str(u): g.sort_values("step")["y_pred"].to_numpy(dtype=float)
            for u, g in df.groupby("unique_id", sort=False)}

met = pd.read_csv(RES / "metrics_per_series.csv").dropna(subset=["mase"])
ve = pd.read_csv(RES / "val_errors.csv")
vep = ve.pivot_table(index=["freq", "unique_id"], columns="model",
                     values="val_mase", aggfunc="first")

# ---- Fig A: example forecast paths (one series per chosen frequency) -------
from fcc.data import load_m4
from fcc.metrics import mase
PANELS = [("monthly", 18), ("weekly", 13), ("hourly", 48), ("yearly", 6)]
fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.2))
rng = np.random.default_rng(SEED)
for ax, (freq, h) in zip(axes.ravel(), PANELS):
    data = load_m4(freq, str(ROOT / "data"), None, SEED)
    tp = cache_preds(freq, "autotheta", "test")   # ids shared across models
    ids = list(tp)
    sub = met[met.freq == freq]
    eq = sub[sub.method == "combo/equal"].set_index("unique_id")["mase"]
    med = float(eq.median())
    uid = min(ids, key=lambda u: abs(float(eq.get(u, np.nan)) - med))
    T = data.train_len[uid]
    ctx = data.series[uid][max(0, T - 3 * h):T]
    y = data.series[uid][T:T + h]
    tctx = np.arange(len(ctx)); tft = np.arange(len(ctx), len(ctx) + h)
    preds = {m: cache_preds(freq, m, "test")[uid] for m in MODELS}
    stack = np.vstack(list(preds.values()))
    equal = stack.mean(axis=0)
    eve = vep.loc[(freq, uid)][MODELS].to_numpy()
    w = 1.0 / (eve + 1e-6); w /= w.sum()
    ies = w @ stack
    ax.plot(tctx, ctx, color="#444", lw=0.9, label="context")
    ax.plot(tft, y, color="#000", lw=1.6, label="actuals (test)")
    ax.plot(tft, equal, color="#1b9e77", lw=1.3, label="equal weight")
    ax.plot(tft, ies, color="#d95f02", lw=1.3, ls="--", label="inv. val. error")
    ax.fill_between(tft, stack.min(axis=0), stack.max(axis=0),
                    color="#7570b3", alpha=0.15, lw=0, label="pool range")
    ax.axvline(len(ctx), color="#999", lw=0.7, ls=":")
    ax.set_title(f"{freq} ({uid}, h={h})", fontsize=9.5)
    ax.tick_params(labelsize=8)
axes[0, 0].legend(fontsize=7.5, frameon=False, ncol=1, loc="best")
fig.tight_layout()
fig.savefig(FIGS / "example_series.png", dpi=200)
plt.close(fig)
print("saved example_series.png")

# ---- Fig B: % of series beating equal weights, per rule and frequency -----
pivot = met.pivot_table(index=["freq", "unique_id"], columns="method",
                        values="mase", aggfunc="first")
eqc = pivot["combo/equal"]
RULES = ["combo/inv_error_series", "combo/trim1_equal", "combo/median",
         "combo/nnls_series", "combo/best_single_val"]
LAB = {"combo/inv_error_series": "inv. val. error",
       "combo/trim1_equal": "trim1", "combo/median": "median",
       "combo/nnls_series": "NNLS", "combo/best_single_val": "best-on-val"}
wr = {r: [float((pivot.loc[f, r] < eqc.loc[f]).mean() * 100) for f in FREQS]
      for r in RULES}
fig, ax = plt.subplots(figsize=(8.6, 4.0))
x = np.arange(len(FREQS)); nb = len(RULES); wd = 0.8 / nb
cols = ["#1b9e77", "#66c2a5", "#8da0cb", "#e78ac3", "#fc8d62"]
for i, r in enumerate(RULES):
    ax.bar(x + i * wd - 0.4 + wd / 2, wr[r], wd, label=LAB[r], color=cols[i])
ax.axhline(50, color="#444", lw=1.0, ls="--")
ax.text(len(FREQS) - 0.45, 50.6, "50%", fontsize=8, color="#444")
ax.set_xticks(x, FREQS, fontsize=9)
ax.set_ylabel("% of series beating equal weights", fontsize=9.5)
ax.legend(fontsize=8, frameon=False, ncol=5, loc="upper center",
          bbox_to_anchor=(0.5, 1.14))
fig.tight_layout()
fig.savefig(FIGS / "winrate.png", dpi=200)
plt.close(fig)
print("saved winrate.png")
wrdf = pd.DataFrame({LAB[r]: wr[r] for r in RULES}, index=FREQS).round(1)
print(wrdf.to_string())
wrdf.to_csv(RES / "analyses" / "winrate_by_freq.csv")

# ---- Fig C: does the validation-best model stay best on test? -------------
agree, chance = [], []
for f in FREQS:
    v = vep.loc[f][MODELS]; t = pivot.loc[f][MODELS]
    bv = v.to_numpy().argmin(axis=1)
    bt = t.reindex(v.index).to_numpy().argmin(axis=1)
    agree.append(float((bv == bt).mean() * 100))
    chance.append(100.0 / len(MODELS))
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ax.bar(np.arange(len(FREQS)), agree, 0.62, color="#7570b3")
ax.axhline(100 / 9, color="#d95f02", lw=1.1, ls="--")
ax.text(len(FREQS) - 0.4, 100 / 9 + 0.8, "chance (11.1%)", fontsize=8.5,
        color="#d95f02", ha="right")
for i, a in enumerate(agree):
    ax.text(i, a + 0.6, f"{a:.0f}", ha="center", fontsize=8.5)
ax.set_xticks(np.arange(len(FREQS)), FREQS, fontsize=9)
ax.set_ylabel("agreement (%)", fontsize=9.5)
ax.set_ylim(0, max(agree) + 9)
fig.tight_layout()
fig.savefig(FIGS / "selagree.png", dpi=200)
plt.close(fig)
print("saved selagree.png")
print("agreement:", dict(zip(FREQS, np.round(agree, 1))))
