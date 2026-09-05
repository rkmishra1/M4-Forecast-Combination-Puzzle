"""Result aggregation: summary tables, markdown report, and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Display order: benchmark, classical, foundation models, then combinations.
def _method_order(methods):
    def key(mn):
        if mn == "naive2":
            return (0, 0, mn)
        if mn.startswith("combo/oracle"):
            return (4, 0, mn)
        if mn.startswith("combo/"):
            return (3, 0, mn)
        if mn.startswith(("chronos", "timesfm", "moirai", "lag-llama", "ttm")):
            return (2, 0, mn)
        return (1, 0, mn)

    return sorted(methods, key=key)


def analyze(results_dir: str | Path) -> dict:
    results_dir = Path(results_dir)
    m = pd.read_csv(results_dir / "metrics_per_series.csv")
    m = m.dropna(subset=["mase"])

    # Per (freq, method) means
    summary = (
        m.groupby(["freq", "method"], as_index=False)[["mase", "smape", "rmsse"]]
        .mean()
        .round(4)
    )

    # Pooled means + OWA relative to naive2 (computed over all series)
    pooled = m.groupby("method", as_index=False)[["mase", "smape", "rmsse"]].mean()
    bench = pooled.loc[pooled["method"] == "naive2"].iloc[0]
    pooled["owa"] = 0.5 * (
        pooled["smape"] / bench["smape"] + pooled["mase"] / bench["mase"]
    )
    pooled["freq"] = "overall"
    pooled = pooled[["freq", "method", "mase", "smape", "rmsse", "owa"]].round(4)
    summary["owa"] = np.nan
    full = pd.concat([summary, pooled], ignore_index=True)

    # Mean per-series rank of MASE (within freq, across methods)
    piv = m.pivot_table(index=["freq", "unique_id"], columns="method", values="mase")
    ranks = piv.rank(axis=1).mean(axis=0).rename("mean_rank").reset_index()
    ranks["mean_rank"] = ranks["mean_rank"].round(3)

    order = _method_order(list(piv.columns))
    full = full.sort_values(
        ["freq"], key=lambda s: s.map(lambda x: 0 if x == "overall" else 1)
    )
    full.to_csv(results_dir / "summary.csv", index=False)
    ranks.set_index("method").loc[order].reset_index().to_csv(
        results_dir / "ranks.csv", index=False
    )

    # ---- markdown report
    md: list[str] = ["# Forecast combination pilot — results summary\n"]
    md.append("## Pooled performance (all frequencies)\n")
    pooled_tbl = (
        pooled.set_index("method").loc[order].reset_index()[
            ["method", "mase", "smape", "rmsse", "owa"]
        ]
    )
    md.append(pooled_tbl.to_markdown(index=False))
    md.append("\n\n## Mean MASE by frequency\n")
    heat = full[full["freq"] != "overall"].pivot(
        index="method", columns="freq", values="mase"
    ).loc[order]
    md.append(heat.round(3).to_markdown())
    md.append("\n\n## Mean per-series MASE rank (lower is better, pooled)\n")
    md.append(
        ranks.set_index("method").loc[order].reset_index().to_markdown(index=False)
    )
    (results_dir / "summary.md").write_text("\n".join(md))

    # ---- figures
    figs = results_dir / "figs"
    figs.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, max(4, 0.42 * len(order))))
    hm = heat.astype(float)
    im = ax.imshow(hm.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(hm.shape[1]), hm.columns, rotation=30, ha="right")
    ax.set_yticks(range(hm.shape[0]), hm.index)
    for i in range(hm.shape[0]):
        for j in range(hm.shape[1]):
            v = hm.values[i, j]
            if np.isfinite(v):
                ax.text(
                    j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if v > np.nanmedian(hm) else "black", fontsize=8,
                )
    ax.set_title("Mean MASE by frequency and method (test window)")
    fig.colorbar(im, ax=ax, label="MASE", shrink=0.8)
    fig.tight_layout()
    fig.savefig(figs / "mase_heatmap.png", dpi=200)
    plt.close(fig)

    rk = ranks.set_index("method").loc[order]["mean_rank"]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(order))))
    colors = ["#888"] * len(rk)
    for i, mn in enumerate(rk.index):
        if mn.startswith("combo/oracle"):
            colors[i] = "#d95f02"
        elif mn.startswith("combo/"):
            colors[i] = "#1b9e77"
        elif mn.startswith(("chronos", "timesfm")):
            colors[i] = "#7570b3"
    ax.barh(range(len(rk)), rk.values, color=colors)
    ax.set_yticks(range(len(rk)), rk.index)
    ax.invert_yaxis()
    ax.set_xlabel("Mean per-series MASE rank (lower is better)")
    ax.set_title("Overall ranking across all series")
    fig.tight_layout()
    fig.savefig(figs / "mean_ranks.png", dpi=200)
    plt.close(fig)

    print(f"Wrote summary.csv, ranks.csv, summary.md, figs/ under {results_dir}")
    return {"summary": full, "ranks": ranks}
