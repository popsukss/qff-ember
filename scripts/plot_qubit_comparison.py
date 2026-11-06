#!/usr/bin/env python3
"""Bar chart: the submitted 5-qubit model against the larger versions that add information.

All numbers are 2022 validation MAE, read from saved result tables (no simulation):
  5 qubits (submitted)                                             E11b
  16-qubit ring of slices (adds the neighbouring slices)           E12
  7 qubits (+ terrain, fuel) and 10 qubits (+ future weather)      E13
Larger circuits that only copy the same 5 inputs (E11, 8-15 qubits) are left out.
Each quantum bar uses the frozen reservoir seed (7); its thin line spans the
other reservoir seeds where they were run. Next to it are the classical
reservoir (ESN) with as many units as the quantum model has features, and
ridge regression on the same inputs.

Output: figures/qubit_comparison.png and results/qubit_comparison_val.csv
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RES, FIG = ROOT / "results", ROOT / "figures"
COLORS = {"quantum": "#0072B2", "esn": "#009E73", "raw": "#CC79A7",
          "fbp": "#D55E00", "persistence": "#333333"}
LEADS = (1, 3)


def tiled_rows():
    """5 and 8-15 qubits: qubit i reads input i mod 5 (E11b)."""
    spread = pd.concat([pd.read_csv(RES / "e11b_seed_spread.csv"),
                        pd.read_csv(RES / "e11b_seed_spread_11to15.csv")]).drop_duplicates()
    paired = pd.concat([pd.read_csv(RES / "e11b_paired_val.csv"),
                        pd.read_csv(RES / "e11b_paired_val_11to15.csv")]).drop_duplicates(
        subset=["n", "reference", "lead"])
    rows = []
    for n in sorted(spread.n.unique()):
        for h in LEADS:
            s = spread[(spread.n == n) & (spread.lead == h)]
            q = float(s.loc[s.seed == 7, "val_mae"].iloc[0])
            p = paired[(paired.n == n) & (paired.lead == h)].set_index("reference")["diff"]
            rows.append({"group": "5 inputs, copied" if n > 5 else "submitted", "qubits": int(n),
                         "label": f"{n}" if n > 5 else "5\nsubmitted", "lead": h, "quantum": q,
                         "q_min": s.val_mae.min(), "q_max": s.val_mae.max(), "n_seeds": len(s),
                         "esn": q - p[f"esn_matched{n}"], "raw": q - p["raw"]})
    return rows


def ring_rows():
    """16-qubit ring: 2 qubits per slice, all 8 slices linked (E12)."""
    r = pd.read_csv(RES / "e12_ring_val_gamma.csv")
    out = []
    for h in LEADS:
        get = lambda m: float(r[(r.model == m) & (r.lead == h)].val_mae.iloc[0])
        q = get("ring")
        out.append({"group": "neighbour slices", "qubits": 16, "label": "16\nring", "lead": h,
                    "quantum": q, "q_min": q, "q_max": q, "n_seeds": 1,
                    "esn": get("esn_ring152"), "raw": get("raw_ring")})
    return out


def driver_rows():
    """7 qubits (+ upslope, conifer) and 10 qubits (+ future wind, ISI, rain) (E13)."""
    m = pd.read_csv(RES / "e13_metrics.csv")
    m = m[m.split == "val"]
    spread = pd.read_csv(RES / "e13_seed_spread_val.csv")
    out = []
    for version, n, label in (("A", 7, "7\n+terrain\n+fuel"), ("B", 10, "10\n+future\nweather*")):
        for h in LEADS:
            get = lambda name: float(m[(m.model == name) & (m.lead == h)].mae.iloc[0])
            s = spread[(spread.version == version) & (spread.lead == h)]
            out.append({"group": "new inputs", "qubits": n, "label": label, "lead": h,
                        "quantum": get(f"qrc_{version}"), "q_min": s.val_mae.min(),
                        "q_max": s.val_mae.max(), "n_seeds": len(s),
                        "esn": get(f"esn_{version}"), "raw": get(f"raw_{version}")})
    return out


def references():
    m = pd.read_csv(RES / "metrics.csv")
    m = m[m.split == "val"]
    return {(name, h): float(m[(m.model == name) & (m.lead == h)].mae.iloc[0])
            for name in ("fbp", "persistence") for h in LEADS}


def main():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    table = pd.DataFrame(tiled_rows() + ring_rows() + driver_rows())
    table = table[table.group != "5 inputs, copied"].reset_index(drop=True)
    refs = references()
    order = table.drop_duplicates("label")[["label", "group"]].reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    x = np.arange(len(order))
    width = 0.26
    for ax, h in zip(axes, LEADS):
        d = table[table.lead == h].set_index("label").loc[order.label]
        ax.bar(x - width, d.quantum, width, color=COLORS["quantum"], label="Quantum reservoir", zorder=2)
        has_spread = (d.n_seeds > 1).to_numpy()
        ax.vlines(x[has_spread] - width, d.q_min[has_spread], d.q_max[has_spread],
                  color="#111111", lw=1.2, zorder=3, label="Quantum, range over reservoir seeds")
        ax.bar(x, d.esn, width, color=COLORS["esn"], label="Classical reservoir (ESN), same size", zorder=2)
        ax.bar(x + width, d.raw, width, color=COLORS["raw"], label="Ridge on the same inputs", zorder=2)
        span = 0.012 * (2.5 if h == 3 else 1)
        for xi, v, top in zip(x, d.quantum, d.q_max):
            ax.text(xi - width, max(v, top) + span * 0.25, f"{v:.3f}", ha="center",
                    va="bottom", fontsize=8.5, color="#0b3d63", zorder=4,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        for name, ls in (("persistence", "--"), ("fbp", ":")):
            v = refs[name, h]
            ax.axhline(v, color=COLORS[name], ls=ls, lw=1.6, zorder=1,
                       label="Persistence" if name == "persistence" else "FBP physics")
            ax.text(1.005, v, f" {'Persistence' if name == 'persistence' else 'FBP'} {v:.3f}",
                    va="center", fontsize=10, color=COLORS[name], transform=ax.get_yaxis_transform())
        low = min(d[["quantum", "esn", "raw"]].min().min(), refs["fbp", h])
        high = max(d[["quantum", "esn", "raw", "q_max"]].max().max(), refs["persistence", h])
        pad = 0.08 * (high - low)
        ax.set_ylim(low - pad, high + (4.0 if h == 1 else 2.0) * pad)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel("Average miss (km)")
        ax.set_title(f"+{24 * h} h forecast  (2022 validation fires, lower is better)", loc="left", pad=8)
        ax.grid(axis="y", color="#dddddd", zorder=0)
        # Light dividers between the four families of models.
        for i in range(1, len(order)):
            if order.group[i] != order.group[i - 1]:
                ax.axvline(i - 0.5, color="#999999", lw=1, zorder=1)
    axes[1].set_xticks(x, order.label)
    axes[1].set_xlabel("Number of qubits")
    for group, name in (("submitted", "Submitted"), ("neighbour slices", "Neighbour slices"),
                        ("new inputs", "New inputs")):
        idx = order.index[order.group == group]
        lo, hi = axes[0].get_ylim()
        axes[0].text(idx.to_numpy().mean(), hi - 0.03 * (hi - lo), name, ha="center", va="top",
                     fontsize=11, fontweight="bold", color="#333333")
    handles, labels = axes[0].get_legend_handles_labels()
    order_idx = [labels.index(k) for k in ("Quantum reservoir", "Quantum, range over reservoir seeds",
                                           "Classical reservoir (ESN), same size", "Ridge on the same inputs",
                                           "Persistence", "FBP physics")]
    handles, labels = [handles[i] for i in order_idx], [labels[i] for i in order_idx]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.08, 0.085, "* assumes a perfect weather forecast (observed regional weather; the fire's own "
             "hotspots removed).\nBars start above zero so small gaps are visible.",
             fontsize=9.5, color="#555555")
    fig.subplots_adjust(left=0.08, right=0.86, top=0.95, bottom=0.22, hspace=0.2)
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / "qubit_comparison.png", dpi=200, facecolor="white")
    table.to_csv(RES / "qubit_comparison_val.csv", index=False)
    print(table.round(4).to_string(index=False))
    print(f"wrote {FIG / 'qubit_comparison.png'}")


if __name__ == "__main__":
    main()
