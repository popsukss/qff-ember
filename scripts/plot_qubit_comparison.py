#!/usr/bin/env python3
"""Bar chart: the submitted 5-qubit model against the larger versions that add information.

All numbers are 2022 validation MAE, read from saved result tables (no simulation):
  5 qubits (submitted), 7 qubits (+ terrain, fuel), 10 qubits (+ future weather)   E14
  16-qubit ring of slices (adds the neighbouring slices)                           E12
E14 gives the quantum reservoir and the classical reservoir (ESN) the same tuning
budget (3 configs each, chosen by seed-mean 2022 MAE) and averages 5 seeds; the
bars are seed means and the thin lines span the seeds. Ridge on the same inputs is
deterministic. The E12 ring is one seed, and its gamma was picked on the same 2022
fires it is scored on, so it is marked optimistic.
Larger circuits that only copy the same 5 inputs (E11, 8-15 qubits) are left out.

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


def ring_rows():
    """16-qubit ring: 2 qubits per slice, all 8 slices linked (E12). One seed; gamma picked on val."""
    r = pd.read_csv(RES / "e12_ring_val_gamma.csv")
    out = []
    for h in LEADS:
        get = lambda m: float(r[(r.model == m) & (r.lead == h)].val_mae.iloc[0])
        q, e = get("ring"), get("esn_ring152")
        out.append({"group": "neighbour slices", "qubits": 16, "label": "16 qubits\n(ring)",
                    "lead": h, "quantum": q, "q_min": q, "q_max": q, "q_seeds": 1, "q_config": "gamma=0.25pi",
                    "esn": e, "e_min": e, "e_max": e, "e_seeds": 1, "e_config": "input_scale=1.0",
                    "raw": get("raw_ring"), "source": "E12"})
    return out


def e14_rows():
    """5 qubits (submitted), 7 qubits (+ upslope, conifer), 10 qubits (+ future wind, ISI, rain) (E14)."""
    sel = pd.read_csv(RES / "e14_selected.csv").set_index(["family", "version", "lead"])
    out = []
    for version, n, group, label in (("base5", 5, "submitted", "5 qubits\n(base)"),
                                     ("A7", 7, "new inputs", "7 qubits\n(+terrain +fuel)"),
                                     ("B10_h", 10, "new inputs", "10 qubits\n(+future weather*)")):
        for h in LEADS:
            q, e, r = (sel.loc[f, version, h] for f in ("qrc", "esn", "raw"))
            out.append({"group": group, "qubits": n, "label": label, "lead": h,
                        "quantum": q.val_mean, "q_min": q.val_min, "q_max": q.val_max,
                        "q_seeds": int(q.n_seeds), "q_config": q.config,
                        "esn": e.val_mean, "e_min": e.val_min, "e_max": e.val_max,
                        "e_seeds": int(e.n_seeds), "e_config": e.config, "raw": r.val_mean, "source": "E14"})
    return out


def references():
    m = pd.read_csv(RES / "metrics.csv")
    m = m[m.split == "val"]
    return {(name, h): float(m[(m.model == name) & (m.lead == h)].mae.iloc[0])
            for name in ("fbp", "persistence") for h in LEADS}


def main():
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    rows = e14_rows()
    table = pd.DataFrame(rows[:2] + ring_rows() + rows[2:])
    refs = references()
    order = table.drop_duplicates("label")[["label", "group"]].reset_index(drop=True)

    fig, axes = plt.subplots(2, 1, figsize=(11, 10), sharex=True)
    x = np.arange(len(order))
    width = 0.26
    for ax, h in zip(axes, LEADS):
        d = table[table.lead == h].set_index("label").loc[order.label]
        ax.bar(x - width, d.quantum, width, color=COLORS["quantum"], label="Quantum reservoir", zorder=2)
        has_spread = (d.q_seeds > 1).to_numpy()
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
        ax.set_ylim(low - pad, high + 2.0 * pad)
        ax.set_xlim(-0.6, len(order) - 0.4)
        ax.set_ylabel("Average miss (km)")
        ax.set_title(f"+{24 * h} h forecast  (2022 validation fires, lower is better)", loc="left", pad=8)
        ax.grid(axis="y", color="#dddddd", zorder=0)
        # Light dividers between the four families of models.
        for i in range(1, len(order)):
            if order.group[i] != order.group[i - 1]:
                ax.axvline(i - 0.5, color="#999999", lw=1, zorder=1)
    axes[1].set_xticks(x, order.label)
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
