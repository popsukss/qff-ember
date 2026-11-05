#!/usr/bin/env python3
"""Compare saved E6 hardware features with exact and offline FakeFez features.

Run with .venv/bin/python -I scripts/e6_hardware.py. This script only reads
locally saved hardware results; it never fetches or submits a Runtime job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember import hardware

SOURCES = ("exact", "fake_fez", "hardware", "persistence")
COLORS = {"exact": "#009E73", "fake_fez": "#D55E00",
          "hardware": "#0072B2", "persistence": "#666666"}
LABELS = {"exact": "Exact", "fake_fez": "FakeFez",
          "hardware": "Hardware", "persistence": "Persistence"}


def load_case():
    """Join targets in archive order, keeping missing targets for lead filtering."""
    with np.load(ROOT / "hardware" / "zf015_case.npz", allow_pickle=False) as case:
        windows = case["windows"]
        rows = pd.DataFrame({"uid": case["uid"], "date": pd.to_datetime(case["dates"]),
                             "sector": case["sectors"]})
    dates = np.repeat(np.arange("2023-08-10", "2023-08-21", dtype="datetime64[D]"), 8)
    if (windows.shape != (88, 5, 5) or not rows.uid.eq(242981).all()
            or not np.array_equal(rows.date.to_numpy(dtype="datetime64[D]"), dates)
            or not np.array_equal(rows.sector.to_numpy(), np.tile(np.arange(8), 11))):
        raise ValueError("case must contain 88 ZF015 windows ordered by date then sector")
    targets = pd.read_parquet(ROOT / "data" / "processed" / "sectors_2023.parquet",
                              columns=["uid", "date", "sector", "dd_km", "y1", "y3"])
    targets = targets.loc[targets.uid.eq(242981)]
    rows = rows.merge(targets, on=["uid", "date", "sector"], how="left", sort=False,
                      validate="one_to_one", indicator=True)
    if not rows["_merge"].eq("both").all() or not np.isfinite(rows.dd_km).all():
        raise ValueError("every case window must match a sector row with finite dd_km")
    return windows, rows.drop(columns="_merge")


def predict_and_score(rows, features, readout):
    predictions, scores = [], []
    feature_mae = {source: float(np.abs(values - features["exact"]).mean())
                   for source, values in features.items()}
    for lead in (1, 3):
        keep = rows[f"y{lead}"].notna().to_numpy()
        table = rows.loc[keep, ["date", "sector"]].copy()
        table["lead"] = lead
        table["y"] = rows.loc[keep, f"y{lead}"].to_numpy()
        fit = readout["readouts"][str(lead)]
        for source, values in features.items():
            table[f"yhat_{source}"] = np.maximum(
                values[keep] @ np.asarray(fit["coefficients"]) + fit["intercept"], 0)
        table["yhat_persistence"] = lead * rows.loc[keep, "dd_km"].to_numpy()
        for source in SOURCES:
            scores.append({"source": source, "lead": lead,
                           "mae": float(np.abs(table.y - table[f"yhat_{source}"]).mean()),
                           "n_rows": len(table), "feature_mae": feature_mae.get(source, np.nan)})
        predictions.append(table)
    return (pd.concat(predictions, ignore_index=True).sort_values(["date", "sector", "lead"]),
            pd.DataFrame(scores))


def plot_comparison(features, scores, custom_input):
    labels = dict(LABELS)
    if custom_input:
        labels["hardware"] = "Custom input (not verified hardware)"
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for source, marker in (("hardware", "o"), ("fake_fez", "x")):
        axes[0].scatter(features["exact"].ravel(), features[source].ravel(),
                        color=COLORS[source], marker=marker, s=12, alpha=0.4,
                        label=labels[source], rasterized=True)
    axes[0].plot([-1, 1], [-1, 1], "--", color="#333333", lw=1, label="y = x")
    axes[0].set(xlabel="Exact Z / ZZ expectation", ylabel="Measured Z / ZZ expectation",
                title="88 windows × 15 individual feature values",
                xlim=(-1.05, 1.05), ylim=(-1.05, 1.05))
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].legend(fontsize=8, loc="upper left")
    for i, source in enumerate(SOURCES):
        values = scores.loc[scores.source.eq(source)].set_index("lead").loc[[1, 3]]
        axes[1].bar(np.arange(2) + (i - 1.5) * 0.19, values.mae, width=0.19,
                    color=COLORS[source], label=labels[source])
    counts = scores.loc[scores.source.eq("exact")].set_index("lead").n_rows
    axes[1].set(xticks=[0, 1], xticklabels=[f"+24 h (n={counts[1]})", f"+72 h (n={counts[3]})"],
                ylabel="MAE (km; lower is better)", title="Frozen exact-trained Z-only read-out")
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=8)
    fig.suptitle("E6: ZF015, 10–20 August 2023", fontsize=16)
    caption = ("Custom feature input: may be a stand-in; these are not verified hardware results."
               if custom_input else "Hardware: saved ibm_fez results; FakeFez: local bundled calibration model (10,000 shots, seed 0).")
    fig.text(0.5, 0.105, caption, ha="center", fontsize=9)
    fig.text(0.5, 0.035,
             "One fire, 88 correlated sector windows; descriptive MAEs, no uncertainty intervals or generalisation claim.\n"
             "QRC predictions clipped at zero; persistence = lead × current daily advance; missing targets omitted per lead.",
             ha="center", fontsize=9, color="#444444")
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.23, top=0.87, wspace=0.28)
    path = ROOT / "figures" / "e6_hardware.png"
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def main():
    start = perf_counter()
    default_features = ROOT / "hardware" / "e6_hw_features.npy"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-features", type=Path, default=default_features)
    args = parser.parse_args()
    measured = np.load(args.hw_features, allow_pickle=False)
    if (measured.shape != (88, 15) or not np.isfinite(measured).all()
            or np.abs(measured).max() > 1 + 1e-12):
        raise ValueError("hardware features must be a finite (88, 15) array in [-1, 1]")
    windows, rows = load_case()
    readout = json.loads((ROOT / "results" / "qrc_z_readout.json").read_text())
    job = json.loads((ROOT / "hardware" / "jobs.json").read_text())[-1]
    metrics_path = ROOT / "hardware" / "e6_job_metrics.json"
    meta = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    custom_input = args.hw_features.resolve() != default_features.resolve()
    meta.update({"job_id": job["job_id"], "backend": job["backend"],
                 "shots": job["budget"]["shots"], "submitted_at": job["submitted_at"],
                 "hw_features_path": str(args.hw_features.resolve()),
                 "custom_hw_features": custom_input, "fake_fez_shots": 10000,
                 "fake_fez_seed": 0})
    reservoir = hardware.frozen_reservoir(ROOT / "results" / "frozen_config.json")
    exact = reservoir.feature_matrix(windows, mode="numpy")[:, :15]
    print("E6: sampling 88 windows locally on FakeFez (10,000 shots, seed 0).", flush=True)
    fake = hardware.run_sampler(reservoir, windows, mode="fake_fez", shots=10000, seed=0)
    features = {"exact": exact, "fake_fez": fake, "hardware": measured}
    predictions, scores = predict_and_score(rows, features, readout)
    results = ROOT / "results"
    predictions.to_csv(results / "e6_predictions.csv", index=False, date_format="%Y-%m-%d")
    scores.to_csv(results / "e6_hardware.csv", index=False)
    plot_comparison(features, scores, custom_input)
    meta["runtime_seconds"] = perf_counter() - start
    (results / "e6_meta.json").write_text(json.dumps(meta, indent=2, allow_nan=False) + "\n")
    print(scores.to_string(index=False))
    print(f"E6 complete in {meta['runtime_seconds']:.2f} s; custom input: {custom_input}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
