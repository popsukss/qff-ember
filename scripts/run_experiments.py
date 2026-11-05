#!/usr/bin/env python3
"""Run validation-only E3, freeze, then E1/E7, E4 and E2 exactly once per fit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.experiments import (FEATURE_MODE, freeze, load_frozen, run_e3,
                               score_models, wind_bin)
from ember.metrics import fire_bootstrap_ci, mae, skill


def main():
    start = perf_counter()
    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    settings = config["experiments"]
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    print(FEATURE_MODE, flush=True)

    # Test tables are not loaded until E3 has finished and its hash guard passes.
    trval_years = config["splits"]["train"] + config["splits"]["val"]
    trval = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
                       for year in trval_years], ignore_index=True)
    e3 = run_e3(trval, settings["e3_window_days"], settings["e3_gamma"], settings["e3_input_maps"])
    e3.to_csv(results / "e3_val.csv", index=False)
    best = e3.loc[e3.val_skill_mean.idxmax()]
    choice = {"L": int(best.L), "gamma": float(best.gamma), "input_map": json.loads(best.input_map),
              "lambdas": {str(h): float(best[f"lam_{h}"]) for h in config["leads"]},
              "val_skill_mean": float(best.val_skill_mean), "feature_mode": FEATURE_MODE,
              "reservoir_seed": config["reservoir"]["seed"], "esn_seed": settings["esn_seed"],
              "e4_haar_seeds": settings["e4_haar_seeds"],
              "e4_reservoir_seeds": settings["e4_reservoir_seeds"],
              "haar_aggregation": "Arithmetic mean of per-seed metrics and CI endpoints over 10 entangler seeds; lam is the mean selected penalty, not an ensemble penalty.",
              "lambda_selection": "Accepted select_lambda on train/val; all reported scores use clipped nonnegative predictions."}
    frozen_path = results / "frozen_config.json"
    freeze(choice, config_path=config_path, out_path=frozen_path)
    frozen = load_frozen(config_path=config_path, frozen_path=frozen_path)
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    print(f"E3 frozen: L={frozen['L']}, gamma={frozen['gamma']}, map={frozen['input_map']}, "
          f"lambdas={frozen['lambdas']}, val skill={frozen['val_skill_mean']:.6f}", flush=True)
    test = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
                      for year in config["splits"]["test"]], ignore_index=True)
    df = pd.concat((trval, test), ignore_index=True)
    common = dict(L=frozen["L"], gamma=frozen["gamma"], input_map=frozen["input_map"],
                  n_boot=settings["bootstrap"])
    predictions = {}
    base_models = ["persistence", "mean", "fbp", "raw", "esn5", "esn20", "qrc", "qrc_raw"]
    main_scores = score_models(df, models=base_models, splits=["val", "test"],
                               frozen_lambdas={"qrc": {int(h): lam for h, lam in frozen["lambdas"].items()}},
                               predictions_out=predictions, **common)
    print("E1/E7 base models scored; starting fixed E4 seeds.", flush=True)
    haar_scores, seed_scores = [], []
    for seed in settings["e4_haar_seeds"]:
        scores = score_models(df, models=["haar"], splits=["val", "test"], ent_seed=seed, **common)
        haar_scores.append(scores)
        seed_scores.append(scores.loc[scores.split.eq("test")].assign(seed=seed))
        print(f"E4 Haar entangler seed {seed} complete.", flush=True)
    for seed in settings["e4_reservoir_seeds"]:
        if seed == config["reservoir"]["seed"]:
            scores = main_scores.loc[main_scores.model.eq("qrc") & main_scores.split.eq("test")]
        else:
            scores = score_models(df, models=["qrc"], splits=["test"], seed=seed, **common)
        seed_scores.append(scores.assign(seed=seed))
        print(f"E4 QRC reservoir seed {seed} complete.", flush=True)
    haar = pd.concat(haar_scores, ignore_index=True).groupby(["model", "split", "lead"], as_index=False).mean()
    for col in ("n_rows", "n_fires"):
        haar[col] = haar[col].astype(int)
    metrics = pd.concat((main_scores, haar), ignore_index=True)
    metrics["frozen_sha256"] = frozen_sha
    metrics.to_csv(results / "metrics.csv", index=False)
    e4 = pd.concat(seed_scores, ignore_index=True)[["model", "seed", "split", "lead", "mae", "skill"]]
    e4.to_csv(results / "e4_seeds.csv", index=False)

    wind_records = []
    for model in ("qrc", "persistence", "fbp", "esn20"):
        for h in config["leads"]:
            rows = predictions[model, "test", h]
            bins = wind_bin(rows.w_k, rows.ws)
            ref = predictions["persistence", "test", h]
            for label in ("downwind", "crosswind", "upwind"):
                keep = bins == label
                error = mae(rows.loc[keep, f"y{h}"], rows.loc[keep, "yhat"])
                ref_error = mae(ref.loc[keep, f"y{h}"], ref.loc[keep, "yhat"])
                wind_records.append({"model": model, "lead": h, "wind_bin": label, "mae": error,
                                     "skill": skill(error, ref_error), "n_rows": int(keep.sum())})
    e2 = pd.DataFrame(wind_records)
    e2.to_csv(results / "e2_wind.csv", index=False)

    # Paired fire draws quantify differences without comparing marginal CI overlap.
    comparisons = []
    for h in config["leads"]:
        qrc = predictions["qrc", "test", h]
        qerr = np.abs(qrc[f"y{h}"].to_numpy() - qrc.yhat.to_numpy())
        for model in ("persistence", "raw", "esn20"):
            other = predictions[model, "test", h]
            assert qrc[["uid", "date", "sector"]].equals(other[["uid", "date", "sector"]])
            delta = qerr - np.abs(other[f"y{h}"].to_numpy() - other.yhat.to_numpy())
            lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": qrc.uid, "abs_err": delta}),
                                       n_boot=settings["bootstrap"], seed=0)
            comparisons.append({"lead": h, "reference": model, "qrc_minus_reference_mae": float(delta.mean()),
                                "difference_lo": lo, "difference_hi": hi})
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(results / "qrc_comparisons.csv", index=False)
    elapsed = perf_counter() - start
    summary = {"runtime_seconds": elapsed, "feature_mode": FEATURE_MODE, "frozen_sha256": frozen_sha,
               "bootstrap": "1000 fire resamples, seed 0, percentile 95% CIs; pooled row-weighted MAE.",
               "iou": "Mean across complete 8-sector fire-day outlines, reconstructed radii r(t)+y and r(t)+clipped prediction.",
               "haar_aggregation": frozen["haar_aggregation"],
               "test_protocol": "Test loaded only after freeze/hash verification. Each fixed model/seed scored once; E2 and seed-7 E4 reuse E1 predictions. No selection uses test.",
               "mae_growing": "Rows with target > 0, per lead.",
               "seed_spreads": e4.groupby(["model", "lead"])[["mae", "skill"]].agg(["mean", "std", "min", "max"]).to_json()}
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    table = metrics.loc[metrics.split.eq("test"), ["model", "lead", "mae", "mae_lo", "mae_hi", "skill", "mae_growing", "iou"]]
    print("\nTest main table (km; 95% fire-bootstrap CIs):")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nQRC minus reference MAE (negative favors QRC), paired 95% fire-bootstrap CIs:")
    print(comparison.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nE4 seed spreads (sample standard deviation):")
    print(e4.groupby(["model", "lead"])[["mae", "skill"]].agg(["mean", "std", "min", "max"]).to_string())
    print("\nE2 wind diagnostics:")
    print(e2.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"\nRuntime: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
