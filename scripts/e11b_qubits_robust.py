#!/usr/bin/env python3
"""E11b (exploratory): is the E11 register-size gain real? Validation (2022) only.

1. Seed spread: val MAE for each n over the frozen E4 reservoir seeds.
2. Paired fire-bootstrap differences on val for each n (seed 7) against
   qrc5, the matched-feature ESN and ridge on raw inputs.
Test rows are never featurized or scored here.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.baselines import ESN, raw_window_features
from ember.experiments import _windows, clip_prediction, load_frozen
from ember.metrics import fire_bootstrap_ci, mae
from ember.readout import RidgeReadout, select_lambda
from scripts.e11_qubits import chain_features


def val_errors(X, rows, h, grid):
    """Absolute val errors (row order) for a train-fit, val-selected ridge."""
    y = rows[f"y{h}"].to_numpy()
    valid = rows[f"y{h}"].notna().to_numpy()
    train = valid & rows["split"].eq("train").to_numpy()
    val = valid & rows["split"].eq("val").to_numpy()
    lam = select_lambda(X[train], y[train], X[val], y[val], grid)
    pred = clip_prediction(RidgeReadout(lam).fit(X[train], y[train]).predict(X[val]))
    return np.abs(y[val] - pred), rows.loc[val, "uid"].to_numpy()


def parallel_features(W, gamma, seed, batch, workers):
    """chain_features split across processes; identical rows, in order."""
    if workers <= 1:
        return chain_features(W, gamma, seed, batch)
    chunks = np.array_split(W, workers * 4)
    with ProcessPoolExecutor(workers) as pool:
        parts = list(pool.map(partial(chain_features, gamma=gamma, seed=seed, batch=batch), chunks))
    return np.concatenate(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[5, 8, 9, 10])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tag", default="", help="suffix for output CSV names")
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    grid, leads = config["readout"]["lambda_grid"], config["leads"]
    gamma, L, input_map = frozen["gamma"], frozen["L"], frozen["input_map"]
    seeds = args.seeds or frozen["e4_reservoir_seeds"]
    if frozen["reservoir_seed"] not in seeds:
        raise SystemExit("seeds must include the frozen reservoir seed (used for paired differences)")

    years = config["splits"]["train"] + config["splits"]["val"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                   ignore_index=True)
    W5, rows = _windows(df, input_map, L, config)
    assert not rows["split"].eq("test").any()
    print(f"train+val windows: {len(W5)}", flush=True)

    errs = {}  # (model, h) -> abs errors on val for seed 7 / references
    spread = []
    uids = {}
    for h in leads:
        errs["raw", h], uids[h] = val_errors(raw_window_features(W5), rows, h, grid)
    for n in args.qubits:
        t0 = perf_counter()
        Wn = W5[:, :, [i % 5 for i in range(n)]]
        batch = max(1, 2048 * 2 ** 10 // 2 ** max(n, 10))
        for seed in seeds:
            X = parallel_features(Wn, gamma, seed, batch, args.workers)
            for h in leads:
                e, _ = val_errors(X, rows, h, grid)
                spread.append({"n": n, "seed": seed, "lead": h, "val_mae": float(e.mean())})
                if seed == frozen["reservoir_seed"]:
                    errs[f"qrc{n}", h] = e
            if seed == frozen["reservoir_seed"]:
                esn = ESN(X.shape[1], 5, frozen["esn_seed"]).features(W5)
                for h in leads:
                    errs[f"esn_matched{n}", h], _ = val_errors(esn, rows, h, grid)
        print(f"n={n} done ({perf_counter() - t0:.0f} s)", flush=True)

    spread = pd.DataFrame(spread)
    paired = []
    for n in args.qubits:
        for ref in ("qrc5", f"esn_matched{n}", "raw"):
            if ref == f"qrc{n}":
                continue
            for h in leads:
                delta = errs[f"qrc{n}", h] - errs[ref, h]
                lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": uids[h], "abs_err": delta}),
                                           n_boot=args.n_boot, seed=0)
                paired.append({"n": n, "reference": ref, "lead": h, "diff": float(delta.mean()),
                               "lo": lo, "hi": hi})
    paired = pd.DataFrame(paired)
    results = ROOT / "results"
    spread.to_csv(results / f"e11b_seed_spread{args.tag}.csv", index=False)
    paired.to_csv(results / f"e11b_paired_val{args.tag}.csv", index=False)
    print("\nVal MAE over reservoir seeds (km):")
    print(spread.groupby(["n", "lead"]).val_mae.agg(["mean", "std", "min", "max"]).round(4).to_string())
    print("\nqrc_n (seed 7) minus reference, val, paired 95% fire-bootstrap (negative favours qrc_n):")
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
