#!/usr/bin/env python3
"""E14 (exploratory): fair version of the E13 "more qubits and inputs" comparison.

E13 scored the quantum reservoir (QRC) at one reservoir seed with a gamma that E3 had
tuned, against an untuned single-seed ESN. E14 gives every learned family the same
budget and averages over seeds:

  QRC   gamma in {pi/4, pi/2, 3pi/4} (the E3 grid) x the 5 frozen E4 reservoir seeds
  ESN   input_scale in {0.3, 1.0, 3.0} at spectral radius 0.9, leak 1.0 x seeds 0-4,
        with as many units as the quantum model has features (as in E13)
  ridge on the raw window (deterministic, one config)

Inputs, windows, scaling and the ridge penalty choice (select_lambda on 2022) are
exactly as in scripts/e13_drivers.py. For each family, version and lead the config
with the lowest seed-mean 2022 validation MAE is selected. 2023 was already used by
the locked study, so its numbers are post-hoc. Paired fire-level bootstrap on 2023
compares per-row seed-averaged absolute errors of the selected configs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.baselines import ESN, raw_window_features
from ember.experiments import load_frozen
from ember.metrics import fire_bootstrap_ci
from scripts.e11b_qubits_robust import parallel_features
from scripts.e13_drivers import FUTURE, HIST, TERRAIN, fit_predict, load_table, windows

GAMMAS = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
ESN_SCALES = [0.3, 1.0, 3.0]
ESN_SEEDS = [0, 1, 2, 3, 4]
ESN_RADIUS, ESN_LEAK = 0.9, 1.0
SPLITS = {"val": "val", "test": "test"}
PAIRS = [("qrc", "raw"), ("qrc", "esn"), ("esn", "raw")]


def version_key(version, h):
    return f"B10_{h}" if version == "B10_h" else version


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--versions", nargs="+", default=["base5", "A7", "B10_h"])
    parser.add_argument("--qrc-seeds", type=int, nargs="+", default=None)
    parser.add_argument("--esn-seeds", type=int, nargs="+", default=ESN_SEEDS)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()
    t_start = perf_counter()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    grid, leads, L = config["readout"]["lambda_grid"], config["leads"], frozen["L"]
    qrc_seeds = args.qrc_seeds or frozen["e4_reservoir_seeds"]
    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = load_table(years)

    inputs = {"base5": HIST, "A7": HIST + TERRAIN}
    inputs.update({f"B10_{h}": HIST + TERRAIN + FUTURE[h] for h in leads})
    Ws, rows = {}, None
    for name, cols in inputs.items():
        Ws[name], r = windows(df, cols, L, config, [c for c in cols if c.startswith("f")])
        if rows is None:
            rows = r
        elif not rows[["uid", "sector", "day_index"]].equals(r[["uid", "sector", "day_index"]]):
            raise RuntimeError("window rows differ between input sets")
    split = rows.split.to_numpy()
    print(f"windows: {len(rows)}", flush=True)

    # Which leads each window set serves: base5 and A7 serve both, B10_h only lead h.
    jobs = {}
    for version in args.versions:
        for h in leads:
            jobs.setdefault(version_key(version, h), []).append((version, h))

    metrics = []
    errs = {}  # (family, version, h, config) -> list of per-row abs errors (one per seed)

    def record(family, version, h, cfg, seed, pred, valid, lam, n, nfeat):
        y = rows[f"y{h}"].to_numpy()
        err = np.where(valid, np.abs(y - pred), np.nan)
        errs.setdefault((family, version, h, cfg), []).append(err)
        for name, s in SPLITS.items():
            keep = valid & (split == s)
            metrics.append({"family": family, "version": version, "n_qubits": n, "lead": h,
                            "config": cfg, "seed": seed, "split": name, "n_features": nfeat,
                            "mae": float(err[keep].mean()), "lam": lam})

    def qfeat(W, gamma, seed):
        n = W.shape[2]
        return parallel_features(W, gamma, seed, max(1, 2048 * 2 ** 10 // 2 ** max(n, 10)), args.workers)

    for key, targets in jobs.items():
        W = Ws[key]
        n = W.shape[2]
        nfeat = None
        for gamma in GAMMAS:
            cfg = f"gamma={gamma / np.pi:.2f}pi"
            for seed in qrc_seeds:
                t0 = perf_counter()
                X = qfeat(W, gamma, seed)
                nfeat = X.shape[1]
                for version, h in targets:
                    pred, valid, lam = fit_predict(X, rows, h, grid)
                    record("qrc", version, h, cfg, seed, pred, valid, lam, n, nfeat)
                print(f"[{perf_counter() - t_start:6.0f} s] qrc {key} {cfg} seed {seed}: "
                      f"{perf_counter() - t0:.1f} s", flush=True)
        for scale in ESN_SCALES:
            cfg = f"input_scale={scale}"
            for seed in args.esn_seeds:
                X = ESN(nfeat, n, seed, spectral_radius=ESN_RADIUS, leak=ESN_LEAK,
                        input_scale=scale).features(W)
                if not np.isfinite(X).all():
                    raise RuntimeError(f"non-finite ESN features: {key} {cfg} seed {seed}")
                for version, h in targets:
                    pred, valid, lam = fit_predict(X, rows, h, grid)
                    record("esn", version, h, cfg, seed, pred, valid, lam, n, nfeat)
        print(f"[{perf_counter() - t_start:6.0f} s] esn {key} done", flush=True)
        X = raw_window_features(W)
        for version, h in targets:
            pred, valid, lam = fit_predict(X, rows, h, grid)
            record("raw", version, h, "raw", 0, pred, valid, lam, n, X.shape[1])

    metrics = pd.DataFrame(metrics)
    keys = ["family", "version", "lead"]
    seed_mean = (metrics[metrics.split == "val"].groupby(keys + ["config"], sort=False).mae.mean()
                 .rename("val_seed_mean").reset_index())
    best = seed_mean.loc[seed_mean.groupby(keys, sort=False).val_seed_mean.idxmin()]

    selected = []
    for b in best.itertuples():
        m = metrics[(metrics.family == b.family) & (metrics.version == b.version) &
                    (metrics.lead == b.lead) & (metrics.config == b.config)]
        out = {"family": b.family, "version": b.version, "n_qubits": int(m.n_qubits.iloc[0]),
               "lead": b.lead, "config": b.config, "n_seeds": int(m.seed.nunique()),
               "n_features": int(m.n_features.iloc[0])}
        for s in SPLITS:
            v = m[m.split == s].mae
            out.update({f"{s}_mean": v.mean(), f"{s}_min": v.min(), f"{s}_max": v.max()})
        selected.append(out)
    selected = pd.DataFrame(selected)

    paired = []
    for b in selected.itertuples():
        if b.family != "qrc":
            continue
        version, h = b.version, b.lead
        choice = selected.set_index(keys).config
        mean_err = {f: np.mean(errs[f, version, h, choice[f, version, h]], axis=0) for f in ("qrc", "esn", "raw")}
        valid = rows[f"y{h}"].notna().to_numpy()
        for s in SPLITS:
            keep = valid & (split == s)
            for a, ref in PAIRS:
                delta = mean_err[a][keep] - mean_err[ref][keep]
                lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": rows.uid.to_numpy()[keep], "abs_err": delta}),
                                           n_boot=args.n_boot, seed=0)
                paired.append({"split": s, "version": version, "lead": h, "model": a, "reference": ref,
                               "diff": float(delta.mean()), "lo": lo, "hi": hi})
    paired = pd.DataFrame(paired)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "e14_metrics.csv", index=False)
    selected.to_csv(out / "e14_selected.csv", index=False)
    paired.to_csv(out / "e14_paired.csv", index=False)
    runtime = perf_counter() - t_start
    (out / "e14_meta.json").write_text(json.dumps({
        "status": "exploratory; configs and ridge penalty chosen on 2022 val; 2023 is a post-hoc "
                  "second look (already used by the locked study)",
        "versions": {v: [inputs[version_key(v, h)] for h in leads] for v in args.versions},
        "qrc": {"gamma_grid": GAMMAS, "seeds": qrc_seeds, "L": L},
        "esn": {"input_scale_grid": ESN_SCALES, "spectral_radius": ESN_RADIUS, "leak": ESN_LEAK,
                "seeds": args.esn_seeds, "units": "as many as the quantum model has features"},
        "raw": "ridge on the flattened window, one config",
        "lambda_grid": grid,
        "selection": "per family, version and lead: config with the lowest seed-mean 2022 val MAE "
                     "(lambda chosen per fit by select_lambda on 2022 val)",
        "paired": "fire-level bootstrap of per-row seed-averaged absolute-error differences "
                  f"(selected configs), n_boot={args.n_boot}, seed=0",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_s": round(runtime, 1)}, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print("\n== selected configs: seed-mean MAE (km) [min, max] ==")
    print(selected.round(4).to_string(index=False))
    print("\n== paired diffs (negative favours model) ==")
    print(paired.round(4).to_string(index=False))
    print(f"\nruntime {runtime:.0f} s")


if __name__ == "__main__":
    main()
