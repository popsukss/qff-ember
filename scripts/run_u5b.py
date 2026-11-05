#!/usr/bin/env python3
"""Rebuild the frozen hardware case, Z read-out, E5 shots and E9 sparsity."""

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

from ember import hardware
from ember.data.sectors import build_sectors
from ember.experiments import (clip_prediction, drop_hotspot_days, featurize,
                               load_frozen, score_models)
from ember.metrics import mae
from ember.readout import RidgeReadout, select_lambda
from ember.reservoir import feature_names
from ember.windows import MinMaxScaler01, make_windows


def prepare_case_and_readout(df, config, frozen, results):
    data = config["data"]
    W, rows = make_windows(df, frozen["input_map"], frozen["L"],
                           data["forecast_from_day"], data["max_masked_in_window"])
    scaler = MinMaxScaler01().fit(W[rows.split.eq("train").to_numpy()])
    W = scaler.transform(W)
    case_mask = rows.uid.eq(242981) & rows.date.between("2023-08-10", "2023-08-20")
    case = rows.loc[case_mask].sort_values(["date", "sector"])
    if len(case) != 88:
        raise ValueError(f"expected 88 case windows, found {len(case)}")
    np.testing.assert_array_equal(case.sector.to_numpy(), np.tile(np.arange(8), 11))
    np.testing.assert_array_equal(case.date.to_numpy(dtype="datetime64[D]"),
                                  np.repeat(np.arange("2023-08-10", "2023-08-21",
                                                      dtype="datetime64[D]"), 8))
    archive = ROOT / "hardware" / "zf015_case.npz"
    archive.parent.mkdir(exist_ok=True)
    np.savez_compressed(archive, windows=W[case.index.to_numpy()], uid=case.uid.to_numpy(),
                        dates=case.date.to_numpy(dtype="datetime64[D]"),
                        sectors=case.sector.to_numpy())
    with np.load(archive, allow_pickle=False) as saved:
        np.testing.assert_array_equal(saved["windows"], W[case.index.to_numpy()])
    X = featurize("qrc_z", W, gamma=frozen["gamma"], seed=frozen["reservoir_seed"],
                  ent_seed=0, esn_seed=frozen["esn_seed"])
    fits, readouts = {}, {}
    for h in config["leads"]:
        y = rows[f"y{h}"].to_numpy()
        valid = rows[f"y{h}"].notna().to_numpy()
        train = valid & rows.split.eq("train").to_numpy()
        val = valid & rows.split.eq("val").to_numpy()
        lam = select_lambda(X[train], y[train], X[val], y[val],
                            config["readout"]["lambda_grid"])
        fit = RidgeReadout(lam).fit(X[train], y[train])
        fits[h] = fit
        readouts[str(h)] = {"lambda": float(lam), "intercept": float(fit.intercept_),
                            "coefficients": fit.coef_.tolist(),
                            "n_train": int(train.sum()), "n_val": int(val.sum())}
    record = {"model": "qrc_z", "feature_names": feature_names(5)[:15],
              "feature_mode": "numpy", "L": frozen["L"], "gamma": frozen["gamma"],
              "reservoir_seed": frozen["reservoir_seed"], "input_map": frozen["input_map"],
              "frozen_sha256": hashlib.sha256((results / "frozen_config.json").read_bytes()).hexdigest(),
              "scaler": {"lo": scaler.lo.tolist(), "hi": scaler.hi.tolist()},
              "readouts": readouts, "prediction_clip_min": 0.0,
              "lambda_selection": "Train-only ridge fit; minimum un-clipped validation MAE using accepted select_lambda."}
    (results / "qrc_z_readout.json").write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return W, rows, X, case, fits, record


def run_e5(W, rows, X, case, fits, config, frozen_path):
    # Sample window rows, not fires, without replacement; exclude the 88 case rows.
    pool = rows.index[rows.split.eq("test") & rows.year.eq(2023)
                      & ~rows.index.isin(case.index)].to_numpy()
    sample = np.random.default_rng(0).choice(pool, size=2000, replace=False)
    selected = np.concatenate((case.index.to_numpy(), sample))
    reservoir = hardware.frozen_reservoir(frozen_path)
    eval_rows, eval_W, exact = rows.loc[selected], W[selected], X[selected]
    records = []
    for shots in (100, 1000, 2000, 10000):
        for noise in ("ideal", "fake_fez"):
            start = perf_counter()
            if noise == "ideal":
                features = reservoir.feature_matrix(eval_W, mode="sampled", shots=shots, seed=0)
            else:
                features = hardware.run_sampler(reservoir, eval_W, mode="fake_fez", shots=shots, seed=0)
            for h in config["leads"]:
                valid = eval_rows[f"y{h}"].notna().to_numpy()
                y = eval_rows.loc[valid, f"y{h}"].to_numpy()
                records.append({"shots": shots, "noise": noise, "lead": h,
                                "mae": mae(y, clip_prediction(fits[h].predict(features[valid]))),
                                "mae_exact": mae(y, clip_prediction(fits[h].predict(exact[valid]))),
                                "n_rows": int(valid.sum())})
            print(f"E5 shots={shots} noise={noise}: {perf_counter() - start:.2f} s", flush=True)
    return pd.DataFrame(records), rows.loc[sample, ["uid", "date", "sector"]]


def run_e9(years, config, frozen, baseline):
    records, checks = [], []
    data = config["data"]
    for frac in (0.0, 0.15, 0.30, 0.45):
        start = perf_counter()
        tables = []
        for year in years:
            hot = pd.read_parquet(ROOT / f"data/processed/hotspots_{year}.parquet")
            sparse = drop_hotspot_days(hot, frac=frac, seed=0)
            table = build_sectors(sparse, isolation_km=data["isolation_km"],
                                  connect_km=data["connect_km"], max_advance_km=data["max_advance_km"])
            if frac == 0:
                accepted = pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
                # Parquet reloads split as StringDtype on pandas 3; the builder
                # uses the schema's object dtype. Compare all values exactly.
                pd.testing.assert_frame_equal(table, accepted, check_exact=True, check_dtype=False)
            tables.append(table)
            print(f"E9 drop={frac:.2f} year={year}: {len(sparse)}/{len(hot)} detections, "
                  f"{len(table)} sector rows", flush=True)
        scores = score_models(pd.concat(tables, ignore_index=True),
                              models=["qrc", "esn20", "persistence", "fbp"],
                              L=frozen["L"], gamma=frozen["gamma"], input_map=frozen["input_map"],
                              seed=frozen["reservoir_seed"], esn_seed=frozen["esn_seed"],
                              splits=["test"], n_boot=config["experiments"]["bootstrap"],
                              frozen_lambdas={"qrc": {int(h): lam for h, lam in frozen["lambdas"].items()}})
        if frac == 0:
            for row in scores.itertuples():
                ref = baseline.loc[baseline.model.eq(row.model) & baseline.split.eq("test")
                                   & baseline.lead.eq(row.lead), "mae"].item()
                delta = row.mae - ref
                checks.append({"model": row.model, "lead": row.lead, "mae_delta": delta})
                if abs(delta) > 1e-9:
                    raise RuntimeError(f"drop 0 differs from E1: {row.model}, lead={row.lead}, "
                                       f"rebuilt={row.mae!r}, E1={ref!r}, delta={delta!r}")
        scores["drop_frac"] = frac
        records.append(scores[["drop_frac", "model", "lead", "mae", "skill", "n_rows", "n_fires"]])
        print(f"E9 drop={frac:.2f} complete: {perf_counter() - start:.2f} s", flush=True)
    return pd.concat(records, ignore_index=True), pd.DataFrame(checks)


def markdown_table(df):
    # Avoid making report generation depend on the optional tabulate package.
    def cell(value):
        return f"{value:.9f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(df.columns) + " |", "| " + " | ".join(["---"] * len(df.columns)) + " |"]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in df.itertuples(index=False, name=None))
    return "\n".join(lines)


def main():
    start = perf_counter()
    config_path, results = ROOT / "config.yaml", ROOT / "results"
    config = yaml.safe_load(config_path.read_text())
    frozen_path = results / "frozen_config.json"
    frozen = load_frozen(config_path=config_path, frozen_path=frozen_path)
    years = sorted({year for split in config["splits"].values() for year in split})
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
                    for year in years], ignore_index=True)
    W, rows, X, case, fits, readout = prepare_case_and_readout(df, config, frozen, results)
    print(f"Case archive: {W[case.index.to_numpy()].shape}; qrc_z lambdas: "
          f"{ {h: fit.lam for h, fit in fits.items()} }", flush=True)
    e9, checks = run_e9(years, config, frozen, pd.read_csv(results / "metrics.csv"))
    e9.to_csv(results / "e9_sparse.csv", index=False)
    e5, sample = run_e5(W, rows, X, case, fits, config, frozen_path)
    e5.to_csv(results / "e5_shots.csv", index=False)
    elapsed = perf_counter() - start
    case_valid = {h: int(case[f"y{h}"].notna().sum()) for h in config["leads"]}
    report = f"""# U5b: sampled Z read-out and sparse detection days

Reproduce from the project root with `.venv/bin/python -I scripts/run_u5b.py`.
Runtime: {elapsed:.3f} seconds. All simulation is local; no Runtime service is initialized.

## Files and frozen configuration

Code: `ember/experiments.py`, `ember/hardware.py`, `scripts/hardware_pilot.py`, `scripts/run_u5b.py`.
Outputs: `hardware/zf015_case.npz`, `results/qrc_z_readout.json`, `results/e5_shots.csv`, `results/e9_sparse.csv`, this report.
Hardware documentation: `hardware/README.md`.
Frozen SHA-256: `{readout['frozen_sha256']}`. Config hash was verified before computation.
L={frozen['L']}, gamma={frozen['gamma']}, reservoir seed={frozen['reservoir_seed']},
input map={frozen['input_map']}. Config and accepted E1 results are unchanged.

## Z read-out and case archive

The first 15 exact NumPy features are Z0..Z4 and lexicographic ZZ pairs.
Each lead fits only valid training targets and selects lambda on validation using the accepted
`select_lambda` helper (unclipped validation predictions). Final predictions are clipped at zero.
Selected lambdas: { {h: fit.lam for h, fit in fits.items()} }.
The JSON contains each intercept and all 15 coefficients, the input map, feature order and training-only scaler extrema.
Scaler lo={readout['scaler']['lo']}; hi={readout['scaler']['hi']}.

The case is rebuilt from all accepted sector tables with frozen L/input map and config
forecast/masking settings. Shape=(88, {frozen['L']}, 5); uid=242981; dates=2023-08-10..20 inclusive;
order=date then sectors 0..7. Dates, sectors and a lossless archive reload were checked.
Windows are scaled to [0,1] with extrema fitted on training windows only.
Case rows with valid targets by lead: {case_valid}.

## E5: shots and bundled FakeFez noise

MAE is in km. The fixed evaluation set is the 88 case windows followed by 2000 other test-2023
window rows sampled without replacement with `np.random.default_rng(0)` from the make_windows order.
The case rows are excluded from the sampling pool. Missing targets are excluded per lead;
`n_rows` records the evaluated count. Exact MAE uses identical rows and the same fitted read-out.
Ideal samples use `ChainReservoir.feature_matrix(mode='sampled')`; noisy samples use
`ember.hardware.run_sampler(mode='fake_fez')`, bundled FakeFez calibration and no mitigation.
Every shot/noise call uses simulator and transpiler seed 0. Reservoir seed remains frozen at 7.
This is one seeded realization per setting; the curve need not decrease monotonically.

{markdown_table(e5)}

## E9: sparse detection days

For each year and fraction, seed 0 drives removal of round(frac*(n_days-1)) UTC detection days
per uid (sorted uid and date order), always preserving the first day. All detections on selected
days are removed. Sectors, weather carry-forward, targets, eligible windows and training-only
scaler are rebuilt. This experiment scores each sparse reconstruction's targets; row populations
and targets can change across fractions, so MAEs are not paired errors on a fixed truth set.
The existing score_models pipeline fits train only, selects ESN lambda on validation, uses
the frozen QRC lambdas and refits FBP burn minutes on train. Frozen L/gamma/input map and seeds
are used throughout. Skill uses persistence on the same sparse rows.

{markdown_table(e9)}

Drop 0 sector values matched all six accepted tables exactly (ignoring the Parquet
round-trip change from object to pandas StringDtype for split). Every test MAE matched E1
within 1e-9; maximum absolute delta={checks.mae_delta.abs().max():.17g} km.

```csv
{checks.to_csv(index=False).rstrip()}
```

## Fixed E5 random sample

The following 2000 uid/date/sector rows are in draw order; sectors identify individual windows.

```csv
{sample.to_csv(index=False, date_format='%Y-%m-%d').rstrip()}
```
"""
    (results / "U5b-report.md").write_text(report)
    print("\nE5 (km):\n" + e5.to_string(index=False), flush=True)
    print("\nE9 (km):\n" + e9.to_string(index=False), flush=True)
    print(f"\nDrop 0 maximum absolute E1 delta: {checks.mae_delta.abs().max():.17g}")
    print(f"Runtime: {elapsed:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
