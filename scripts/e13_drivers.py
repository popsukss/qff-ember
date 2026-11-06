#!/usr/bin/env python3
"""E13 (exploratory): one qubit per physical driver. 2022 val first, 2023 post-hoc.

Version A (7 qubits, usable today):  w_k, isi, dd_km, bui, rh, up_k, conif_k
Version B (10 qubits, perfect weather forecast): A + future regional wind toward the
sector, ISI and log rain. For +24 h the future value is day t+1; for +72 h it is the
mean (wind, ISI) or sum (rain) over t+1..t+3. Each future value is fed at every step
of the 5-day window. Qubit i reads input i (no tiling). Frozen gamma, seed 7, L = 5.

Every learned model gets the same inputs: ridge on raw inputs and a classical
reservoir (ESN) with as many units as the quantum model has features.
No setting is chosen here except the ridge penalty, on 2022. 2023 was already
scored for the locked study, so its numbers here are a second look, not a clean test.
Inputs come from scripts/e13_build_drivers.py.
"""

from __future__ import annotations

import argparse
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
from ember.experiments import clip_prediction, load_frozen, mae_growing, _mean_iou
from ember.metrics import fire_bootstrap_ci, mae, skill
from ember.readout import RidgeReadout, select_lambda
from ember.baselines import persistence
from ember.windows import MinMaxScaler01, make_windows
from scripts.e11b_qubits_robust import parallel_features

HIST = ["w_k", "isi", "dd_km", "bui", "rh"]
TERRAIN = ["up_k", "conif_k"]
FUTURE = {1: ["fw_1", "fisi_1", "fpcp_1"], 3: ["fw_3", "fisi_3", "fpcp_3"]}
NICE = {"up_k": "upslope", "conif_k": "conifer", "fw": "future wind", "fisi": "future ISI",
        "fpcp": "future rain"}


def load_table(years):
    sectors = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                        ignore_index=True)
    drivers = pd.concat([pd.read_parquet(ROOT / f"data/processed/drivers_{y}.parquet") for y in years],
                        ignore_index=True).sort_values(["uid", "sector", "day_index"])
    g = drivers.groupby(["uid", "sector"], sort=False)
    shifted = {(c, s): g[c].shift(-s) for c in ("rw_k", "risi", "rpcp") for s in (1, 2, 3)}
    drivers["fw_1"] = shifted["rw_k", 1]
    drivers["fisi_1"] = shifted["risi", 1]
    drivers["fpcp_1"] = shifted["rpcp", 1]
    drivers["fw_3"] = (shifted["rw_k", 1] + shifted["rw_k", 2] + shifted["rw_k", 3]) / 3
    drivers["fisi_3"] = (shifted["risi", 1] + shifted["risi", 2] + shifted["risi", 3]) / 3
    drivers["fpcp_3"] = shifted["rpcp", 1] + shifted["rpcp", 2] + shifted["rpcp", 3]
    keep = ["uid", "day_index", "sector"] + TERRAIN + FUTURE[1] + FUTURE[3]
    df = sectors.merge(drivers.loc[drivers.in_fire, keep], on=["uid", "day_index", "sector"],
                       how="left", validate="one_to_one")
    # The driver tables run to day n_days + 3, so every origin has its next three days.
    if df[TERRAIN + FUTURE[1] + FUTURE[3]].isna().any().any():
        raise RuntimeError("missing driver values")
    for c in ("fpcp_1", "fpcp_3"):
        df[c] = np.log1p(df[c])
    train = df.split.eq("train")
    lo, hi = df.loc[train, "up_k"].quantile([0.01, 0.99])
    df["up_k"] = df.up_k.clip(lo, hi)  # train-only bounds; slopes have a long tail
    return df


def windows(df, columns, L, config, future=()):
    data = config["data"]
    W, rows = make_windows(df, columns, L, data["forecast_from_day"], data["max_masked_in_window"])
    for c in future:  # the origin's forecast value, at every step
        j = columns.index(c)
        W[:, :, j] = W[:, -1:, j]
    train = rows["split"].eq("train").to_numpy()
    return MinMaxScaler01().fit(W[train]).transform(W), rows


def fit_predict(X, rows, h, grid):
    y = rows[f"y{h}"].to_numpy()
    valid = rows[f"y{h}"].notna().to_numpy()
    train = valid & rows.split.eq("train").to_numpy()
    val = valid & rows.split.eq("val").to_numpy()
    lam = select_lambda(X[train], y[train], X[val], y[val], grid)
    pred = clip_prediction(RidgeReadout(lam).fit(X[train], y[train]).predict(X))
    return pred, valid, lam


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    grid, leads, L, gamma = config["readout"]["lambda_grid"], config["leads"], frozen["L"], frozen["gamma"]
    seed0 = frozen["reservoir_seed"]
    seeds = args.seeds or frozen["e4_reservoir_seeds"]
    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = load_table(years)

    def qfeat(W, seed):
        n = W.shape[2]
        return parallel_features(W, gamma, seed, max(1, 2048 * 2 ** 10 // 2 ** max(n, 10)), args.workers)

    inputs = {"base5": HIST, "A7": HIST + TERRAIN}
    inputs.update({f"B10_{h}": HIST + TERRAIN + FUTURE[h] for h in leads})
    Ws, rows = {}, None
    for name, cols in inputs.items():
        fut = [c for c in cols if c.startswith("f")]
        Ws[name], r = windows(df, cols, L, config, fut)
        if rows is None:
            rows = r
        elif not rows[["uid", "sector", "day_index"]].equals(r[["uid", "sector", "day_index"]]):
            raise RuntimeError("window rows differ between input sets")
    print(f"windows: {len(rows)} (train {rows.split.eq('train').sum()}, val {rows.split.eq('val').sum()}, "
          f"test {rows.split.eq('test').sum()})", flush=True)

    def wkey(version, h):
        return {"base": "base5", "A": "A7", "B": f"B10_{h}"}[version]

    preds = {}  # (model, h) -> (pred, valid, lam)
    spread = []
    for version in ("base", "A", "B"):
        for h in leads:
            W = Ws[wkey(version, h)]
            n = W.shape[2]
            t0 = perf_counter()
            for seed in seeds:
                X = qfeat(W, seed)
                pred, valid, lam = fit_predict(X, rows, h, grid)
                val = valid & rows.split.eq("val").to_numpy()
                spread.append({"version": version, "n": n, "lead": h, "seed": seed,
                               "val_mae": mae(rows[f"y{h}"].to_numpy()[val], pred[val])})
                if seed == seed0:
                    preds[f"qrc_{version}", h] = (pred, valid, lam)
                    nfeat = X.shape[1]
            preds[f"raw_{version}", h] = fit_predict(raw_window_features(W), rows, h, grid)
            esn = ESN(nfeat, n, frozen["esn_seed"]).features(W)
            preds[f"esn_{version}", h] = fit_predict(esn, rows, h, grid)
            print(f"{version} n={n} lead={h}: {nfeat} features ({perf_counter() - t0:.0f} s)", flush=True)
    for h in leads:
        preds["persistence", h] = (clip_prediction(persistence(rows.dd_km, h)),
                                   rows[f"y{h}"].notna().to_numpy(), None)

    ablation = []
    if not args.no_ablation:
        for h in leads:
            cols = inputs[f"B10_{h}"]
            for drop in TERRAIN + FUTURE[h]:
                keep = [c for c in cols if c != drop]
                W, _ = windows(df, keep, L, config, [c for c in keep if c.startswith("f")])
                for model in ("qrc", "raw"):
                    X = qfeat(W, seed0) if model == "qrc" else raw_window_features(W)
                    pred, valid, _ = fit_predict(X, rows, h, grid)
                    val = valid & rows.split.eq("val").to_numpy()
                    full = preds[f"{model}_B", h][0]
                    y = rows[f"y{h}"].to_numpy()
                    ablation.append({"model": model, "lead": h, "dropped": NICE[drop.split("_")[0]]
                                     if drop.startswith("f") else NICE[drop],
                                     "val_mae": mae(y[val], pred[val]),
                                     "increase": mae(y[val], pred[val]) - mae(y[val], full[val])})
            print(f"ablation lead={h} done", flush=True)

    def metrics(split):
        out = []
        for (model, h), (pred, valid, lam) in preds.items():
            keep = valid & rows.split.eq(split).to_numpy()
            sel, y = rows.loc[keep], rows[f"y{h}"].to_numpy()[keep]
            err = np.abs(y - pred[keep])
            lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": sel.uid.to_numpy(), "abs_err": err}),
                                       n_boot=args.n_boot, seed=0)
            ref = mae(y, preds["persistence", h][0][keep])
            out.append({"model": model, "split": split, "lead": h, "mae": float(err.mean()),
                        "mae_lo": lo, "mae_hi": hi, "skill": skill(float(err.mean()), ref),
                        "mae_growing": mae_growing(y, pred[keep]), "iou": _mean_iou(sel, y, pred[keep]),
                        "lam": lam})
        return out

    def paired(split):
        out = []
        for version in ("A", "B"):
            for ref in ("qrc_base", f"raw_{version}", f"esn_{version}"):
                for h in leads:
                    a, valid, _ = preds[f"qrc_{version}", h]
                    keep = valid & rows.split.eq(split).to_numpy()
                    y = rows[f"y{h}"].to_numpy()[keep]
                    delta = np.abs(y - a[keep]) - np.abs(y - preds[ref, h][0][keep])
                    lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": rows.uid.to_numpy()[keep],
                                                             "abs_err": delta}), n_boot=args.n_boot, seed=0)
                    out.append({"split": split, "model": f"qrc_{version}", "reference": ref, "lead": h,
                                "diff": float(delta.mean()), "lo": lo, "hi": hi})
        return out

    results = ROOT / "results"
    val_m, val_p = pd.DataFrame(metrics("val")), pd.DataFrame(paired("val"))
    spread = pd.DataFrame(spread)
    print("\n== 2022 val MAE (km) ==")
    print(val_m.pivot_table(index="model", columns="lead", values="mae", sort=False).round(4).to_string())
    print("\nquantum seed spread, val:")
    print(spread.groupby(["version", "lead"]).val_mae.agg(["mean", "std", "min", "max"]).round(4).to_string())
    print("\nqrc minus reference, val, paired 95% (negative favours qrc):")
    print(val_p.round(4).to_string(index=False))
    if ablation:
        ablation = pd.DataFrame(ablation)
        print("\nversion B, val: MAE increase when one new input is removed (bigger = more useful):")
        print(ablation.pivot_table(index="dropped", columns=["model", "lead"], values="increase").round(4).to_string())
        ablation.to_csv(results / "e13_ablation_val.csv", index=False)

    test_m, test_p = pd.DataFrame(metrics("test")), pd.DataFrame(paired("test"))
    print("\n== 2023 second look (post-hoc) MAE (km) ==")
    print(test_m.pivot_table(index="model", columns="lead", values="mae", sort=False).round(4).to_string())
    print("\nqrc minus reference, 2023, paired 95%:")
    print(test_p.round(4).to_string(index=False))

    pd.concat([val_m, test_m]).to_csv(results / "e13_metrics.csv", index=False)
    pd.concat([val_p, test_p]).to_csv(results / "e13_paired.csv", index=False)
    spread.to_csv(results / "e13_seed_spread_val.csv", index=False)
    (results / "e13_meta.json").write_text(json.dumps({
        "status": "exploratory; ridge penalty chosen on 2022; 2023 is a post-hoc second look "
                  "(already used by the locked study)",
        "inputs": inputs, "frozen": {"L": L, "gamma": gamma, "seed": seed0},
        "future_weather": "regional (3x3 one-degree cells, all hotspots) observed weather = "
                          "perfect-forecast assumption; t+1 for +24 h, t+1..t+3 for +72 h",
        "seeds": seeds}, indent=2) + "\n")


if __name__ == "__main__":
    main()
