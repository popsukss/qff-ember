"""Validation-only selection and common evaluation for E1--E4 and E7."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ember.baselines import (ESN, fbp_fit_burn_minutes, fbp_predict,
                             mean_advance, persistence, raw_window_features)
from ember.metrics import fire_bootstrap_ci, mae, outline_iou, skill
from ember.readout import RidgeReadout, select_lambda
from ember.reservoir import ChainReservoir
from ember.windows import MinMaxScaler01, make_windows

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"
FEATURE_MODE = "Exact noiseless expectation values computed with the NumPy statevector twin (mode='numpy')."
RIDGE_MODELS = {"raw", "esn5", "esn20", "qrc", "qrc_z", "qrc_raw", "haar"}


def _config(path=CONFIG_PATH):
    return yaml.safe_load(Path(path).read_text())


def featurize(name, W, *, gamma, seed, ent_seed, esn_seed):
    """Use the accepted feature implementations, with no learned reservoir gates."""
    if name == "raw":
        return raw_window_features(W)
    if name in ("esn5", "esn20"):
        return ESN(int(name[3:]), 5, esn_seed).features(W)
    if name in ("qrc", "qrc_z", "qrc_raw", "haar"):
        reservoir = ChainReservoir(n=5, gamma=gamma, seed=seed,
                                   entangler="haar" if name == "haar" else "cz_chain",
                                   ent_seed=ent_seed)
        features = reservoir.feature_matrix(W, mode="numpy")
        if name == "qrc_z":
            return features[:, :15]
        if name == "qrc_raw":
            return np.column_stack((features, raw_window_features(W)))
        return features
    raise ValueError(f"unknown feature model {name!r}")


def drop_hotspot_days(hot, frac, seed):
    """Remove seeded UTC detection days per fire, preserving its first day."""
    if not np.isfinite(frac) or not 0 <= frac <= 1:
        raise ValueError("frac must be finite and between 0 and 1")
    if frac == 0:
        return hot
    rng = np.random.default_rng(seed)
    days = pd.to_datetime(hot["rep_date"], utc=True).dt.normalize()
    keep = np.ones(len(hot), dtype=bool)
    for positions in hot.groupby("uid", sort=True).indices.values():
        fire_days = days.iloc[positions]
        candidates = fire_days.drop_duplicates().sort_values().iloc[1:]
        count = round(frac * len(candidates))
        dropped = rng.choice(candidates.to_numpy(), size=count, replace=False)
        keep[positions] = ~fire_days.isin(dropped).to_numpy()
    return hot.loc[keep]


def clip_prediction(yhat):
    return np.maximum(np.asarray(yhat, dtype=float), 0.0)


def mae_growing(y, yhat):
    y, yhat = np.asarray(y), np.asarray(yhat)
    growing = y > 0
    return mae(y[growing], yhat[growing]) if growing.any() else float("nan")


def wind_bin(w_k, ws):
    thresholds = _config()["experiments"]["wind_bins"]
    w_k, ws = np.broadcast_arrays(np.asarray(w_k, dtype=float), np.asarray(ws, dtype=float))
    cosine = np.divide(w_k, ws, out=np.zeros_like(w_k), where=ws != 0)
    return np.where((ws != 0) & (cosine > thresholds["downwind"]), "downwind",
                    np.where((ws != 0) & (cosine < thresholds["upwind"]), "upwind", "crosswind"))


def _windows(df, input_map, L, config):
    data = config["data"]
    W, rows = make_windows(df, input_map, L, data["forecast_from_day"],
                           data["max_masked_in_window"])
    train = rows["split"].eq("train").to_numpy()
    if not train.any():
        raise ValueError("training windows are required for scaling")
    scaler = MinMaxScaler01().fit(W[train])
    return scaler.transform(W), rows


def run_e3(df, window_days, gammas, input_maps):
    """Select QRC configuration on train/val only, never accepting test rows."""
    if df["split"].eq("test").any():
        raise ValueError("E3 must not receive test rows")
    config = _config()
    records = []
    for L in window_days:
        for input_map in input_maps:
            W, rows = _windows(df, input_map, L, config)
            for gamma in gammas:
                X = featurize("qrc", W, gamma=gamma, seed=config["reservoir"]["seed"],
                              ent_seed=0, esn_seed=config["experiments"]["esn_seed"])
                record = {"L": int(L), "gamma": float(gamma), "input_map": json.dumps(list(input_map))}
                skills = []
                for h in config["leads"]:
                    valid = rows[f"y{h}"].notna().to_numpy()
                    train = valid & rows["split"].eq("train").to_numpy()
                    val = valid & rows["split"].eq("val").to_numpy()
                    y = rows[f"y{h}"].to_numpy()
                    lam = select_lambda(X[train], y[train], X[val], y[val],
                                        config["readout"]["lambda_grid"])
                    pred = clip_prediction(RidgeReadout(lam).fit(X[train], y[train]).predict(X[val]))
                    error = mae(y[val], pred)
                    ref = mae(y[val], clip_prediction(persistence(rows.loc[val, "dd_km"], h)))
                    record[f"lam_{h}"] = float(lam)
                    record[f"val_mae_{h}"] = error
                    skills.append(skill(error, ref))
                record["val_skill_mean"] = float(np.mean(skills))
                records.append(record)
    return pd.DataFrame(records)


def freeze(choice, *, config_path, out_path):
    """Bind a validation choice to the exact source configuration bytes."""
    record = dict(choice)
    record["config_sha256"] = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    record["created_utc"] = datetime.now(timezone.utc).isoformat()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def load_frozen(*, config_path, frozen_path):
    frozen_path = Path(frozen_path)
    if not frozen_path.is_file():
        raise RuntimeError(f"missing frozen configuration: {frozen_path}")
    record = json.loads(frozen_path.read_text())
    if record.get("config_sha256") != hashlib.sha256(Path(config_path).read_bytes()).hexdigest():
        raise RuntimeError("frozen configuration hash does not match config file")
    return record


def _mean_iou(rows, y, pred):
    outlines = rows[["uid", "date", "sector", "r_km"]].copy()
    outlines["truth"] = outlines["r_km"].to_numpy() + y
    outlines["prediction"] = outlines["r_km"].to_numpy() + pred
    # Never fill missing sectors with zero or average partial outlines.
    complete = outlines.groupby(["uid", "date"])["sector"].transform("nunique").eq(8)
    outlines = outlines.loc[complete].sort_values(["uid", "date", "sector"])
    truth = outlines["truth"].to_numpy().reshape(-1, 8)
    prediction = outlines["prediction"].to_numpy().reshape(-1, 8)
    return float(np.mean([outline_iou(a, b) for a, b in zip(truth, prediction)])) if len(truth) else float("nan")


def score_models(df, *, models, L, gamma, input_map, splits, n_boot,
                 seed=None, ent_seed=0, esn_seed=None, bootstrap_seed=0,
                 frozen_lambdas=None, predictions_out=None):
    """Fit only on train, select ridge penalties on val, then score requested splits.

    ``predictions_out`` optionally receives the scored rows keyed by
    (model, split, lead), allowing diagnostics to reuse the same predictions.
    ``frozen_lambdas`` maps model -> {lead: penalty} for already selected fits.
    The accepted select_lambda helper sets the penalty; all reported metrics
    (including E3) use nonnegative predictions.
    """
    models, splits = list(models), list(splits)
    unknown = set(models) - RIDGE_MODELS - {"persistence", "mean", "fbp"}
    if unknown:
        raise ValueError(f"unknown models: {sorted(unknown)}")
    config = _config()
    seed = config["reservoir"]["seed"] if seed is None else seed
    esn_seed = config["experiments"]["esn_seed"] if esn_seed is None else esn_seed
    # Do not even featurize test data during a validation-only call.
    df = df[df["split"].isin({"train", "val", *splits})]
    W, rows = _windows(df, input_map, L, config)
    features = {}
    for name in models:
        if name in RIDGE_MODELS:
            if name == "qrc_raw" and "qrc" in features:
                features[name] = np.column_stack((features["qrc"], raw_window_features(W)))
            else:
                features[name] = featurize(name, W, gamma=gamma, seed=seed,
                                          ent_seed=ent_seed, esn_seed=esn_seed)
    records = []
    for h in config["leads"]:
        y = rows[f"y{h}"].to_numpy()
        valid = rows[f"y{h}"].notna().to_numpy()
        train = valid & rows["split"].eq("train").to_numpy()
        val = valid & rows["split"].eq("val").to_numpy()
        for name in models:
            lam = float("nan")
            if name in RIDGE_MODELS:
                X = features[name]
                fixed = (frozen_lambdas or {}).get(name, {})
                lam = fixed.get(h)
                if lam is None:
                    lam = select_lambda(X[train], y[train], X[val], y[val],
                                        config["readout"]["lambda_grid"])
                readout = RidgeReadout(lam).fit(X[train], y[train])
            elif name == "fbp":
                tr = rows.loc[train]
                burn_minutes = fbp_fit_burn_minutes(tr.ros, tr.ws, tr.w_k, y[train], h)
            for split in splits:
                keep = valid & rows["split"].eq(split).to_numpy()
                selected = rows.loc[keep]
                if selected.empty:
                    continue
                if name in RIDGE_MODELS:
                    pred = readout.predict(X[keep])
                elif name == "persistence":
                    pred = persistence(selected.dd_km, h)
                elif name == "mean":
                    pred = mean_advance(y[train], len(selected))
                else:
                    pred = fbp_predict(selected.ros, selected.ws, selected.w_k, h, burn_minutes)
                pred = clip_prediction(pred)
                errors = pd.DataFrame({"uid": selected.uid.to_numpy(), "abs_err": np.abs(y[keep] - pred)})
                lo, hi = fire_bootstrap_ci(errors, n_boot=n_boot, seed=bootstrap_seed)
                error = mae(y[keep], pred)
                ref = mae(y[keep], clip_prediction(persistence(selected.dd_km, h)))
                records.append({"model": name, "split": split, "lead": h,
                                "mae": error, "mae_lo": lo, "mae_hi": hi,
                                "skill": skill(error, ref), "mae_growing": mae_growing(y[keep], pred),
                                "iou": _mean_iou(selected, y[keep], pred),
                                "n_rows": len(selected), "n_fires": selected.uid.nunique(), "lam": lam})
                if predictions_out is not None:
                    scored = selected.copy()
                    scored["yhat"] = pred
                    predictions_out[name, split, h] = scored.reset_index(drop=True)
    return pd.DataFrame(records)
