#!/usr/bin/env python3
"""Render accepted E1/E2/E4 results and predict the frozen E8 case.

Run with .venv/bin/python -I scripts/make_figures.py. Accepted result tables
and source data are read only; outputs are figures/* and results/e8_*.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.data.hotspots import load_perimeters
from ember.data.sectors import fire_coordinates
from ember.experiments import load_frozen, score_models

FIG = ROOT / "figures"
RES = ROOT / "results"
UID = 242981
START, END = pd.Timestamp("2023-08-10"), pd.Timestamp("2023-08-20")
COLORS = {"qrc": "#0072B2", "fbp": "#D55E00", "persistence": "#333333",
          "esn20": "#009E73", "raw": "#CC79A7", "haar": "#8064A2"}
LABELS = {"persistence": "Persistence", "mean": "Mean advance", "fbp": "FBP physics",
          "raw": "Raw ridge", "esn5": "ESN (5)", "esn20": "ESN (20)",
          "qrc": "QRC", "qrc_raw": "QRC + raw", "haar": "Haar (10-seed mean)"}


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, facecolor="white")
    plt.close(fig)


def e1(metrics):
    models = list(LABELS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, h in zip(axes, (1, 3)):
        rows = metrics.loc[metrics.lead.eq(h)].set_index("model").loc[models]
        ax.axvline(rows.loc["persistence", "mae"], color=COLORS["persistence"],
                   ls="--", lw=1.5, label="Persistence MAE")
        for i, (model, row) in enumerate(rows.iterrows()):
            color = COLORS.get(model, "#888888")
            ax.barh(i, row.mae, color=color, alpha=1 if model == "qrc" else 0.65,
                    height=0.65, zorder=2)
            ax.errorbar(row.mae, i, xerr=[[row.mae - row.mae_lo], [row.mae_hi - row.mae]],
                        fmt="none", ecolor="#222222", capsize=4, zorder=3)
        ax.set(yticks=range(len(models)), yticklabels=[LABELS[m] for m in models],
               xlabel="Test MAE (km; lower is better)", title=f"+{24 * h} h", xlim=(0, None))
        ax.grid(axis="x", alpha=0.2, zorder=0)
        ax.legend(loc="lower right", fontsize=11)
        for label, model in zip(ax.get_yticklabels(), models):
            if model == "qrc":
                label.set_color(COLORS["qrc"])
                label.set_fontweight("bold")
    axes[0].invert_yaxis()
    fig.suptitle("Test forecast error with 95% fire-bootstrap intervals", fontsize=18)
    fig.text(0.5, 0.065, "FBP physics has the lowest MAE at both leads; QRC improves on persistence only at +72 h.",
             ha="center", fontsize=12)
    fig.text(0.5, 0.025, "Haar: mean of 10 per-seed MAEs and CI endpoints (not an ensemble confidence interval).",
             ha="center", fontsize=11, color="#444444")
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.87, wspace=0.14)
    save(fig, "e1_test_mae.png")


def e2():
    table = pd.read_csv(RES / "e2_wind.csv")
    if "split" in table:
        table = table.loc[table.split.eq("test")]
    bins = ["downwind", "crosswind", "upwind"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, h in zip(axes, (1, 3)):
        for model, marker, ls in [("qrc", "o", "-"), ("persistence", "s", "--"),
                                  ("fbp", "^", "-.") , ("esn20", "D", ":")]:
            values = table.loc[table.lead.eq(h) & table.model.eq(model)].set_index("wind_bin").loc[bins]
            ax.plot(range(3), values.mae, marker=marker, ls=ls, color=COLORS[model],
                    lw=2.5 if model == "qrc" else 2, ms=8, label=LABELS[model])
        ax.set(xticks=range(3), xticklabels=[b.capitalize() for b in bins],
               ylabel="Test MAE (km; lower is better)", title=f"+{24 * h} h", ylim=(0, None))
        ax.grid(axis="y", alpha=0.2)
    axes[1].legend(loc="upper right", fontsize=11)
    fig.suptitle("Test forecast error by wind direction", fontsize=18)
    fig.text(0.5, 0.03, "Wind bins use the sector wind component / wind speed; calm conditions enter the crosswind bin.",
             ha="center", fontsize=11)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.16, top=0.83, wspace=0.23)
    save(fig, "e2_wind.png")


def e4(metrics):
    seeds = pd.read_csv(RES / "e4_seeds.csv")
    seeds = seeds.loc[seeds.split.eq("test")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, h in zip(axes, (1, 3)):
        positions, labels = [], []
        offset = 0
        for model in ("haar", "qrc"):
            rows = seeds.loc[seeds.lead.eq(h) & seeds.model.eq(model)].sort_values("seed")
            x = np.arange(len(rows)) + offset
            ax.scatter(x, rows.mae, s=70, marker="o" if model == "haar" else "D",
                       color=COLORS[model], zorder=3,
                       label="Haar random circuits (10)" if model == "haar" else "QRC reservoir seeds (5)")
            positions.extend(x)
            labels.extend(rows.seed.astype(str))
            offset += len(rows) + 1
        for model, ls in (("raw", "--"), ("esn20", ":")):
            ref = metrics.loc[metrics.lead.eq(h) & metrics.model.eq(model), "mae"].item()
            ax.axhline(ref, color=COLORS[model], ls=ls, lw=2, label=f"E1 {LABELS[model]}")
        ax.axvline(10, color="#BBBBBB", lw=1)
        ax.set(xticks=positions, xticklabels=labels, xlabel="Seed (Haar entangler  |  QRC reservoir)",
               ylabel="Test MAE (km; lower is better)", title=f"+{24 * h} h")
        ax.grid(axis="y", alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.025), fontsize=11)
    fig.suptitle("Variation across fixed circuit and reservoir seeds", fontsize=18)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.83, bottom=0.27, wspace=0.23)
    save(fig, "e4_seeds.png")


def local_xy(lon, lat, lat0, lon0):
    """Exactly the local km frame used by ember.data.sectors.fire_coordinates."""
    return ((np.asarray(lon) - lon0) * 111.320 * np.cos(np.deg2rad(lat0)),
            (np.asarray(lat) - lat0) * 110.574)


def e8(config, frozen):
    # The hash-checked choice is loaded before test data. score_models fits its
    # scaler/readouts/FBP calibration on train only; frozen QRC lambdas bypass
    # selection. Keep the complete training/window context exactly as in E1.
    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
                    for year in years], ignore_index=True)
    predictions = {}
    print("E8: scoring frozen QRC, FBP and persistence on test (train-only fits).", flush=True)
    scores = score_models(
        df, models=["qrc", "fbp", "persistence"], splits=["test"],
        L=frozen["L"], gamma=frozen["gamma"], input_map=frozen["input_map"],
        seed=frozen["reservoir_seed"], esn_seed=frozen["esn_seed"],
        n_boot=config["experiments"]["bootstrap"],
        frozen_lambdas={"qrc": {int(h): lam for h, lam in frozen["lambdas"].items()}},
        predictions_out=predictions)
    accepted = pd.read_csv(RES / "metrics.csv")
    for row in scores.itertuples():
        ref = accepted.loc[accepted.model.eq(row.model) & accepted.split.eq("test") &
                           accepted.lead.eq(row.lead), "mae"].item()
        np.testing.assert_allclose(row.mae, ref, rtol=1e-9, atol=1e-10)

    case = df.loc[df.uid.eq(UID)].copy()
    targets = case[["date", "sector", "r_km"]].rename(columns={"date": "target_date", "r_km": "r_obs"})
    tables = []
    for h in (1, 3):
        rows = predictions["qrc", "test", h]
        table = rows.loc[rows.uid.eq(UID) & rows.date.between(START, END),
                         ["date", "sector", "r_km", "yhat"]].copy()
        table = table.rename(columns={"r_km": "r_now", "yhat": "yhat_qrc"})
        for model in ("fbp", "persistence"):
            other = predictions[model, "test", h]
            table = table.merge(other.loc[other.uid.eq(UID), ["date", "sector", "yhat"]].rename(
                columns={"yhat": f"yhat_{model}"}), on=["date", "sector"], validate="one_to_one")
        table["lead"] = h
        table["target_date"] = table.date + pd.Timedelta(days=h)
        table = table.merge(targets, on=["target_date", "sector"], validate="one_to_one")
        for model in ("qrc", "fbp", "persistence"):
            table[f"r_pred_{model}"] = table.r_now + table[f"yhat_{model}"].clip(lower=0)
        tables.append(table)
    table = pd.concat(tables, ignore_index=True).sort_values(["date", "sector", "lead"])
    expected = len(pd.date_range(START, END)) * 8 * 2
    if len(table) != expected or table.isna().any().any():
        raise ValueError("E8 requires complete observed targets and predictions for every requested origin")
    columns = ["date", "sector", "lead", "target_date", "r_obs", "r_now",
               "r_pred_qrc", "r_pred_fbp", "r_pred_persistence",
               "yhat_qrc", "yhat_fbp", "yhat_persistence"]
    table[columns].to_csv(RES / "e8_zf015.csv", index=False, date_format="%Y-%m-%d")

    fire = pd.read_parquet(ROOT / "data/processed/hotspots_2023.parquet", filters=[("uid", "==", UID)])
    _, xy, lat0, lon0 = fire_coordinates(fire, isolation_km=config["data"]["isolation_km"],
                                        connect_km=config["data"]["connect_km"])
    town_lat, town_lon = 62.4540, -114.3718
    tx, ty = local_xy(town_lon, town_lat, lat0, lon0)
    distance = float(np.hypot(tx, ty))
    bearing = float(np.rad2deg(np.arctan2(tx, ty)) % 360)
    sector = int(np.rint(bearing / 45.0)) % 8
    eta = {"town": "Yellowknife", "town_lat": town_lat, "town_lon": town_lon,
           "distance_km": distance, "sector": sector, "bearing_degrees": bearing,
           "fire_uid": UID, "fire_center_lat": lat0, "fire_center_lon": lon0,
           "distance_method": "Euclidean distance in the fire_coordinates local km frame",
           "eta_method": "(distance_km - r_now) / clipped lead-1 advance; null if advance <= 0. Assumes constant radial advance; not an arrival forecast.",
           "r_obs_definition": "Observed radius in the town sector on the origin date (not the lead-1 target date)",
           "frozen_sha256": hashlib.sha256((RES / "frozen_config.json").read_bytes()).hexdigest(),
           "frozen_qrc": {k: frozen[k] for k in ("L", "gamma", "input_map", "lambdas", "reservoir_seed")},
           "fit_protocol": "score_models: fit on train only; use frozen validation-selected QRC lambdas; predict test only",
           "per_date": []}
    for row in table.loc[table.lead.eq(1) & table.sector.eq(sector)].itertuples():
        record = {"date": row.date.strftime("%Y-%m-%d"), "r_obs": float(row.r_now)}
        for model in ("qrc", "fbp"):
            yhat = float(getattr(row, f"yhat_{model}"))
            record[f"eta_days_{model}"] = float((distance - row.r_now) / yhat) if yhat > 0 else None
        eta["per_date"].append(record)
    (RES / "e8_eta.json").write_text(json.dumps(eta, indent=2, allow_nan=False) + "\n")
    e8_map(case, table, fire, xy, lat0, lon0, tx, ty, distance)
    print(json.dumps(eta, indent=2), flush=True)


def e8_map(case, predictions, fire, xy, lat0, lon0, tx, ty, distance):
    fig, ax = plt.subplots(figsize=(13, 9))
    perimeters = load_perimeters(ROOT / "data/raw/cwfis_perimeters/2023_perimeters.zip")
    official = perimeters.loc[perimeters.uid.eq(UID)].sort_values("lastdate").iloc[-1]
    boundary = official.geometry.boundary
    lines = list(boundary.geoms) if hasattr(boundary, "geoms") else [boundary]
    for i, line in enumerate(lines):
        lon, lat = line.xy
        ax.plot(*local_xy(lon, lat, lat0, lon0), color="#777777", lw=1.2,
                label="Official final perimeter (season end)" if i == 0 else None, zorder=1)
    keep = fire.rep_date.dt.tz_convert("UTC").dt.tz_localize(None).lt(END + pd.Timedelta(days=1))
    ax.scatter(xy[keep, 0], xy[keep, 1], s=4, c="#999999", alpha=0.18, linewidths=0,
               label="Hotspot detections through 20 Aug", zorder=2, rasterized=True)
    theta = np.deg2rad(np.arange(8) * 45)

    def outline(radii, **style):
        x, y = radii * np.sin(theta), radii * np.cos(theta)
        ax.plot(np.r_[x, x[0]], np.r_[y, y[0]], **style)

    for day, color, ls in (("2023-08-10", "#CC79A7", ":"), ("2023-08-20", "#222222", "-")):
        radii = case.loc[case.date.eq(day)].set_index("sector").loc[range(8), "r_km"].to_numpy()
        outline(radii, color=color, ls=ls, lw=2.5, marker="o", ms=4,
                label=f"Observed 8-sector outline: {day[5:]}", zorder=4)
    forecast = predictions.loc[predictions.date.eq("2023-08-17") & predictions.lead.eq(3)]
    forecast = forecast.set_index("sector").loc[range(8)]
    for model, ls in (("qrc", "--"), ("fbp", "-.")):
        outline(forecast[f"r_pred_{model}"].to_numpy(), color=COLORS[model], ls=ls, lw=2.4,
                label=f"{LABELS[model]}: 17 Aug +72 h", zorder=5)
    ax.scatter([0], [0], marker="+", c="black", s=100, label="Fire centre", zorder=6)
    ax.plot([0, tx], [0, ty], color="#555555", lw=1, ls=":", zorder=2)
    ax.scatter([tx], [ty], marker="*", s=210, c="#E69F00", edgecolors="black",
               label=f"Yellowknife ({distance:.1f} km from centre)", zorder=7)
    ax.annotate("Yellowknife", (tx, ty), xytext=(8, -18), textcoords="offset points", fontsize=12)
    ax.set(xlabel="East of fire centre (km)", ylabel="North of fire centre (km)",
           title="ZF015: reconstructed spread and 17 Aug forecasts for 20 Aug")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.18)
    ax.margins(0.12)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=11)
    fig.text(0.07, 0.035, "Observed outlines use dated hotspots; the official final perimeter is a season-end spatial reference.\n"
             "All layers use the same fire-centred local km frame. Forecasts add clipped nonnegative advances to the origin radii.",
             fontsize=10, color="#444444")
    fig.subplots_adjust(left=0.08, right=0.66, bottom=0.14, top=0.9)
    save(fig, "e8_zf015_map.png")


def main():
    start = perf_counter()
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.spines.top": False,
                         "axes.spines.right": False, "legend.frameon": False})
    FIG.mkdir(exist_ok=True)
    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=RES / "frozen_config.json")
    metrics = pd.read_csv(RES / "metrics.csv")
    metrics = metrics.loc[metrics.split.eq("test")]
    e1(metrics)
    e2()
    e4(metrics)
    e8(config, frozen)
    print(f"Runtime: {perf_counter() - start:.3f} seconds", flush=True)


if __name__ == "__main__":
    main()
