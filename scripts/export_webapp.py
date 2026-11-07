#!/usr/bin/env python3
"""Export per-fire outlines and forecasts (2018-2023) for the local web app.

Uses the frozen study exactly as E1: 5-qubit reservoir (frozen gamma, seed, window,
input order and ridge penalties), ESN20, ridge on raw inputs, FBP and persistence,
all fit on 2018-21 only. 2018-21 forecasts are therefore in-sample, 2022 is the
validation year and 2023 the test year. Writes webapp/data/<year>.json and
webapp/data/summary.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.experiments import _config, load_frozen, score_models

MODELS = ["qrc", "esn20", "raw", "fbp", "persistence"]
SPLITS = ["train", "val", "test"]
OUT = ROOT / "webapp" / "data"


def r3(values):
    return [None if pd.isna(v) else round(float(v), 2) for v in values]


def main():
    config = _config()
    frozen = load_frozen(config_path=ROOT / "config.yaml", frozen_path=ROOT / "results" / "frozen_config.json")
    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                   ignore_index=True)
    preds = {}
    metrics = score_models(df, models=MODELS, L=frozen["L"], gamma=frozen["gamma"],
                           input_map=frozen["input_map"], splits=SPLITS, n_boot=10,
                           frozen_lambdas={"qrc": {int(k): v for k, v in frozen["lambdas"].items()}},
                           predictions_out=preds)
    key = ["uid", "date", "sector"]
    table = df[key + ["year", "day_index", "r_km", "dd_km", "y1", "y3", "split"]].copy()
    for (name, split, h), scored in preds.items():
        part = scored[key + ["yhat"]].rename(columns={"yhat": f"{name}_{h}"})
        col = f"{name}_{h}"
        if col in table:
            table = table.merge(part, on=key, how="left", suffixes=("", "_new"))
            table[col] = table[col].fillna(table.pop(f"{col}_new"))
        else:
            table = table.merge(part, on=key, how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"models": MODELS, "years": {}, "leads": [1, 3],
               "metrics": metrics[["model", "split", "lead", "mae", "n_rows", "n_fires"]]
               .round(4).to_dict(orient="records")}
    for year, ydf in table.groupby("year"):
        fires = []
        for uid, f in ydf.groupby("uid"):
            f = f.sort_values(["day_index", "sector"])
            days = sorted(f.day_index.unique())
            grid = f.pivot(index="day_index", columns="sector", values="r_km").reindex(days)
            if grid.shape[1] != 8:
                continue
            entry = {"uid": int(uid), "split": f.split.iloc[0], "start": str(f.date.min().date()),
                     "dates": [str(d.date()) for d in f.drop_duplicates("day_index").date],
                     "r": [r3(row) for row in grid.to_numpy()], "f": {}}
            for name in MODELS:
                for h in (1, 3):
                    col = f"{name}_{h}"
                    g = f.pivot(index="day_index", columns="sector", values=col).reindex(days)
                    if g.notna().any().any():
                        entry["f"][col] = [r3(row) for row in g.to_numpy()]
            for h in (1, 3):
                g = f.pivot(index="day_index", columns="sector", values=f"y{h}").reindex(days)
                entry["f"][f"truth_{h}"] = [r3(row) for row in g.to_numpy()]
            entry["max_r"] = round(float(np.nanmax(grid.to_numpy())), 2)
            entry["n_days"] = len(days)
            fires.append(entry)
        fires.sort(key=lambda e: -e["max_r"])
        (OUT / f"{year}.json").write_text(json.dumps({"year": int(year), "fires": fires},
                                                    separators=(",", ":")))
        summary["years"][int(year)] = {"split": fires[0]["split"] if fires else None, "n_fires": len(fires)}
        print(year, len(fires), "fires", f"{(OUT / f'{year}.json').stat().st_size / 1e6:.1f} MB", flush=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(metrics[metrics.split == "test"][["model", "lead", "mae"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
