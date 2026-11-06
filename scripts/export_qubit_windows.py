#!/usr/bin/env python3
"""Save the scaled 7- and 10-qubit input windows the walkthrough notebook trains on.

Writes results/qubit_windows.npz with the E13/E14 windows (same inputs, scaling and
masking as scripts/e13_drivers.py) for the 2018-21 training fires, the 2022 validation
fires and the 88 Yellowknife (ZF015) windows of 10-20 Aug 2023. Future weather is
constant across a window, so it is stored once per window. Values are float32 to keep
the notebook small.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.experiments import load_frozen
from scripts.e13_drivers import FUTURE, HIST, TERRAIN, load_table, windows

ZF015_UID = 242981


def main():
    config = yaml.safe_load((ROOT / "config.yaml").read_text())
    frozen = load_frozen(config_path=ROOT / "config.yaml", frozen_path=ROOT / "results/frozen_config.json")
    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = load_table(years)
    cols = HIST + TERRAIN + FUTURE[1] + FUTURE[3]
    W, rows = windows(df, cols, frozen["L"], config, [c for c in cols if c.startswith("f")])
    dates = rows.date.to_numpy().astype("datetime64[D]")
    case = ((rows.uid.eq(ZF015_UID)).to_numpy()
            & (dates >= np.datetime64("2023-08-10")) & (dates <= np.datetime64("2023-08-20")))
    keep = rows.split.isin(["train", "val"]).to_numpy() | case
    if case.sum() != 88:
        raise RuntimeError(f"expected 88 ZF015 windows, got {case.sum()}")
    W, rows = W[keep], rows.loc[keep].reset_index(drop=True)
    part = np.where(rows.split.eq("train"), 0, np.where(rows.split.eq("val"), 1, 2)).astype(np.int8)
    daily = len(HIST) + len(TERRAIN)
    static = W[:, -1, daily:]
    if not np.array_equal(W[:, :, daily:], np.repeat(static[:, None, :], W.shape[1], axis=1)):
        raise RuntimeError("future inputs are not constant across the window")
    out = ROOT / "results" / "qubit_windows.npz"
    np.savez_compressed(
        out, daily=W[:, :, :daily].astype(np.float32), static=static.astype(np.float32),
        part=part, y1=rows.y1.to_numpy(np.float32), y3=rows.y3.to_numpy(np.float32),
        dates=rows.date.to_numpy().astype("datetime64[D]"), sectors=rows.sector.to_numpy(np.int8),
        daily_names=np.array(HIST + TERRAIN), static_names=np.array(FUTURE[1] + FUTURE[3]))
    print(f"{out.relative_to(ROOT)}: {len(rows)} windows "
          f"(train {np.sum(part == 0)}, val {np.sum(part == 1)}, ZF015 {np.sum(part == 2)}), "
          f"{out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
