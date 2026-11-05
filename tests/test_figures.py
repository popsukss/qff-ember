"""Checks for U6: slide figures and the ZF015 case study (E8)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"


def test_figures_exist():
    for name in ["e1_test_mae.png", "e2_wind.png", "e4_seeds.png", "e8_zf015_map.png"]:
        p = FIG / name
        assert p.exists() and p.stat().st_size > 10_000, name


def test_e8_table():
    t = pd.read_csv(RES / "e8_zf015.csv")
    assert {"date", "sector", "lead", "r_obs", "r_pred_qrc", "r_pred_fbp", "r_pred_persistence"} <= set(t.columns)
    assert set(t.sector) == set(range(8)) and set(t.lead) == {1, 3}
    dates = pd.to_datetime(t.date)
    assert dates.min() >= pd.Timestamp("2023-08-10") and dates.max() <= pd.Timestamp("2023-08-20")
    assert (t[["r_pred_qrc", "r_pred_fbp", "r_pred_persistence"]].min() >= 0).all()


def test_e8_eta():
    e = json.loads((RES / "e8_eta.json").read_text())
    assert {"town", "town_lat", "town_lon", "distance_km", "sector", "per_date"} <= set(e)
    assert 0 <= e["sector"] < 8 and e["distance_km"] > 0
    for row in e["per_date"]:
        assert {"date", "r_obs", "eta_days_qrc", "eta_days_fbp"} <= set(row)
