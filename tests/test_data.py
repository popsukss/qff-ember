"""Checks for U1: hotspots -> daily outlines -> sector table.

Synthetic tests pin the behaviour; the real-data tests at the bottom need
`data/processed/sectors_2023.parquet` and `results/e10_2023.json` to exist.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ember.data.sectors import build_sectors
from ember.schema import INPUT_COLUMNS, SECTOR_COLUMNS, validate_sectors

ROOT = Path(__file__).resolve().parents[1]
LAT0, LON0 = 62.0, -114.0
KM_PER_DEG_LAT = 110.574
KM_PER_DEG_LON = 111.320 * math.cos(math.radians(LAT0))


def _pt(uid, day, east_km=0.0, north_km=0.0, ws=10.0, wd=270.0, rh=30.0, hour=20):
    return {
        "uid": uid,
        "rep_date": pd.Timestamp("2023-07-01", tz="UTC") + pd.Timedelta(days=day - 1, hours=hour),
        "lat": LAT0 + north_km / KM_PER_DEG_LAT,
        "lon": LON0 + east_km / KM_PER_DEG_LON,
        "ws": ws, "wd": wd, "rh": rh, "isi": 5.0, "bui": 40.0, "ros": 2.0,
    }


def synthetic(drop_day: int | None = None, isolated: bool = False, placeholder: bool = False) -> pd.DataFrame:
    """Fire 1 grows east 1 km/day for 12 days; fire 2 is small and lasts 4 days."""
    rows = []
    for d in range(1, 13):
        if d == drop_day:
            continue
        # small static core around the centre (all directions)
        for e, n in [(0.1, 0), (-0.1, 0), (0, 0.1), (0, -0.1), (0.05, 0.05)]:
            rows.append(_pt(1, d, e, n))
        # eastern head: distance d-1 km (day 1 -> core only), plus a trailing point so it is not isolated
        if d >= 2:
            rows.append(_pt(1, d, float(d - 1), 0.0))
            rows.append(_pt(1, d, float(d - 1) - 0.5, 0.0))
    if placeholder:
        # day 4: one real reading and one placeholder (rh == 0, ws == 0) -> median must ignore it
        rows = [r for r in rows if not (r["uid"] == 1 and r["rep_date"].day == 4)]
        rows.append(_pt(1, 4, 3.0, 0.0, ws=10.0))
        rows.append(_pt(1, 4, 2.5, 0.0, ws=0.0, wd=0.0, rh=0.0))
    if isolated:
        rows.append(_pt(1, 5, 0.0, 20.0))  # lone point 20 km north
    for d in range(1, 5):
        rows.append(_pt(2, d, 0.0, 0.0))
        rows.append(_pt(2, d, 0.3, 0.0))
    df = pd.DataFrame(rows)
    df["lon"] = np.where(df["uid"] == 2, df["lon"] + 5.0, df["lon"])  # fire 2 far away
    return df


def _row(tab, uid, day, sector):
    m = (tab.uid == uid) & (tab.day_index == day) & (tab.sector == sector)
    assert m.sum() == 1
    return tab[m].iloc[0]


def test_schema_and_contract():
    tab = build_sectors(synthetic())
    assert list(tab.columns) == list(SECTOR_COLUMNS)
    assert validate_sectors(tab) == []


def test_days_contiguous_and_day_index():
    tab = build_sectors(synthetic())
    f1 = tab[tab.uid == 1]
    assert sorted(f1.day_index.unique()) == list(range(1, 13))
    assert len(f1) == 12 * 8
    f2 = tab[tab.uid == 2]
    assert sorted(f2.day_index.unique()) == [1, 2, 3, 4]


def test_east_growth_one_km_per_day():
    tab = build_sectors(synthetic())
    for d in range(3, 13):
        r = _row(tab, 1, d, 2)  # sector 2 = east
        assert r.r_km == pytest.approx(d - 1, rel=0.03)
        assert r.dd_km == pytest.approx(1.0, abs=0.05)
    west = tab[(tab.uid == 1) & (tab.sector == 6)]
    assert west.r_km.max() < 0.5


def test_r_never_decreases_and_dd_nonnegative():
    tab = build_sectors(synthetic())
    assert (tab.dd_km >= -1e-9).all()
    for (_, _), g in tab.groupby(["uid", "sector"]):
        assert (np.diff(g.sort_values("date").r_km.to_numpy()) >= -1e-9).all()


def test_targets():
    tab = build_sectors(synthetic())
    r5 = _row(tab, 1, 5, 2)
    assert r5.y1 == pytest.approx(1.0, abs=0.05)
    assert r5.y3 == pytest.approx(3.0, abs=0.1)
    assert math.isnan(_row(tab, 1, 12, 2).y1)
    assert math.isnan(_row(tab, 1, 10, 2).y3)
    assert not math.isnan(_row(tab, 1, 9, 2).y3)


def test_no_future_leakage():
    full = build_sectors(synthetic())
    cut = synthetic()
    cut = cut[cut.rep_date < pd.Timestamp("2023-07-07", tz="UTC")]  # days 1..6 only
    part = build_sectors(cut)
    cols = list(INPUT_COLUMNS)
    a = full[(full.uid == 1) & (full.day_index <= 6)].sort_values(["date", "sector"])[cols].reset_index(drop=True)
    b = part[(part.uid == 1) & (part.day_index <= 6)].sort_values(["date", "sector"])[cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_isolated_point_ignored():
    tab = build_sectors(synthetic(isolated=True))
    north = tab[(tab.uid == 1) & (tab.sector == 0)]
    assert north.r_km.max() < 1.0


def test_masked_day_carries_weather_and_jumps_after():
    tab = build_sectors(synthetic(drop_day=8))
    r8 = _row(tab, 1, 8, 2)
    assert bool(r8.masked) and r8.n_hot == 0
    assert r8.dd_km == pytest.approx(0.0, abs=1e-9)
    assert r8.ws == pytest.approx(10.0)
    r9 = _row(tab, 1, 9, 2)
    assert not bool(r9.masked)
    assert r9.dd_km == pytest.approx(2.0, abs=0.08)


def test_placeholder_weather_dropped():
    tab = build_sectors(synthetic(placeholder=True))
    assert _row(tab, 1, 4, 2).ws == pytest.approx(10.0)
    assert _row(tab, 1, 4, 2).rh == pytest.approx(30.0)


def test_wind_component_signs():
    tab = build_sectors(synthetic())  # wind from 270 (west) blows east
    assert _row(tab, 1, 5, 2).w_k == pytest.approx(10.0, abs=1e-6)
    assert _row(tab, 1, 5, 6).w_k == pytest.approx(-10.0, abs=1e-6)
    assert _row(tab, 1, 5, 0).w_k == pytest.approx(0.0, abs=1e-6)


def test_split_from_year():
    tab = build_sectors(synthetic())
    assert set(tab.split) == {"test"} and set(tab.year) == {2023}


# ---------------------------------------------------------------- real data (U1 acceptance)
SECTORS_2023 = ROOT / "data" / "processed" / "sectors_2023.parquet"
E10_2023 = ROOT / "results" / "e10_2023.json"


def test_real_2023_contract():
    tab = pd.read_parquet(SECTORS_2023)
    assert validate_sectors(tab) == []
    assert tab.uid.nunique() >= 20


def test_real_2023_zf015_case_window():
    tab = pd.read_parquet(SECTORS_2023)
    z = tab[tab.uid == 242981]
    assert len(z) > 0, "ZF015-23 (uid 242981) missing"
    days = pd.date_range("2023-08-10", "2023-08-20", freq="D")
    for d in days:
        sub = z[z.date == d]
        assert len(sub) == 8, f"{d.date()} has {len(sub)} rows"
        assert not sub.masked.any(), f"{d.date()} masked but hotspots exist that day"


def test_real_2023_kept_fires_rule():
    tab = pd.read_parquet(SECTORS_2023)
    hot_days = tab[~tab.masked].groupby("uid").date.nunique()
    assert (hot_days >= 10).all()


def test_real_e10_report():
    rep = json.loads(E10_2023.read_text())
    assert 0.0 <= rep["median_iou"] <= 1.0
    assert len(rep["per_fire"]) >= 20
    assert all(0.0 <= f["iou"] <= 1.0 for f in rep["per_fire"])
