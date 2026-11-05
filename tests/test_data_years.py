"""Checks for U1b: sector tables for every year (2018-2023) and the fire-count report."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ember.schema import SPLIT_YEARS, split_for_year, validate_sectors

ROOT = Path(__file__).resolve().parents[1]
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
COUNTS = ROOT / "results" / "fire_counts.json"


def _sectors(year):
    return pd.read_parquet(ROOT / "data" / "processed" / f"sectors_{year}.parquet")


@pytest.mark.parametrize("year", YEARS)
def test_year_contract(year):
    tab = _sectors(year)
    assert validate_sectors(tab) == []
    assert set(tab.year) == {year}
    assert set(tab.split) == {split_for_year(year)}
    assert tab.uid.nunique() >= 10


@pytest.mark.parametrize("year", YEARS)
def test_year_kept_fire_rule(year):
    tab = _sectors(year)
    assert (tab[~tab.masked].groupby("uid").date.nunique() >= 10).all()


@pytest.mark.parametrize("year", YEARS)
def test_year_continuity(year):
    tab = _sectors(year)
    assert tab.dd_km.max() <= 50.0
    assert (tab.dd_km > 30.0).sum() <= 20


def test_fires_never_shared_between_years():
    seen = {}
    for year in YEARS:
        for uid in _sectors(year).uid.unique():
            assert uid not in seen, f"uid {uid} in {seen.get(uid)} and {year}"
            seen[uid] = year


def test_enough_training_fires():
    train = sum(_sectors(y).uid.nunique() for y in SPLIT_YEARS["train"])
    assert train >= 100


def test_fire_count_report_matches_tables():
    rep = json.loads(COUNTS.read_text())
    for year in YEARS:
        tab = _sectors(year)
        row = rep[str(year)]
        assert row["fires"] == tab.uid.nunique()
        assert row["sector_rows"] == len(tab)
        assert row["split"] == split_for_year(year)
        # forecast origins: fire-days with day_index >= 3 and a +1 day target
        org = tab[(tab.day_index >= 3) & tab.y1.notna()].drop_duplicates(["uid", "date"])
        assert row["origin_days"] == len(org)
