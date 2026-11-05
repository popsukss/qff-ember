"""Checks for E6: real ibm_fez features for the 88 ZF015 windows vs FakeFez and exact."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def test_hardware_features_shape_and_range():
    F = np.load(ROOT / "hardware" / "e6_hw_features.npy")
    assert F.shape == (88, 15)
    assert np.isfinite(F).all() and np.abs(F).max() <= 1.0 + 1e-12


def test_e6_summary():
    t = pd.read_csv(RES / "e6_hardware.csv")
    assert {"source", "lead", "mae", "n_rows", "feature_mae"} <= set(t.columns)
    assert set(t.source) == {"exact", "fake_fez", "hardware", "persistence"}
    assert set(t.lead) == {1, 3}
    assert (t.loc[t.source == "exact", "feature_mae"] == 0).all()
    assert (t.loc[t.source == "hardware", "feature_mae"] > 0).all()


def test_e6_predictions_reproduce_summary():
    p = pd.read_csv(RES / "e6_predictions.csv")
    t = pd.read_csv(RES / "e6_hardware.csv")
    assert {"date", "sector", "lead", "y", "yhat_exact", "yhat_fake_fez", "yhat_hardware", "yhat_persistence"} <= set(p.columns)
    for _, r in t.iterrows():
        sub = p[p.lead == r.lead]
        assert np.mean(np.abs(sub.y - sub[f"yhat_{r.source}"])) == pytest.approx(r.mae, abs=1e-12)
        assert len(sub) == r.n_rows
    assert (p[["yhat_exact", "yhat_fake_fez", "yhat_hardware"]] >= 0).all().all()


def test_e6_uses_logged_job():
    jobs = json.loads((ROOT / "hardware" / "jobs.json").read_text())
    meta = json.loads((RES / "e6_meta.json").read_text())
    assert meta["job_id"] in {j["job_id"] for j in jobs}
    assert meta["shots"] == 10000


def test_e6_figure():
    p = ROOT / "figures" / "e6_hardware.png"
    assert p.exists() and p.stat().st_size > 10_000
