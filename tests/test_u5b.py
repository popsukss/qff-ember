"""Checks for U5b: Z-only read-out, E5 shots curve, E9 sparse data, frozen hardware case."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import ember.hardware as hw
from ember import experiments as ex
from ember.windows import MinMaxScaler01, make_windows

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FROZEN = RES / "frozen_config.json"


def test_qrc_z_is_first_15_features():
    W = np.random.default_rng(0).uniform(size=(4, 5, 5))
    full = ex.featurize("qrc", W, gamma=np.pi / 4, seed=7, ent_seed=0, esn_seed=0)
    z = ex.featurize("qrc_z", W, gamma=np.pi / 4, seed=7, ent_seed=0, esn_seed=0)
    np.testing.assert_allclose(z, full[:, :15])


def _hot(n_days=12):
    rows = [{"uid": 1, "rep_date": pd.Timestamp("2023-07-01", tz="UTC") + pd.Timedelta(days=d, hours=h)}
            for d in range(n_days) for h in (1, 2)]
    return pd.DataFrame(rows)


def test_drop_hotspot_days_keeps_day_one_and_fraction():
    hot = _hot()
    out = ex.drop_hotspot_days(hot, frac=0.30, seed=0)
    days = out.rep_date.dt.normalize()
    assert pd.Timestamp("2023-07-01", tz="UTC") in set(days)
    assert days.nunique() == 12 - round(0.30 * 11)  # day one is never dropped
    pd.testing.assert_frame_equal(out, ex.drop_hotspot_days(hot, frac=0.30, seed=0))
    pd.testing.assert_frame_equal(ex.drop_hotspot_days(hot, frac=0.0, seed=0).reset_index(drop=True),
                                  hot.reset_index(drop=True))


def test_frozen_reservoir_uses_frozen_gamma():
    fr = json.loads(FROZEN.read_text())
    res = hw.frozen_reservoir(FROZEN)
    assert res.gamma == pytest.approx(fr["gamma"])
    ref = __import__("ember.reservoir", fromlist=["ChainReservoir"]).ChainReservoir(
        gamma=fr["gamma"], seed=fr["reservoir_seed"])
    np.testing.assert_allclose(res.gains, ref.gains)


# ---------------------------------------------------------------- real outputs
def test_real_case_archive_matches_independent_rebuild():
    fr = json.loads(FROZEN.read_text())
    years = range(2018, 2024)
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years], ignore_index=True)
    import yaml
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text())["data"]
    W, rows = make_windows(df, fr["input_map"], fr["L"], cfg["forecast_from_day"], cfg["max_masked_in_window"])
    sc = MinMaxScaler01().fit(W[(rows.split == "train").to_numpy()])
    Ws = sc.transform(W)
    m = (rows.uid == 242981) & rows.date.between("2023-08-10", "2023-08-20")
    sel = rows[m].sort_values(["date", "sector"])
    with np.load(ROOT / "hardware" / "zf015_case.npz", allow_pickle=False) as case:
        assert case["windows"].shape == (88, fr["L"], 5)
        np.testing.assert_allclose(case["windows"], Ws[sel.index.to_numpy()], atol=1e-12)
        assert (case["uid"] == 242981).all()
        assert (case["sectors"] == np.tile(np.arange(8), 11)).all()


def test_real_e5_shots():
    e5 = pd.read_csv(RES / "e5_shots.csv")
    assert {"shots", "noise", "mae", "mae_exact", "n_rows", "lead"} <= set(e5.columns)
    assert set(e5.shots) == {100, 1000, 2000, 10000}
    assert set(e5.noise) == {"ideal", "fake_fez"}
    assert set(e5.lead) == {1, 3}
    assert np.isfinite(e5.mae).all()


def test_real_e9_sparse_and_consistent_with_e1():
    e9 = pd.read_csv(RES / "e9_sparse.csv")
    assert set(e9.drop_frac.round(2)) == {0.0, 0.15, 0.30, 0.45}
    assert {"qrc", "esn20", "persistence", "fbp"} <= set(e9.model)
    assert set(e9.lead) == {1, 3}
    m = pd.read_csv(RES / "metrics.csv")
    for _, r in e9[e9.drop_frac == 0].iterrows():
        ref = m[(m.model == r.model) & (m.split == "test") & (m.lead == r.lead)].mae.item()
        assert r.mae == pytest.approx(ref, abs=1e-9)
