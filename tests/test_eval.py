"""Checks for U3: windows, scaling, read-out, baselines, metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge

from ember.baselines import ESN, fbp_fit_burn_minutes, fbp_predict, mean_advance, persistence, raw_window_features
from ember.metrics import fire_bootstrap_ci, mae, outline_iou, skill
from ember.readout import RidgeReadout, select_lambda
from ember.schema import SECTOR_COLUMNS
from ember.windows import MinMaxScaler01, make_windows

INPUT_MAP = ["w_k", "isi", "dd_km", "bui", "rh"]


def fake_sectors(n_fires=6, n_days=12, seed=0):
    """Contract-shaped table; years spread over train/val/test."""
    rng = np.random.default_rng(seed)
    years = [2018, 2019, 2020, 2021, 2022, 2023]
    rows = []
    for f in range(n_fires):
        year = years[f % len(years)]
        for d in range(1, n_days + 1):
            for k in range(8):
                rows.append({
                    "uid": 1000 + f, "year": year, "date": pd.Timestamp(f"{year}-07-01") + pd.Timedelta(days=d - 1),
                    "day_index": d, "sector": k, "r_km": float(d), "dd_km": float(rng.uniform(0, 2)),
                    "ws": 10.0, "w_k": float(rng.uniform(-10, 10)), "isi": float(rng.uniform(0, 20)),
                    "bui": float(rng.uniform(0, 100)), "rh": float(rng.uniform(10, 90)), "ros": 2.0,
                    "n_hot": 5, "masked": bool(d == 6), "y1": np.nan, "y3": np.nan,
                    "split": {2018: "train", 2019: "train", 2020: "train", 2021: "train", 2022: "val", 2023: "test"}[year],
                })
    df = pd.DataFrame(rows)[list(SECTOR_COLUMNS)]
    df["y1"] = df["dd_km"] * 1.1
    df["y3"] = df["dd_km"] * 3.0
    return df


# ---------------------------------------------------------------- metrics
def test_mae_and_skill():
    assert mae([1, 2], [2, 4]) == pytest.approx(1.5)
    assert skill(1.0, 2.0) == pytest.approx(0.5)
    assert skill(2.0, 2.0) == pytest.approx(0.0)


def test_outline_iou():
    r = np.full(8, 2.0)
    assert outline_iou(r, r) == pytest.approx(1.0)
    assert outline_iou(r, r * 0.5) == pytest.approx(0.25, rel=1e-9)  # similar octagons, area ratio 1/4
    assert 0.0 <= outline_iou(r, np.array([2, 0, 0, 0, 2, 0, 0, 0.0])) <= 1.0


def test_fire_bootstrap_resamples_fires_not_rows():
    errs = pd.DataFrame({"uid": [1] * 100 + [2] * 100, "abs_err": [0.0] * 100 + [2.0] * 100})
    lo, hi = fire_bootstrap_ci(errs, n_boot=400, seed=0)
    # resampling 2 fires -> means are 0, 1 or 2; a row bootstrap would give a tight interval around 1
    assert lo == pytest.approx(0.0) and hi == pytest.approx(2.0)


# ---------------------------------------------------------------- windows + scaling
def test_windows_stay_inside_one_fire_and_respect_start_day():
    df = fake_sectors()
    W, idx = make_windows(df, input_map=INPUT_MAP, L=5, forecast_from_day=3, max_masked=2)
    assert W.shape[1:] == (5, 5)
    assert len(idx) == len(W)
    assert idx.day_index.min() >= 3
    # day 3 window = days -1..3 -> padded days have dd_km = 0 and use day-1 weather
    assert set(idx.columns) >= {"uid", "date", "sector", "day_index", "split", "y1", "y3"}
    one = idx[(idx.uid == 1000) & (idx.sector == 0) & (idx.day_index == 7)].index[0]
    raw = df[(df.uid == 1000) & (df.sector == 0)].sort_values("day_index")
    np.testing.assert_allclose(W[one][:, INPUT_MAP.index("dd_km")], raw.dd_km.to_numpy()[2:7])


def test_padding_before_day_one():
    df = fake_sectors()
    W, idx = make_windows(df, input_map=INPUT_MAP, L=5, forecast_from_day=3, max_masked=2)
    i = idx[(idx.uid == 1000) & (idx.sector == 0) & (idx.day_index == 3)].index[0]
    np.testing.assert_allclose(W[i][:2, INPUT_MAP.index("dd_km")], [0.0, 0.0])


def test_max_masked_rule():
    df = fake_sectors()
    df.loc[df.day_index.isin([5, 6, 7]), "masked"] = True
    W, idx = make_windows(df, input_map=INPUT_MAP, L=5, forecast_from_day=3, max_masked=2)
    assert not ((idx.uid == 1000) & (idx.day_index == 7)).any()  # window 3..7 has 3 masked days


def test_scaler_fit_on_train_only_and_clipped():
    df = fake_sectors()
    W, idx = make_windows(df, input_map=INPUT_MAP, L=5, forecast_from_day=3, max_masked=2)
    tr = (idx.split == "train").to_numpy()
    sc = MinMaxScaler01().fit(W[tr])
    W2 = W.copy()
    W2[~tr] *= 100.0  # wild val/test values must not move the scaler
    sc2 = MinMaxScaler01().fit(W2[tr])
    np.testing.assert_allclose(sc.lo, sc2.lo)
    np.testing.assert_allclose(sc.hi, sc2.hi)
    out = sc.transform(W2)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_no_fire_in_two_splits():
    df = fake_sectors()
    _, idx = make_windows(df, input_map=INPUT_MAP, L=5, forecast_from_day=3, max_masked=2)
    assert idx.groupby("uid").split.nunique().max() == 1


# ---------------------------------------------------------------- read-out
def test_ridge_matches_sklearn():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 20))
    y = X @ rng.normal(size=20) + 0.3 + rng.normal(scale=0.1, size=200)
    for lam in [1e-6, 1e-2, 10.0]:
        ours = RidgeReadout(lam).fit(X, y).predict(X)
        ref = Ridge(alpha=lam, fit_intercept=True).fit(X, y).predict(X)
        np.testing.assert_allclose(ours, ref, atol=1e-8)


def test_select_lambda_uses_validation():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(60, 30))
    y = X[:, 0] + rng.normal(scale=0.5, size=60)
    Xv = rng.normal(size=(200, 30))
    yv = Xv[:, 0] + rng.normal(scale=0.5, size=200)
    lam = select_lambda(X, y, Xv, yv, grid=[1e-6, 1.0, 1e6])
    assert lam == 1.0 or lam == 1e-6
    assert lam != 1e6


# ---------------------------------------------------------------- baselines
def test_persistence_and_mean():
    assert persistence(np.array([2.0]), h=3)[0] == pytest.approx(6.0)
    assert mean_advance(np.array([1.0, 3.0]), n=2).tolist() == [2.0, 2.0]


def test_fbp_physics_downwind_only():
    ros = np.array([2.0, 2.0, 2.0])
    ws = np.array([10.0, 10.0, 10.0])
    w_k = np.array([10.0, -10.0, 0.0])  # downwind, upwind, crosswind
    p = fbp_predict(ros, ws, w_k, h=1, burn_minutes=100.0)
    assert p[0] == pytest.approx(2.0 * 100.0 / 1000.0)
    assert p[1] == pytest.approx(0.0)
    assert p[2] == pytest.approx(0.0, abs=1e-12)
    assert fbp_predict(ros, np.zeros(3), np.zeros(3), h=1, burn_minutes=100.0).tolist() == [0.0, 0.0, 0.0]


def test_fbp_fit_recovers_burn_minutes():
    rng = np.random.default_rng(3)
    ros = rng.uniform(0.5, 5, 300)
    ws = np.full(300, 10.0)
    w_k = rng.uniform(-10, 10, 300)
    y = fbp_predict(ros, ws, w_k, h=1, burn_minutes=240.0)
    assert fbp_fit_burn_minutes(ros, ws, w_k, y, h=1) == pytest.approx(240.0, rel=1e-6)


def test_raw_window_features_shape():
    W = np.random.default_rng(0).uniform(size=(4, 5, 5))
    assert raw_window_features(W).shape == (4, 25)


def test_esn_fully_connected_and_restart():
    esn = ESN(n_units=5, n_inputs=5, seed=0, spectral_radius=0.9)
    assert np.count_nonzero(esn.W) == 25 and np.count_nonzero(esn.W_in) == 25
    assert max(abs(np.linalg.eigvals(esn.W))) == pytest.approx(0.9, rel=1e-6)
    W = np.random.default_rng(0).uniform(size=(3, 5, 5))
    F = esn.features(W)
    assert F.shape == (3, 5)
    np.testing.assert_allclose(esn.features(W[1:2]), F[1:2])  # restart per window: no carry-over
