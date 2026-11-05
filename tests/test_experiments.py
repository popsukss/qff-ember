"""Checks for U5: E1, E2, E3, E4, E7 and the validation-only freeze.

Unit tests run on synthetic tables; the real-output tests at the bottom need
results/ files produced by scripts/run_experiments.py.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ember import experiments as ex
from ember.schema import SECTOR_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
MODELS = {"persistence", "mean", "fbp", "raw", "esn5", "esn20", "qrc", "qrc_raw", "haar"}


def fake_sectors(n_fires=8, n_days=14, seed=0):
    rng = np.random.default_rng(seed)
    years = [2018, 2019, 2020, 2021, 2022, 2022, 2023, 2023]
    split = {2018: "train", 2019: "train", 2020: "train", 2021: "train", 2022: "val", 2023: "test"}
    rows = []
    for f in range(n_fires):
        year = years[f % len(years)]
        r = np.zeros(8)
        for d in range(1, n_days + 1):
            dd = rng.exponential(0.3, size=8) * (rng.uniform(size=8) < 0.3)
            r = r + dd
            for k in range(8):
                rows.append({"uid": 100 + f, "year": year,
                             "date": pd.Timestamp(f"{year}-07-01") + pd.Timedelta(days=d - 1),
                             "day_index": d, "sector": k, "r_km": r[k], "dd_km": dd[k],
                             "ws": 10.0, "w_k": float(10 * np.cos(np.deg2rad(45 * k - 90))),
                             "isi": float(rng.uniform(0, 20)), "bui": float(rng.uniform(0, 100)),
                             "rh": float(rng.uniform(10, 90)), "ros": float(rng.uniform(0, 5)),
                             "n_hot": 5, "masked": False, "y1": np.nan, "y3": np.nan, "split": split[year]})
    df = pd.DataFrame(rows)[list(SECTOR_COLUMNS)]
    g = df.groupby(["uid", "sector"]).r_km
    df["y1"] = g.shift(-1) - df.r_km
    df["y3"] = g.shift(-3) - df.r_km
    return df


# ---------------------------------------------------------------- unit
@pytest.mark.parametrize("name,width", [("raw", 25), ("esn5", 5), ("esn20", 20), ("qrc", 20), ("qrc_raw", 45),
                                        ("haar", 20)])
def test_feature_widths(name, width):
    W = np.random.default_rng(0).uniform(size=(6, 5, 5))
    F = ex.featurize(name, W, gamma=np.pi / 2, seed=7, ent_seed=0, esn_seed=0)
    assert F.shape == (6, width)
    assert np.isfinite(F).all()


def test_predictions_clipped_at_zero():
    assert ex.clip_prediction(np.array([-1.0, 0.0, 2.0])).tolist() == [0.0, 0.0, 2.0]


def test_mae_growing_uses_only_growing_rows():
    y = np.array([0.0, 0.0, 2.0, 4.0])
    yhat = np.array([5.0, 5.0, 1.0, 4.0])
    assert ex.mae_growing(y, yhat) == pytest.approx(0.5)


def test_wind_bins():
    b = ex.wind_bin(w_k=np.array([9.0, -9.0, 1.0, 0.0]), ws=np.array([10.0, 10.0, 10.0, 0.0]))
    assert b.tolist() == ["downwind", "upwind", "crosswind", "crosswind"]


def test_e3_refuses_test_rows():
    df = fake_sectors()
    with pytest.raises(ValueError):
        ex.run_e3(df, window_days=[5], gammas=[np.pi / 2], input_maps=[["w_k", "isi", "dd_km", "bui", "rh"]])


def test_e3_independent_of_test_rows():
    df = fake_sectors()
    trval = df[df.split != "test"]
    a = ex.run_e3(trval, window_days=[3, 5], gammas=[np.pi / 2], input_maps=[["w_k", "isi", "dd_km", "bui", "rh"]])
    assert {"L", "gamma", "input_map", "lam_1", "lam_3", "val_mae_1", "val_mae_3", "val_skill_mean"} <= set(a.columns)
    assert len(a) == 2


def test_freeze_and_hash_guard(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("a: 1\n")
    frozen = tmp_path / "frozen.json"
    ex.freeze({"L": 5, "gamma": 1.0}, config_path=cfg, out_path=frozen)
    got = ex.load_frozen(config_path=cfg, frozen_path=frozen)
    assert got["L"] == 5 and got["config_sha256"] == hashlib.sha256(b"a: 1\n").hexdigest()
    cfg.write_text("a: 2\n")
    with pytest.raises(RuntimeError):
        ex.load_frozen(config_path=cfg, frozen_path=frozen)
    with pytest.raises(RuntimeError):
        ex.load_frozen(config_path=cfg, frozen_path=tmp_path / "missing.json")


def test_persistence_skill_zero_on_fake():
    df = fake_sectors()
    table = ex.score_models(df, models=["persistence", "mean", "qrc"], L=5, gamma=np.pi / 2,
                            input_map=["w_k", "isi", "dd_km", "bui", "rh"], splits=["val"], n_boot=50)
    p = table[(table.model == "persistence")]
    assert np.allclose(p.skill, 0.0)
    assert (table.mae_lo <= table.mae + 1e-12).all() and (table.mae <= table.mae_hi + 1e-12).all()
    assert set(table.lead) == {1, 3}


# ---------------------------------------------------------------- real outputs (U5 acceptance)
def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_real_e3_grid():
    e3 = pd.read_csv(RES / "e3_val.csv")
    assert len(e3) == 3 * 3 * 2
    assert e3.val_skill_mean.notna().all()


def test_real_frozen_config_matches_current_config():
    fr = json.loads((RES / "frozen_config.json").read_text())
    assert fr["config_sha256"] == _sha(ROOT / "config.yaml")
    e3 = pd.read_csv(RES / "e3_val.csv")
    best = e3.loc[e3.val_skill_mean.idxmax()]
    assert fr["L"] == int(best.L) and fr["gamma"] == pytest.approx(float(best.gamma))


def test_real_metrics_table():
    m = pd.read_csv(RES / "metrics.csv")
    need = {"model", "split", "lead", "mae", "mae_lo", "mae_hi", "skill", "mae_growing", "iou",
            "n_rows", "n_fires", "lam", "frozen_sha256"}
    assert need <= set(m.columns)
    assert MODELS <= set(m.model)
    assert set(m.split) == {"val", "test"} and set(m.lead) == {1, 3}
    assert len(m) == len(m.drop_duplicates(["model", "split", "lead"]))
    assert np.allclose(m[m.model == "persistence"].skill, 0.0)
    assert (m.mae_lo <= m.mae + 1e-9).all() and (m.mae <= m.mae_hi + 1e-9).all()
    assert ((m.iou >= 0) & (m.iou <= 1)).all()
    fr_sha = _sha(RES / "frozen_config.json")
    assert (m.frozen_sha256 == fr_sha).all()
    assert m[m.split == "test"].n_fires.max() <= 392


def test_real_e4_seeds():
    e4 = pd.read_csv(RES / "e4_seeds.csv")
    assert e4[e4.model == "haar"].seed.nunique() == 10
    assert e4[e4.model == "qrc"].seed.nunique() == 5
    assert set(e4.lead) == {1, 3} and set(e4.split) == {"test"}


def test_real_e2_wind():
    e2 = pd.read_csv(RES / "e2_wind.csv")
    assert set(e2.wind_bin) == {"downwind", "crosswind", "upwind"}
    assert {"qrc", "persistence", "fbp"} <= set(e2.model)
    assert (e2.n_rows > 0).all()
