"""Checks on the saved E14 fair-baseline outputs (no simulation)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
VERSIONS = {"base5", "A7", "B10_h"}
CONFIGS = {
    "qrc": {"gamma=0.25pi", "gamma=0.50pi", "gamma=0.75pi"},
    "esn": {"input_scale=0.3", "input_scale=1.0", "input_scale=3.0"},
    "raw": {"raw"},
}
SEEDS = {"qrc": {7, 11, 13, 17, 19}, "esn": {0, 1, 2, 3, 4}, "raw": {0}}


@pytest.fixture(scope="module")
def metrics():
    return pd.read_csv(RES / "e14_metrics.csv")


@pytest.fixture(scope="module")
def selected():
    return pd.read_csv(RES / "e14_selected.csv")


def test_metrics_grid_complete(metrics):
    assert {"family", "version", "n_qubits", "lead", "config", "seed", "split", "mae", "lam"} <= set(metrics)
    assert set(metrics.version) == VERSIONS and set(metrics.lead) == {1, 3}
    assert set(metrics.split) == {"val", "test"}
    for family, configs in CONFIGS.items():
        m = metrics[metrics.family == family]
        assert set(m.config) == configs
        assert set(m.seed) == SEEDS[family]
        counts = m.groupby(["version", "lead", "config", "split"]).seed.nunique()
        assert (counts == len(SEEDS[family])).all()
        assert len(counts) == len(VERSIONS) * 2 * len(configs) * 2
    assert dict(metrics.groupby("version").n_qubits.first()) == {"base5": 5, "A7": 7, "B10_h": 10}
    assert np.isfinite(metrics.mae).all() and (metrics.mae > 0).all()


def test_selected_is_best_seed_mean_on_val(metrics, selected):
    assert len(selected) == len(VERSIONS) * 2 * 3
    for row in selected.itertuples():
        m = metrics[(metrics.family == row.family) & (metrics.version == row.version) &
                    (metrics.lead == row.lead) & (metrics.split == "val")]
        means = m.groupby("config").mae.mean()
        assert row.config == means.idxmin()
        assert row.val_mean == pytest.approx(means.min())
        assert row.n_seeds == len(SEEDS[row.family])
        for split in ("val", "test"):
            assert getattr(row, f"{split}_min") <= getattr(row, f"{split}_mean") <= getattr(row, f"{split}_max")


def test_paired_and_meta():
    paired = pd.read_csv(RES / "e14_paired.csv")
    assert {"split", "version", "lead", "model", "reference", "diff", "lo", "hi"} <= set(paired)
    test = paired[paired.split == "test"]
    assert set(zip(test.model, test.reference)) == {("qrc", "raw"), ("qrc", "esn"), ("esn", "raw")}
    assert len(test) == len(VERSIONS) * 2 * 3
    assert (test.lo <= test["diff"]).all() and (test["diff"] <= test.hi).all()
    meta = json.loads((RES / "e14_meta.json").read_text())
    assert meta["qrc"]["seeds"] == [7, 11, 13, 17, 19]
    assert meta["esn"]["seeds"] == [0, 1, 2, 3, 4]
    assert meta["esn"]["input_scale_grid"] == [0.3, 1.0, 3.0]
    assert len(meta["qrc"]["gamma_grid"]) == 3 and meta["runtime_s"] > 0
