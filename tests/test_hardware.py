"""Checks for U4: hardware preparation. Nothing here may contact IBM."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit_ibm_runtime.fake_provider import FakeFez

import ember.hardware as hw
from ember.reservoir import ChainReservoir


def windows(k=4, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(k, 5, 5))


@pytest.fixture(autouse=True)
def no_ibm_account(monkeypatch):
    """Any attempt to build a real runtime service fails the test."""
    def boom(*a, **k):
        raise AssertionError("tried to contact IBM")
    monkeypatch.setattr(hw, "QiskitRuntimeService", boom, raising=False)


def test_build_and_transpile_cz_count():
    res = ChainReservoir()
    circs = hw.build_circuits(res, windows())
    isa = hw.transpile_for(circs, FakeFez(), seed=11)
    assert len(isa) == 4
    for c in isa:
        ops = c.count_ops()
        assert ops.get("cz", 0) == 20
        assert "swap" not in ops


def test_local_sampler_path_matches_exact_without_noise():
    res = ChainReservoir()
    W = windows(3, seed=1)
    feats = hw.run_sampler(res, W, mode="aer_ideal", shots=20000, seed=3)
    exact = res.feature_matrix(W, mode="exact")[:, :15]
    assert feats.shape == (3, 15)
    assert np.abs(feats - exact).max() < 0.05


def test_fake_backend_noise_changes_features_but_not_wildly():
    res = ChainReservoir()
    W = windows(3, seed=2)
    noisy = hw.run_sampler(res, W, mode="fake_fez", shots=4000, seed=4)
    exact = res.feature_matrix(W, mode="exact")[:, :15]
    diff = np.abs(noisy - exact)
    assert diff.max() > 0.01   # noise is really applied
    assert diff.mean() < 0.3   # but the signal survives


def test_submit_refuses_without_confirm():
    res = ChainReservoir()
    with pytest.raises(PermissionError):
        hw.submit(res, windows(2), backend_name="ibm_fez", shots=2000, confirm=False)


def test_budget_estimate_is_reported():
    est = hw.budget(n_circuits=88, shots=2000)
    assert est["n_circuits"] == 88 and est["total_shots"] == 176000
    assert est["cz_per_circuit"] == 20
    assert 0.0 < est["fidelity_2q"] < 1.0
