"""Checks for U2: the 5-qubit chain reservoir."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez

from ember.reservoir import ChainReservoir, counts_to_features, feature_names

N, L = 5, 5


def windows(k=5, seed=0):
    return np.random.default_rng(seed).uniform(0, 1, size=(k, L, N))


def test_feature_names_order():
    names = feature_names(N)
    assert names[:5] == ["Z0", "Z1", "Z2", "Z3", "Z4"]
    assert names[5:15] == [f"Z{i}Z{j}" for i, j in itertools.combinations(range(5), 2)]
    assert names[15:] == ["X0", "X1", "X2", "X3", "X4"]


def test_qiskit_matches_numpy():
    res = ChainReservoir(n=N, gamma=np.pi / 2, seed=7)
    for w in windows():
        q = res.features_exact(w)
        p = res.features_numpy(w)
        assert q.shape == (20,)
        np.testing.assert_allclose(q, p, atol=1e-10)


def test_deterministic_seed_and_seed_changes_features():
    w = windows(1)[0]
    a = ChainReservoir(seed=7).features_numpy(w)
    b = ChainReservoir(seed=7).features_numpy(w)
    c = ChainReservoir(seed=8).features_numpy(w)
    np.testing.assert_array_equal(a, b)
    assert np.abs(a - c).max() > 1e-3


def test_input_changes_features():
    res = ChainReservoir()
    w = windows(1)[0]
    w2 = w.copy()
    w2[0, 2] = 1.0 - w2[0, 2]  # oldest day, qubit 2 -> memory must carry it
    assert np.abs(res.features_numpy(w) - res.features_numpy(w2)).max() > 1e-4


def test_inputs_must_be_scaled():
    res = ChainReservoir()
    bad = windows(1)[0]
    bad[0, 0] = 1.5
    with pytest.raises(ValueError):
        res.features_numpy(bad)


def test_feature_matrix_shape():
    res = ChainReservoir()
    X = res.feature_matrix(windows(7), mode="exact")
    assert X.shape == (7, 20)


def test_counts_to_features_endianness():
    # Qiskit key: qubit 0 is the RIGHTMOST bit. '00001' = qubit 0 measured 1.
    f = counts_to_features({"00001": 100}, n=5)
    assert f.shape == (15,)
    np.testing.assert_allclose(f[:5], [-1, 1, 1, 1, 1])
    names = feature_names(5)[:15]
    assert f[names.index("Z0Z1")] == -1
    assert f[names.index("Z1Z2")] == 1


def test_counts_to_features_mixture():
    f = counts_to_features({"00000": 50, "00011": 50}, n=5)
    np.testing.assert_allclose(f[:2], [0, 0])
    assert f[feature_names(5).index("Z0Z1")] == pytest.approx(1.0)


def test_cz_count_on_fez_no_swaps():
    res = ChainReservoir()
    qc = res.measured_circuit(windows(1)[0], basis="Z")
    tq = transpile(qc, backend=FakeFez(), optimization_level=1, seed_transpiler=11)
    ops = tq.count_ops()
    assert ops.get("cz", 0) == L * (N - 1)
    assert "swap" not in ops


def test_aer_sampled_close_to_exact():
    res = ChainReservoir()
    w = windows(1, seed=3)[0]
    exact = res.features_exact(w)[:15]
    sampled = res.features_sampled(w, shots=20000, seed=5)
    assert sampled.shape == (15,)
    assert np.abs(sampled - exact).max() < 0.05


def test_random_entangler_control_differs():
    w = windows(1)[0]
    a = ChainReservoir(seed=7).features_numpy(w)
    b = ChainReservoir(seed=7, entangler="haar", ent_seed=1).features_numpy(w)
    assert np.abs(a - b).max() > 1e-3
    np.testing.assert_allclose(
        ChainReservoir(seed=7, entangler="haar", ent_seed=1).features_exact(w), b, atol=1e-10
    )
