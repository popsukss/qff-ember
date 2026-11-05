"""Persistence, physical, raw-window and classical reservoir baselines."""

from __future__ import annotations

import numpy as np


def persistence(dd_now, h):
    """Extrapolate the current daily advance for h days."""
    return h * np.asarray(dd_now, dtype=float)


def mean_advance(y_train, n):
    """Predict the training-target mean for every origin."""
    return np.full(n, np.mean(y_train), dtype=float)


def fbp_predict(ros, ws, w_k, h, burn_minutes):
    """Convert downwind rate of spread to advance in kilometres."""
    ws, w_k = np.broadcast_arrays(np.asarray(ws, dtype=float), np.asarray(w_k, dtype=float))
    cosine = np.divide(w_k, ws, out=np.zeros_like(ws), where=ws != 0)
    return h * np.asarray(ros, dtype=float) * burn_minutes / 1000.0 * np.maximum(0.0, cosine)


def fbp_fit_burn_minutes(ros, ws, w_k, y, h):
    """Fit burn duration by least squares through the origin."""
    x = fbp_predict(ros, ws, w_k, h, burn_minutes=1.0)
    denominator = np.sum(x * x)
    if denominator == 0:
        return 0.0
    return float(np.sum(x * np.asarray(y, dtype=float)) / denominator)


def raw_window_features(W):
    """Flatten each window without changing its step-feature order."""
    W = np.asarray(W)
    return W.reshape(W.shape[0], W.shape[1] * W.shape[2])


class ESN:
    """Dense echo-state reservoir, restarted independently for each window."""

    def __init__(self, n_units, n_inputs, seed, spectral_radius=0.9, leak=1.0, input_scale=1.0):
        rng = np.random.default_rng(seed)
        self.W = rng.uniform(-1.0, 1.0, size=(n_units, n_units))
        radius = np.max(np.abs(np.linalg.eigvals(self.W)))
        self.W *= spectral_radius / radius
        self.W_in = rng.uniform(-1.0, 1.0, size=(n_units, n_inputs)) * input_scale
        self.leak = leak

    def features(self, windows):
        """Return final reservoir states, vectorised across windows."""
        windows = np.asarray(windows, dtype=float)
        state = np.zeros((len(windows), self.W.shape[0]), dtype=float)
        for step in range(windows.shape[1]):
            driven = windows[:, step] @ self.W_in.T + state @ self.W.T
            state = (1.0 - self.leak) * state + self.leak * np.tanh(driven)
        return state
