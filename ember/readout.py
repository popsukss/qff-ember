"""Linear ridge readout with an unpenalised intercept."""

from __future__ import annotations

import numpy as np

from ember.metrics import mae


class RidgeReadout:
    """Fit ridge regression after centring features and targets."""

    def __init__(self, lam):
        if lam < 0:
            raise ValueError("lam must be nonnegative")
        self.lam = lam

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        x_mean = X.mean(axis=0)
        y_mean = y.mean(axis=0)
        Xc = X - x_mean
        yc = y - y_mean
        if self.lam == 0:
            self.coef_ = np.linalg.lstsq(Xc, yc, rcond=None)[0]
        else:
            gram = Xc.T @ Xc + self.lam * np.eye(X.shape[1])
            self.coef_ = np.linalg.solve(gram, Xc.T @ yc)
        self.intercept_ = y_mean - x_mean @ self.coef_
        return self

    def predict(self, X):
        return np.asarray(X, dtype=float) @ self.coef_ + self.intercept_


def select_lambda(Xtr, ytr, Xva, yva, grid):
    """Return the first grid value attaining the lowest validation MAE."""
    return min(grid, key=lambda lam: mae(yva, RidgeReadout(lam).fit(Xtr, ytr).predict(Xva)))
