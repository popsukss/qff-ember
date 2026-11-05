"""Raw history windows and training-fitted feature scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_windows(df, input_map, L, forecast_from_day, max_masked):
    """Build oldest-first windows within each fire and sector."""
    if L < 1:
        raise ValueError("L must be positive")
    columns = list(input_map)
    rows = df.sort_values(["uid", "sector", "day_index"]).reset_index(drop=True)
    if rows.empty:
        return np.empty((0, L, len(columns)), dtype=float), rows

    # Each group contains contiguous days starting at day 1 (the sector contract).
    positions = np.arange(len(rows))
    new_group = rows[["uid", "sector"]].ne(rows[["uid", "sector"]].shift()).any(axis=1)
    starts = np.maximum.accumulate(np.where(new_group, positions, 0))
    lower = np.maximum(starts, positions - L + 1)
    masked_sum = np.concatenate(([0], np.cumsum(rows["masked"].to_numpy(dtype=int))))
    masked_count = masked_sum[positions + 1] - masked_sum[lower]
    keep = (rows["day_index"].to_numpy() >= forecast_from_day) & (masked_count <= max_masked)
    origins = positions[keep]

    history = origins[:, None] - np.arange(L - 1, -1, -1)
    padded = history < starts[origins, None]
    history = np.maximum(history, starts[origins, None])
    W = rows[columns].to_numpy(dtype=float)[history]
    if "dd_km" in columns:
        W[:, :, columns.index("dd_km")][padded] = 0.0
    return W, rows.loc[keep].reset_index(drop=True)


class MinMaxScaler01:
    """Scale each feature using extrema across training rows and steps."""

    def fit(self, W):
        values = np.asarray(W, dtype=float)
        self.lo = values.min(axis=(0, 1))
        self.hi = values.max(axis=(0, 1))
        return self

    def transform(self, W):
        span = self.hi - self.lo
        scaled = (np.asarray(W, dtype=float) - self.lo) / np.where(span == 0, 1.0, span)
        scaled[..., span == 0] = 0.0
        return np.clip(scaled, 0.0, 1.0)
