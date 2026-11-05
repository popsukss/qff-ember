"""Forecast errors, outline overlap and fire-level uncertainty."""

from __future__ import annotations

import numpy as np
from shapely.geometry import Polygon


def mae(y, yhat):
    """Mean absolute error over the supplied targets."""
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def skill(mae_model, mae_ref):
    """Fractional MAE improvement relative to a reference."""
    return 1.0 - mae_model / mae_ref


def outline_iou(r_true, r_pred):
    """Intersection over union for eight clockwise radii from north."""
    r_true = np.asarray(r_true, dtype=float)
    r_pred = np.asarray(r_pred, dtype=float)
    theta = np.deg2rad(np.arange(8) * 45.0)
    directions = np.column_stack((np.sin(theta), np.cos(theta)))
    polygons = []
    for radii in (r_true, r_pred):
        polygon = Polygon(radii[:, None] * directions)
        # Zero radii can create touching or degenerate polygon edges.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        polygons.append(polygon)
    union_area = polygons[0].union(polygons[1]).area
    if union_area == 0:
        return float(np.all(r_true == 0) and np.all(r_pred == 0))
    return float(polygons[0].intersection(polygons[1]).area / union_area)


def fire_bootstrap_ci(errs, n_boot=1000, seed=0, alpha=0.05):
    """Bootstrap fires with replacement, pooling all rows of each draw."""
    grouped = errs.groupby("uid")["abs_err"].agg(["sum", "count"])
    if grouped.empty:
        raise ValueError("errs must contain at least one fire")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(grouped), size=(n_boot, len(grouped)))
    sums = grouped["sum"].to_numpy()[draws].sum(axis=1)
    counts = grouped["count"].to_numpy()[draws].sum(axis=1)
    lo, hi = np.quantile(sums / counts, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
