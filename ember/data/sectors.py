"""Causal daily sector edges reconstructed from dated hotspot detections."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from ember import schema


def fire_coordinates(fire: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return UTC day offsets and local km coordinates around the day-one centre."""
    dates = fire.rep_date.dt.tz_convert("UTC").dt.normalize()
    days = (dates - dates.min()).dt.days.to_numpy()
    lat0 = float(fire.loc[days == 0, "lat"].mean())
    lon0 = float(fire.loc[days == 0, "lon"].mean())
    xy = np.column_stack(((fire.lon.to_numpy() - lon0) * 111.320 * np.cos(np.deg2rad(lat0)),
                          (fire.lat.to_numpy() - lat0) * 110.574))
    return days, xy, lat0, lon0


def activation_days(xy: np.ndarray, days: np.ndarray, isolation_km: float) -> np.ndarray:
    """Find the first day each point has another detection within the radius.

    Query one day's tree at a time, retaining only points without a neighbour.
    Two nearest results let us exclude the point itself, including duplicates.
    A future detection can only activate a point on that future day or later.
    """
    if not np.isfinite(isolation_km) or isolation_km < 0:
        raise ValueError("isolation_km must be finite and nonnegative")
    activated = np.full(len(days), int(days.max()) + 1, dtype=np.int64)
    pending = np.arange(len(days))
    for day in np.unique(days):
        members = np.flatnonzero(days == day)
        tree = cKDTree(xy[members])
        distance, neighbour = tree.query(xy[pending], k=2)
        first = members[neighbour[:, 0]]
        # When the nearest is self, only the second neighbour can qualify.
        nearest_distance = np.where(first == pending, distance[:, 1], distance[:, 0])
        found = nearest_distance <= isolation_km
        activated[pending[found]] = np.maximum(days[pending[found]], day)
        pending = pending[~found]
        if not len(pending):
            break
    return activated


def build_sectors(hot: pd.DataFrame, *, isolation_km: float = 2.0) -> pd.DataFrame:
    """Build one row per fire, contiguous UTC day and compass sector, without I/O.

    Weather is zero until the first complete, non-placeholder reading; later
    missing days carry previous weather forward, without using future readings.
    """
    tables = []
    weather_columns = ["ws", "wd", "rh", "isi", "bui", "ros"]
    for uid, fire in hot.groupby("uid", sort=True):
        days, xy, _, _ = fire_coordinates(fire)
        n_days = int(days.max()) + 1
        dates = pd.date_range(fire.rep_date.min().tz_convert("UTC").normalize().tz_localize(None),
                              periods=n_days, freq="D")
        year = dates[0].year
        activation = activation_days(xy, days, isolation_km)
        bearing = np.arctan2(xy[:, 0], xy[:, 1])
        sectors = np.rint(np.rad2deg(bearing) / 45.0).astype(int) % schema.N_SECTORS
        radius = np.zeros((n_days, schema.N_SECTORS))
        usable = activation < n_days
        np.maximum.at(radius, (activation[usable], sectors[usable]), np.linalg.norm(xy[usable], axis=1))
        radius = np.maximum.accumulate(radius, axis=0)
        increments = np.vstack((np.zeros((1, schema.N_SECTORS)), np.diff(radius, axis=0)))

        # A complete weather reading excludes placeholder RH=0 and missing fields.
        valid = fire[weather_columns].notna().all(axis=1) & fire.rh.ne(0)
        readings = fire.loc[valid, weather_columns].copy()
        readings.index = days[valid.to_numpy()]
        weather = readings.groupby(level=0).median().reindex(range(n_days)).ffill().fillna(0.0)
        wind = np.zeros((n_days, schema.N_SECTORS))
        for sector in range(schema.N_SECTORS):
            components = pd.Series(schema.wind_toward_sector(readings.ws, readings.wd, sector),
                                   index=readings.index)
            wind[:, sector] = components.groupby(level=0).median().reindex(range(n_days)).ffill().fillna(0.0)
        counts = np.bincount(days, minlength=n_days)
        table = pd.DataFrame({
            "uid": uid, "year": year, "date": dates.repeat(schema.N_SECTORS),
            "day_index": np.repeat(np.arange(1, n_days + 1), schema.N_SECTORS),
            "sector": np.tile(np.arange(schema.N_SECTORS), n_days),
            "r_km": radius.ravel(), "dd_km": increments.ravel(),
            "w_k": wind.ravel(), "n_hot": np.repeat(counts, schema.N_SECTORS),
            "masked": np.repeat(counts == 0, schema.N_SECTORS),
            "split": schema.split_for_year(year),
        })
        for name in ("ws", "rh", "isi", "bui", "ros"):
            table[name] = np.repeat(weather[name].to_numpy(), schema.N_SECTORS)
        for lead, name in schema.LEADS.items():
            target = np.full_like(radius, np.nan)
            if n_days > lead:
                target[:-lead] = radius[lead:] - radius[:-lead]
            table[name] = target.ravel()
        tables.append(table)
    result = pd.concat(tables, ignore_index=True) if tables else pd.DataFrame(columns=list(schema.SECTOR_COLUMNS))
    return result[list(schema.SECTOR_COLUMNS)].astype(schema.SECTOR_COLUMNS)
