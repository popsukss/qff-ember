"""Causal daily sector edges reconstructed from dated hotspot detections."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ember import schema


def _largest_cluster(xy: np.ndarray, connect_km: float) -> np.ndarray:
    """Indices of the largest single-linkage cluster; ties use input order."""
    if not np.isfinite(connect_km) or connect_km < 0:
        raise ValueError("connect_km must be finite and nonnegative")
    if not len(xy):
        return np.empty(0, dtype=np.int64)
    pairs = cKDTree(xy).query_pairs(connect_km, output_type="ndarray")
    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])),
                       shape=(len(xy), len(xy))).tocsr()
    _, labels = connected_components(graph, directed=False)
    return np.flatnonzero(labels == np.bincount(labels).argmax())


def fire_coordinates(fire: pd.DataFrame, *, isolation_km: float = 2.0,
                     connect_km: float = 2.0) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return UTC offsets and local km coordinates about the day-one seed mean.

    Select the seed using only activated day-one detections. If none activate,
    retain the day-one mean as a coordinate origin, but no points can join.
    """
    dates = fire.rep_date.dt.tz_convert("UTC").dt.normalize()
    days = (dates - dates.min()).dt.days.to_numpy()
    first = fire.loc[days == 0]
    lat0, lon0 = float(first.lat.mean()), float(first.lon.mean())
    first_xy = np.column_stack(((first.lon.to_numpy() - lon0) * 111.320 * np.cos(np.deg2rad(lat0)),
                                (first.lat.to_numpy() - lat0) * 110.574))
    active = np.flatnonzero(activation_days(first_xy, np.zeros(len(first), dtype=int), isolation_km) == 0)
    seed = active[_largest_cluster(first_xy[active], connect_km)]
    if len(seed):
        lat0, lon0 = float(first.iloc[seed].lat.mean()), float(first.iloc[seed].lon.mean())
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


def joining_days(xy: np.ndarray, days: np.ndarray, activation: np.ndarray,
                 connect_km: float) -> np.ndarray:
    """First causal day connected to the largest activated day-one cluster.

    Newly reached points form the next frontier, so a same-day chain also
    reaches older disconnected detections. Nearest-neighbour queries avoid
    materialising the dense all-pairs graph of cumulative satellite detections.
    The sentinel max(day)+1 denotes detections that never joined.
    """
    joined = np.full(len(days), int(days.max()) + 1, dtype=np.int64)
    first = np.flatnonzero(activation == 0)
    seed = first[_largest_cluster(xy[first], connect_km)]
    joined[seed] = 0
    if not len(seed):
        return joined
    for day in np.unique(activation[activation <= days.max()]):
        pending = np.flatnonzero((activation <= day) & (joined > day))
        frontier = np.flatnonzero(joined <= day)
        while len(pending) and len(frontier):
            distance, _ = cKDTree(xy[frontier]).query(xy[pending], k=1)
            reached = distance <= connect_km
            frontier = pending[reached]
            joined[frontier] = day
            pending = pending[~reached]
    return joined


def build_sectors(hot: pd.DataFrame, *, isolation_km: float = 2.0,
                  connect_km: float = 2.0, max_advance_km: float = 30.0) -> pd.DataFrame:
    """Build one row per fire, contiguous UTC day and compass sector, without I/O.

    Weather is zero until the first complete, non-placeholder reading; later
    missing days carry previous weather forward, without using future readings.
    After day one, a detection contributes only if its first join-day distance
    is within max_advance_km of yesterday's sector edge. Rejected detections
    remain linkage points but never contribute to the radius on later days.
    """
    if not np.isfinite(max_advance_km) or max_advance_km < 0:
        raise ValueError("max_advance_km must be finite and nonnegative")
    tables = []
    weather_columns = ["ws", "wd", "rh", "isi", "bui", "ros"]
    for uid, fire in hot.groupby("uid", sort=True):
        days, xy, _, _ = fire_coordinates(fire, isolation_km=isolation_km, connect_km=connect_km)
        n_days = int(days.max()) + 1
        dates = pd.date_range(fire.rep_date.min().tz_convert("UTC").normalize().tz_localize(None),
                              periods=n_days, freq="D")
        year = dates[0].year
        activation = activation_days(xy, days, isolation_km)
        joined = joining_days(xy, days, activation, connect_km)
        bearing = np.arctan2(xy[:, 0], xy[:, 1])
        sectors = np.rint(np.rad2deg(bearing) / 45.0).astype(int) % schema.N_SECTORS
        radius = np.zeros((n_days, schema.N_SECTORS))
        distance = np.linalg.norm(xy, axis=1)
        for day in range(n_days):
            members = np.flatnonzero(joined == day)
            if day:
                radius[day] = radius[day - 1]
                # Judge once, against yesterday's radius, not today's frontier.
                members = members[distance[members] <= radius[day - 1, sectors[members]] + max_advance_km]
            np.maximum.at(radius[day], sectors[members], distance[members])
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
