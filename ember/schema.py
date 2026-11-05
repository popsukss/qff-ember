"""Shared data contracts for Ember.

Every module reads and writes tables with exactly these columns. Changing this
file changes every unit, so edit it only together with all callers.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- geometry
N_SECTORS = 8
SECTOR_BEARINGS_DEG = tuple(45.0 * k for k in range(N_SECTORS))  # 0=N, 1=NE, ..., 7=NW (clockwise from north)
SECTOR_HALF_WIDTH_DEG = 22.5

# ---------------------------------------------------------------- splits (by fire year)
SPLIT_YEARS = {
    "train": (2018, 2019, 2020, 2021),
    "val": (2022,),
    "test": (2023,),
}


def split_for_year(year: int) -> str:
    for name, years in SPLIT_YEARS.items():
        if int(year) in years:
            return name
    raise ValueError(f"year {year} is in no split")


# ---------------------------------------------------------------- hotspot points
# One row = one satellite detection assigned to one kept fire (uid).
HOTSPOT_COLUMNS = {
    "uid": "int64",          # CWFIS final-perimeter polygon id
    "rep_date": "datetime64[ns, UTC]",
    "lat": "float64",
    "lon": "float64",
    "ws": "float64",         # local-noon wind speed, km/h
    "wd": "float64",         # local-noon wind direction, degrees, direction the wind blows FROM
    "rh": "float64",         # local-noon relative humidity, %
    "isi": "float64",
    "bui": "float64",
    "ros": "float64",        # modelled FBP rate of spread, m/min
}

# ---------------------------------------------------------------- sector table
# One row = one fire (uid) x one UTC calendar day x one sector.
# Days run contiguously from the fire's first to last detection day.
SECTOR_COLUMNS = {
    "uid": "int64",
    "year": "int64",
    "date": "datetime64[ns]",  # UTC calendar day, tz-naive, normalised to midnight
    "day_index": "int64",      # 1 = first detection day
    "sector": "int64",         # 0..7
    "r_km": "float64",         # edge distance from the fixed centre, using detections up to end of `date`
    "dd_km": "float64",        # r_km(t) - r_km(t-1); 0 on day 1
    "ws": "float64",           # daily median over the fire's detections (placeholders dropped, gaps carried)
    "w_k": "float64",          # wind component toward this sector, km/h
    "isi": "float64",
    "bui": "float64",
    "rh": "float64",
    "ros": "float64",
    "n_hot": "int64",          # detections of this fire on `date` (all sectors)
    "masked": "bool",          # True if n_hot == 0 (weather carried from the last observed day)
    "y1": "float64",           # r_km(t+1) - r_km(t); NaN if t+1 is past the last day
    "y3": "float64",           # r_km(t+3) - r_km(t); NaN if t+3 is past the last day
    "split": "object",         # train / val / test, from `year`
}

# Columns a model may read at forecast time t (everything known by the end of day t).
INPUT_COLUMNS = ("r_km", "dd_km", "ws", "w_k", "isi", "bui", "rh", "ros", "n_hot", "masked")
TARGET_COLUMNS = ("y1", "y3")
LEADS = {1: "y1", 3: "y3"}


def wind_toward_sector(ws: np.ndarray, wd: np.ndarray, sector: np.ndarray) -> np.ndarray:
    """Component of the wind blowing toward each sector's bearing (km/h).

    wd is where the wind comes FROM, so it blows toward wd + 180.
    """
    bearing = np.deg2rad(np.asarray(sector, dtype=float) * 45.0)
    toward = np.deg2rad(np.asarray(wd, dtype=float) + 180.0)
    return np.asarray(ws, dtype=float) * np.cos(bearing - toward)


def validate_sectors(df: pd.DataFrame) -> list[str]:
    """Return a list of contract violations (empty list = valid)."""
    errors: list[str] = []
    missing = [c for c in SECTOR_COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    extra = [c for c in df.columns if c not in SECTOR_COLUMNS]
    if extra:
        errors.append(f"unexpected columns: {extra}")
    if df.duplicated(["uid", "date", "sector"]).any():
        errors.append("duplicate (uid, date, sector) rows")
    if not df["sector"].between(0, N_SECTORS - 1).all():
        errors.append("sector outside 0..7")
    per_day = df.groupby(["uid", "date"])["sector"].nunique()
    if (per_day != N_SECTORS).any():
        errors.append("some (uid, date) do not have all 8 sectors")
    if (df["dd_km"] < -1e-9).any():
        errors.append("negative dd_km (edge shrank)")
    if df[list(INPUT_COLUMNS)].isna().any().any():
        cols = [c for c in INPUT_COLUMNS if df[c].isna().any()]
        errors.append(f"NaN in input columns {cols}")
    for name in df["split"].unique():
        if name not in SPLIT_YEARS:
            errors.append(f"unknown split {name!r}")
    bad_split = df["split"] != df["year"].map(split_for_year)
    if bad_split.any():
        errors.append("split does not match year")
    if df.groupby("uid")["split"].nunique().max() > 1:
        errors.append("a fire appears in more than one split")
    for uid, g in df.groupby("uid"):
        days = np.sort(g["date"].unique())
        if len(days) > 1:
            gaps = np.diff(days).astype("timedelta64[D]").astype(int)
            if (gaps != 1).any():
                errors.append(f"uid {uid}: dates not contiguous")
                break
        r = g.sort_values(["sector", "date"])
        if (r.groupby("sector")["r_km"].diff().dropna() < -1e-9).any():
            errors.append(f"uid {uid}: r_km decreases")
            break
    return errors


def isclose(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)
