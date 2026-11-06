#!/usr/bin/env python3
"""E13 data: extra physical drivers per fire, day and sector (no change to the frozen tables).

Writes data/processed/drivers_<year>.parquet with one row per (uid, day_index, sector):

  up_k    upslope toward the sector (slope x cos(uphill bearing - sector bearing)),
          causal median over joined hotspots in that sector up to that day;
          falls back to the fire-wide median so far, then 0.
  conif_k conifer share (pconif, %) with the same causal rule.
  rw_k    regional wind toward the sector (km/h)  } from ALL hotspots of the year in the
  risi    regional ISI                            } 3x3 block of 1-degree cells around the
  rpcp    regional 24 h precipitation (mm)        } fire centre, that UTC day (any fire)

The fire's own hotspots are removed from its regional block, so the series do
not depend on whether, or where, this fire burned that day (the leak the fire's
own hotspot weather would have). They run to day n_days + 3, so a forecast
origin near the end of a fire still has its next three days.
Missing regional days carry the previous day forward (then 0).

Hotspot rows are reloaded with the same loader and checked equal to the frozen
hotspot table, so day and sector assignment matches build_sectors exactly.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter
import zipfile

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember import schema
from ember.data import hotspots as hotspot_loader
from ember.data.sectors import activation_days, fire_coordinates, joining_days

EXTRA = {"slope": "float64", "aspect": "float64", "pconif": "float64", "pcp": "float64"}
LEAD_MAX = 3


def load_extended(year):
    """Same loader as the frozen pipeline, with four extra raw columns kept."""
    original = dict(schema.HOTSPOT_COLUMNS)
    schema.HOTSPOT_COLUMNS = {**original, **EXTRA}
    try:
        hot = hotspot_loader.load_hotspots(year, ROOT / f"data/raw/cwfis_perimeters/{year}_hotspots.zip",
                                           ROOT / f"data/raw/cwfis_perimeters/{year}_perimeters.zip")
    finally:
        schema.HOTSPOT_COLUMNS = original
    frozen = pd.read_parquet(ROOT / f"data/processed/hotspots_{year}.parquet")
    for name in ("uid", "rep_date", "lat", "lon"):
        if not np.array_equal(hot[name].to_numpy(), frozen[name].to_numpy()):
            raise RuntimeError(f"{year}: reloaded hotspots differ from frozen table in {name}")
    return hot


def _cell_sums(frame, day):
    """Per-row wind vector, ISI, precipitation and count, keyed by UTC day and 1-degree cell."""
    toward = np.deg2rad(frame.wd.to_numpy() + 180.0)
    ws = frame.ws.to_numpy()
    return pd.DataFrame({
        "day": day, "ci": np.floor(frame.lat.to_numpy()).astype(int),
        "cj": np.floor(frame.lon.to_numpy()).astype(int),
        "u": ws * np.sin(toward), "v": ws * np.cos(toward),
        "isi": frame.isi.to_numpy(), "pcp": frame.pcp.to_numpy(), "n": 1.0})


def _complete(frame):
    """Rows with a complete weather reading (RH = 0 marks a placeholder)."""
    frame = frame.dropna(subset=["lat", "lon", "rh", "ws", "wd", "isi", "pcp"])
    return frame[frame.rh.gt(0)]


def regional_grid(year):
    """Per (UTC day, 1-degree cell): sums of wind vector, ISI, precipitation, and counts."""
    cols = ["rep_date", "lat", "lon", "rh", "ws", "wd", "isi", "pcp"]
    parts = []
    with zipfile.ZipFile(ROOT / f"data/raw/cwfis_perimeters/{year}_hotspots.zip") as archive:
        name = [n for n in archive.namelist() if n.lower().endswith(".csv")][0]
        with archive.open(name) as source:
            for chunk in pd.read_csv(source, usecols=cols, chunksize=500_000,
                                     dtype={c: "float64" for c in cols if c != "rep_date"}):
                chunk = _complete(chunk)
                day = pd.to_datetime(chunk.rep_date, utc=True, format="mixed").dt.normalize()
                parts.append(_cell_sums(chunk, day.dt.tz_localize(None).to_numpy()))
    return pd.concat(parts, ignore_index=True).groupby(["day", "ci", "cj"]).sum()


def regional_series(grid, fire, lat0, lon0, dates):
    """Regional daily means around one fire centre, with this fire's own hotspots removed.

    Removing the fire's own detections means the series does not depend on whether,
    or where, this fire burned that day. Days with no other reading carry forward.
    Returns speed, from-direction, ISI, precipitation and the share of missing days.
    """
    ci, cj = int(np.floor(lat0)), int(np.floor(lon0))
    cells = {(ci + a, cj + b) for a in (-1, 0, 1) for b in (-1, 0, 1)}

    def block(table):
        table = table[table.index.get_level_values("day").isin(dates)]
        keep = [(ri, rj) in cells for ri, rj in zip(table.index.get_level_values("ci"),
                                                     table.index.get_level_values("cj"))]
        return table[keep].groupby(level="day").sum().reindex(dates, fill_value=0.0)

    own = _complete(fire)
    own_day = own.rep_date.dt.tz_convert("UTC").dt.normalize().dt.tz_localize(None).to_numpy()
    own_grid = _cell_sums(own, own_day).groupby(["day", "ci", "cj"]).sum()
    daily = block(grid) - block(own_grid)
    if (daily.n < -0.5).any():
        raise RuntimeError("own hotspots not found in the regional grid")
    present = daily.n > 0.5
    mean = daily[["u", "v", "isi", "pcp"]].div(daily.n.where(present), axis=0)
    mean = mean.ffill().fillna(0.0)
    speed = np.hypot(mean.u, mean.v)
    from_dir = (np.rad2deg(np.arctan2(mean.u, mean.v)) + 180.0) % 360.0
    return (speed.to_numpy(), from_dir.to_numpy(), mean.isi.to_numpy(), mean.pcp.to_numpy(),
            float(1 - present.mean()))


def causal_sector_median(values, sectors, joined, n_days):
    """(n_days, 8): per-sector median of values[:, k] over points joined by each day.

    values is (N, 8): column k is the quantity as seen from sector k. A sector with
    no joined point yet uses the fire-wide median of column k so far, then 0.
    """
    out = np.zeros((n_days, schema.N_SECTORS))
    for day in range(n_days):
        seen = joined <= day
        for k in range(schema.N_SECTORS):
            mask = seen & (sectors == k)
            if mask.any():
                out[day, k] = np.median(values[mask, k])
            elif seen.any():
                out[day, k] = np.median(values[seen, k])
    return out


def build_year(year, config):
    hot = load_extended(year)
    grid = regional_grid(year)
    tables = []
    for uid, fire in hot.groupby("uid", sort=True):
        days, xy, lat0, lon0 = fire_coordinates(fire, isolation_km=config["isolation_km"],
                                                connect_km=config["connect_km"])
        n_days = int(days.max()) + 1
        start = fire.rep_date.min().tz_convert("UTC").normalize().tz_localize(None)
        activation = activation_days(xy, days, config["isolation_km"])
        joined = joining_days(xy, days, activation, config["connect_km"])
        bearing = np.arctan2(xy[:, 0], xy[:, 1])
        sectors = np.rint(np.rad2deg(bearing) / 45.0).astype(int) % schema.N_SECTORS

        ok = fire[["slope", "aspect", "pconif"]].notna().all(axis=1).to_numpy()
        sector_bearing = np.arange(schema.N_SECTORS) * 45.0
        # aspect = direction the slope faces (downhill); uphill = aspect + 180.
        comp = fire.slope.to_numpy()[:, None] * np.cos(
            np.deg2rad(fire.aspect.to_numpy()[:, None] + 180.0 - sector_bearing[None, :]))
        up = causal_sector_median(comp[ok], sectors[ok], joined[ok], n_days)
        pconif = np.repeat(fire.pconif.to_numpy()[ok, None], schema.N_SECTORS, axis=1)
        conif = causal_sector_median(pconif, sectors[ok], joined[ok], n_days)

        total = n_days + LEAD_MAX
        dates = pd.date_range(start, periods=total, freq="D")
        speed, from_dir, risi, rpcp, missing = regional_series(grid, fire, lat0, lon0, dates)
        rw = np.column_stack([schema.wind_toward_sector(speed, from_dir, k) for k in range(schema.N_SECTORS)])

        pad = lambda a: np.vstack((a, np.repeat(a[-1:], LEAD_MAX, axis=0)))
        tables.append(pd.DataFrame({
            "uid": uid, "day_index": np.repeat(np.arange(1, total + 1), schema.N_SECTORS),
            "sector": np.tile(np.arange(schema.N_SECTORS), total),
            "up_k": pad(up).ravel(), "conif_k": pad(conif).ravel(), "rw_k": rw.ravel(),
            "risi": np.repeat(risi, schema.N_SECTORS), "rpcp": np.repeat(rpcp, schema.N_SECTORS),
            "in_fire": np.repeat(np.arange(total) < n_days, schema.N_SECTORS),
            "regional_missing": missing}))
    return pd.concat(tables, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--years", type=int, nargs="+", default=[2018, 2019, 2020, 2021, 2022, 2023])
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "config.yaml").read_text())["data"]
    for year in args.years:
        t0 = perf_counter()
        table = build_year(year, config)
        sectors = pd.read_parquet(ROOT / f"data/processed/sectors_{year}.parquet")
        inner = table[table.in_fire].merge(sectors[["uid", "day_index", "sector"]], how="outer", indicator=True)
        if not inner["_merge"].eq("both").all():
            raise RuntimeError(f"{year}: driver rows do not match the sector table")
        table.to_parquet(ROOT / f"data/processed/drivers_{year}.parquet", index=False)
        print(f"year={year} rows={len(table)} fires={table.uid.nunique()} "
              f"up_k med={table.up_k.median():.3f} conif med={table.conif_k.median():.0f} "
              f"rpcp>0 {100 * table.rpcp.gt(0).mean():.0f}% "
              f"regional days missing {100 * table.groupby('uid').regional_missing.first().mean():.0f}% ({perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
