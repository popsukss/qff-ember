"""Load dated CWFIS hotspots and assign them to eligible annual fire outlines."""

from __future__ import annotations

from pathlib import Path
import zipfile

import geopandas as gpd
import pandas as pd

from ember import schema


def load_perimeters(path: str | Path) -> gpd.GeoDataFrame:
    """Read a perimeter ZIP in place, normalising historical field names and dates."""
    perimeters = gpd.read_file(f"zip://{Path(path).resolve()}")
    perimeters.columns = perimeters.columns.str.lower()
    for name in ("firstdate", "lastdate"):
        perimeters[name] = pd.to_datetime(perimeters[name], format="mixed", utc=True)
    return perimeters.to_crs(4326)


def load_hotspots(year: int, hotspots: str | Path, perimeters: str | Path,
                  *, chunksize: int = 200_000) -> pd.DataFrame:
    """Load one year; keep provider area >=10 km² and >=10 UTC detection days.

    Coordinates are assumed WGS84. Boundary points count as inside. If polygons
    overlap, the lowest temporally eligible uid owns the detection.
    """
    polygons = load_perimeters(perimeters).sort_values("uid")
    polygons = polygons[["uid", "firstdate", "lastdate", "geometry"]].copy().assign(
        area_km2=polygons["area"] / 100)
    bounds = polygons.total_bounds
    columns = [name for name in schema.HOTSPOT_COLUMNS if name != "uid"]
    numeric = {name: "float64" for name in columns if name != "rep_date"}
    tables = []
    with zipfile.ZipFile(hotspots) as archive:
        csvs = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(csvs) != 1:
            raise ValueError(f"Expected one hotspot CSV, found {csvs}")
        with archive.open(csvs[0]) as source:
            for chunk in pd.read_csv(source, usecols=columns, dtype=numeric, chunksize=chunksize):
                chunk = chunk[chunk.lon.between(bounds[0], bounds[2]) &
                              chunk.lat.between(bounds[1], bounds[3])].copy()
                if chunk.empty:
                    continue
                chunk["rep_date"] = pd.to_datetime(chunk.rep_date, utc=True, format="mixed")
                chunk = chunk[chunk.rep_date.dt.year == year]
                points = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(chunk.lon, chunk.lat), crs=4326)
                joined = gpd.sjoin(points, polygons, how="inner", predicate="intersects")
                joined = joined[(joined.rep_date >= joined.firstdate - pd.Timedelta(days=1)) &
                                (joined.rep_date <= joined.lastdate + pd.Timedelta(days=1))]
                joined = joined.sort_values("uid").loc[lambda df: ~df.index.duplicated()]
                tables.append(pd.DataFrame(joined[list(schema.HOTSPOT_COLUMNS)]))
    if not tables:
        return pd.DataFrame(columns=list(schema.HOTSPOT_COLUMNS)).astype(schema.HOTSPOT_COLUMNS)
    hot = pd.concat(tables, ignore_index=True)
    detection_days = hot.rep_date.dt.normalize().groupby(hot.uid).nunique()
    eligible = polygons.loc[polygons.area_km2 >= 10, "uid"]
    kept = detection_days.index[(detection_days >= 10) & detection_days.index.isin(eligible)]
    hot = hot[hot.uid.isin(kept)].sort_values(["uid", "rep_date"], kind="stable").reset_index(drop=True)
    return hot.astype(schema.HOTSPOT_COLUMNS)
