"""Audit CWFIS annual outlines and spatially match dated hotspots.

Run with python -I. Reads ZIPs in place; never extracts archive members.
Dependencies: geopandas, pyogrio, shapely, pandas (and pypdf only for --pdf).
JSON goes to stdout or --output; no downloads or changes to input files.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

import geopandas as gpd
import pandas as pd
import pyogrio
import shapely


def archive_info(path):
    with zipfile.ZipFile(path) as z:
        members = [{"name": m.filename, "bytes": m.file_size} for m in z.infolist()]
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(), "members": members}


def distribution(series):
    return {"min": int(series.min()), "median": float(series.median()),
            "max": int(series.max()),
            "histogram": {str(k): int(v) for k, v in series.value_counts().sort_index().items()}}


def inspect_perimeters(path, args):
    info = archive_info(path)
    shapes = [m["name"] for m in info["members"] if m["name"].endswith(".shp")]
    if len(shapes) != 1:
        raise ValueError(f"Expected one shapefile in {path}, found {shapes}")
    uri = f"/vsizip/{path.resolve()}/{shapes[0]}"
    g = gpd.read_file(uri)
    schema = pyogrio.read_info(uri)
    info.update(crs_wkt=g.crs.to_wkt(), crs_epsg=g.crs.to_epsg(),
                features=len(g), geometry_types=g.geom_type.value_counts().to_dict(),
                invalid_geometries=int((~g.is_valid).sum()),
                fields=[{"name": name, "dtype": str(dtype),
                         "examples": g[name].dropna().drop_duplicates().head(3).tolist(),
                         "nulls": int(g[name].isna().sum())}
                        for name, dtype in zip(schema["fields"], schema["dtypes"])])
    g.columns = g.columns.str.lower()
    for c in ("firstdate", "lastdate"):
        g[c] = pd.to_datetime(g[c], format="mixed", errors="raise", utc=True)
    area_km2 = g.to_crs(6933).geometry.area / 1e6
    first_days = g.groupby("uid").firstdate.nunique()
    last_days = g.groupby("uid").lastdate.apply(lambda x: x.dt.normalize().nunique())
    info.update(uid_count=int(g.uid.nunique()), uid_nulls=int(g.uid.isna().sum()),
                records_per_uid=distribution(g.groupby("uid").size()),
                distinct_firstdates_per_uid=distribution(first_days),
                distinct_lastdate_days_per_uid=distribution(last_days),
                qualifying_uid_count=int(((last_days >= 10) &
                    (g.groupby("uid")["area"].max() / 100 >= 10)).sum()),
                final_area_geometric_ge10_count=int((area_km2 >= 10).sum()),
                final_area_attribute_ge10_count=int((g["area"] / 100 >= 10).sum()),
                activity_span_days=distribution((g.lastdate.dt.normalize() -
                                                 g.firstdate.dt.normalize()).dt.days + 1))
    if "consis_id" in g:
        valid = g.dropna(subset=["consis_id"])
        info["consis_id"] = {"unique_nonnull": int(valid.consis_id.nunique()),
                             "nulls": int(g.consis_id.isna().sum()),
                             "records_per_id": distribution(valid.groupby("consis_id").size()),
                             "lastdate_days_per_id": distribution(valid.groupby("consis_id").lastdate.apply(
                                 lambda x: x.dt.normalize().nunique())),
                             "duplicate_rows": valid[valid.consis_id.duplicated(False)].drop(
                                 columns="geometry").astype(str).to_dict("records")}
    # Azimuthal equidistant centered on town makes distances from town meaningful.
    local_crs = f"+proj=aeqd +lat_0={args.lat} +lon_0={args.lon} +datum=WGS84 +units=m"
    town = gpd.GeoSeries.from_xy([args.lon], [args.lat], crs=4326).to_crs(local_crs).iloc[0]
    distance = g.to_crs(local_crs).boundary.distance(town) / 1000
    selected = g.loc[distance <= args.radius_km].copy()
    selected["distance_km"] = distance.loc[selected.index]
    selected["geometry_area_km2"] = area_km2.loc[selected.index]
    selected["attribute_area_km2"] = selected["area"] / 100
    if args.reference_perimeters and path.name.startswith(args.hotspot_year):
        reference = gpd.read_file(args.reference_perimeters).to_crs(6933)
        matches = []
        for _, row in selected.to_crs(6933).iterrows():
            overlap = reference.geometry.intersection(row.geometry).area
            best = overlap.idxmax()
            matches.append({"uid": int(row.uid), "fireid": reference.loc[best, "fireid"] if overlap.loc[best] > 0 else None,
                            "outline_fraction_inside_reference": float(overlap.loc[best] / row.geometry.area)})
        info["reference_spatial_matches"] = matches
    selected = selected.sort_values("distance_km")
    records = selected.drop(columns="geometry").astype(object).assign(
        firstdate=selected.firstdate.astype(str), lastdate=selected.lastdate.astype(str)
    )
    info["near_town"] = records.where(pd.notna(records), None).to_dict("records")
    return info, selected


def inspect_hotspots(path, targets, args):
    info = archive_info(path)
    members = [m["name"] for m in info["members"] if m["name"].endswith(".csv")]
    if len(members) != 1:
        raise ValueError(f"Expected one CSV in {path}, found {members}")
    counts = {str(int(uid)): Counter() for uid in targets.uid}
    polygons = targets.to_crs(4326)
    bounds = polygons.total_bounds
    examples, nulls, rows = {}, Counter(), 0
    invalid_dates = 0
    date_min, date_max = None, None
    with zipfile.ZipFile(path) as z, z.open(members[0]) as f:
        for chunk in pd.read_csv(f, dtype=str, chunksize=200_000):
            rows += len(chunk)
            for c in chunk:
                nulls[c] += int(chunk[c].isna().sum())
                values = examples.setdefault(c, [])
                if len(values) < 3:
                    for v in chunk[c].dropna().drop_duplicates():
                        if v not in values:
                            values.append(v)
                        if len(values) == 3:
                            break
            dates = pd.to_datetime(chunk.rep_date, errors="coerce", utc=True)
            invalid_dates += int(dates.isna().sum())
            if dates.notna().any():
                lo, hi = dates.min(), dates.max()
                date_min = lo if date_min is None else min(lo, date_min)
                date_max = hi if date_max is None else max(hi, date_max)
            x, y = pd.to_numeric(chunk.lon, errors="coerce"), pd.to_numeric(chunk.lat, errors="coerce")
            keep = x.between(bounds[0], bounds[2]) & y.between(bounds[1], bounds[3])
            x, y = x[keep], y[keep]
            day = dates[keep].dt.strftime("%Y-%m-%d")
            for _, target in polygons.iterrows():
                inside = shapely.intersects_xy(target.geometry, x.to_numpy(), y.to_numpy())
                counts[str(int(target.uid))].update(day[inside].dropna())
    info.update(rows=rows, invalid_dates=invalid_dates, first_detection=str(date_min),
                last_detection=str(date_max), fields=[{"name": k, "examples": v, "nulls": nulls[k]}
                                                     for k, v in examples.items()])
    info["spatial_assignment"] = "Point intersects unbuffered final outline; no fire ID in CSV; UTC days"
    info["targets"] = {}
    for uid, counter in counts.items():
        days = pd.to_datetime(sorted(counter))
        gaps = (days[1:] - days[:-1]).days
        info["targets"][uid] = {"all_year_points": sum(counter.values()),
            "distinct_days": len(counter), "first_day": min(counter) if counter else None,
            "last_day": max(counter) if counter else None,
            "gaps_over_one_day": int((gaps > 1).sum()),
            "max_gap_days": int(max(gaps)) if len(gaps) else None,
            "all_daily_counts": dict(sorted(counter.items())),
            "window_counts": {d.strftime("%Y-%m-%d"): counter[d.strftime("%Y-%m-%d")]
                              for d in pd.date_range(args.start, args.end)}}
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perimeters", nargs="+", type=Path, required=True)
    parser.add_argument("--hotspots", type=Path)
    parser.add_argument("--reference-perimeters", type=Path,
                        help="Optional official GeoJSON with fireid for spatial crosswalk")
    parser.add_argument("--lat", type=float, default=62.454)
    parser.add_argument("--lon", type=float, default=-114.372)
    parser.add_argument("--radius-km", type=float, default=60)
    parser.add_argument("--hotspot-year", default="2023")
    parser.add_argument("--start", default="2023-08-10")
    parser.add_argument("--end", default="2023-08-20")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    result = {"method": {"town": [args.lon, args.lat], "radius_km": args.radius_km,
                          "area": "EPSG:6933 equal-area polygon area / 1e6; AREA / 100 separately; qualification uses AREA",
                          "distance": "polygon boundary to town in town-centered WGS84 AEQD, km"},
              "versions": {m.__name__: m.__version__ for m in (gpd, pd, pyogrio, shapely)},
              "perimeters": []}
    hotspot_targets = None
    for path in args.perimeters:
        info, targets = inspect_perimeters(path, args)
        result["perimeters"].append(info)
        if path.name.startswith(args.hotspot_year):
            hotspot_targets = targets
    if args.hotspots:
        if hotspot_targets is None or hotspot_targets.empty:
            parser.error("Need matching --hotspot-year perimeters with targets within --radius-km")
        result["hotspots"] = inspect_hotspots(args.hotspots, hotspot_targets, args)
    if args.pdf:
        from pypdf import PdfReader
        result["pdf_text"] = {str(p): "\n".join(page.extract_text() or "" for page in PdfReader(p).pages)
                              for p in args.pdf}
    output = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(output)
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
