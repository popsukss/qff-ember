"""Compare final radial outlines and buffered detections to CWFIS perimeters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from shapely.geometry import Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ember.data.hotspots import load_perimeters
from ember.data.sectors import activation_days, fire_coordinates


def iou(a, b) -> float:
    """Area intersection over union, including degenerate reconstructed outlines."""
    union_area = a.union(b).area
    return float(a.intersection(b).area / union_area) if union_area else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, default=2023)
    parser.add_argument("--hotspots", type=Path, help="Processed hotspot Parquet")
    parser.add_argument("--sectors", type=Path, help="Processed sector Parquet")
    parser.add_argument("--perimeters", type=Path, help="Official perimeter ZIP")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())["data"]
    start = perf_counter()
    hot = pd.read_parquet(args.hotspots or f"data/processed/hotspots_{args.year}.parquet")
    sectors = pd.read_parquet(args.sectors or f"data/processed/sectors_{args.year}.parquet")
    polygons = load_perimeters(args.perimeters or f"data/raw/cwfis_perimeters/{args.year}_perimeters.zip")
    # EPSG:6933 is equal-area and in metres, matching the perimeter audit.
    official = polygons.to_crs(6933).set_index("uid").geometry
    per_fire = []
    for uid, fire in hot.groupby("uid", sort=True):
        days, xy, lat0, lon0 = fire_coordinates(fire, isolation_km=config["isolation_km"],
                                               connect_km=config["connect_km"])
        final = sectors[sectors.uid == uid].sort_values(["date", "sector"]).tail(8)
        bearings = np.deg2rad(final.sector.to_numpy() * 45.0)
        radius = final.r_km.to_numpy()
        lon = lon0 + radius * np.sin(bearings) / (111.320 * np.cos(np.deg2rad(lat0)))
        lat = lat0 + radius * np.cos(bearings) / 110.574
        radial = Polygon(np.column_stack((lon, lat)))
        if not radial.is_valid:
            radial = radial.buffer(0)
        radial = gpd.GeoSeries([radial], crs=4326).to_crs(6933).iloc[0]
        activated = activation_days(xy, days, config["isolation_km"]) <= days.max()
        # Buffer in a fire-centred equidistant metric CRS so 375 m has local meaning.
        local_crs = f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
        points = gpd.GeoSeries(gpd.points_from_xy(fire.loc[activated, "lon"], fire.loc[activated, "lat"]),
                              crs=4326).to_crs(local_crs)
        buffered = points.buffer(375).union_all()
        buffered = gpd.GeoSeries([buffered], crs=local_crs).to_crs(6933).iloc[0]
        target = official.loc[uid]
        per_fire.append({"uid": int(uid), "iou": iou(radial, target),
                         "iou_buffered": iou(buffered, target),
                         "area_rebuilt_km2": float(radial.area / 1e6),
                         "area_official_km2": float(target.area / 1e6)})
        if len(per_fire) % 50 == 0:
            print(f"checked_fires={len(per_fire)} run_seconds={perf_counter() - start:.2f}", flush=True)
    if not per_fire:
        raise ValueError("No kept fires to evaluate")
    report = {"median_iou": float(np.median([f["iou"] for f in per_fire])),
              "median_iou_buffered": float(np.median([f["iou_buffered"] for f in per_fire])),
              "per_fire": per_fire}
    output = args.output or Path(f"results/e10_{args.year}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"fires={len(per_fire)} median_iou={report['median_iou']:.6f} "
          f"median_iou_buffered={report['median_iou_buffered']:.6f} run_seconds={perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
