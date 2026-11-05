"""Build a year's assigned hotspot and daily sector Parquet tables."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from time import perf_counter

import yaml

# Isolated Python omits the repository from sys.path; add only the trusted source root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ember.data.hotspots import load_hotspots
from ember.data.sectors import build_sectors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--hotspots", type=Path, help="Raw hotspot ZIP (defaults to the annual archive)")
    parser.add_argument("--perimeters", type=Path, help="Official perimeter ZIP (defaults to the annual archive)")
    parser.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    args = parser.parse_args()
    config = yaml.safe_load((Path(__file__).resolve().parents[1] / "config.yaml").read_text())["data"]
    start = perf_counter()
    hot = load_hotspots(args.year,
                        args.hotspots or f"data/raw/cwfis_perimeters/{args.year}_hotspots.zip",
                        args.perimeters or f"data/raw/cwfis_perimeters/{args.year}_perimeters.zip")
    loaded = perf_counter()
    sectors = build_sectors(hot, isolation_km=config["isolation_km"], connect_km=config["connect_km"],
                            max_advance_km=config["max_advance_km"])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    hot.to_parquet(args.out_dir / f"hotspots_{args.year}.parquet", index=False)
    sectors.to_parquet(args.out_dir / f"sectors_{args.year}.parquet", index=False)
    print(f"year={args.year} fires={hot.uid.nunique()} hotspot_rows={len(hot)} sector_rows={len(sectors)} "
          f"load_seconds={loaded - start:.2f} run_seconds={perf_counter() - start:.2f}")


if __name__ == "__main__":
    main()
