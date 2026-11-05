"""Write annual fire and forecast-origin counts and print totals by split."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

# Isolated Python omits the repository from sys.path; add only the trusted source root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.schema import SPLIT_YEARS, split_for_year


def main() -> None:
    report = {}
    totals = {split: {"fires": 0, "origin_days": 0} for split in SPLIT_YEARS}
    for year in sorted(year for years in SPLIT_YEARS.values() for year in years):
        table = pd.read_parquet(ROOT / "data" / "processed" / f"sectors_{year}.parquet")
        origins = table.loc[(table.day_index >= 3) & table.y1.notna(), ["uid", "date"]]
        split = split_for_year(year)
        row = {
            "fires": int(table.uid.nunique()),
            "sector_rows": int(len(table)),
            "split": split,
            "origin_days": int(len(origins.drop_duplicates())),
        }
        report[str(year)] = row
        for name in totals[split]:
            totals[split][name] += row[name]

    output = ROOT / "results" / "fire_counts.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{'year':>6} {'split':>6} {'fires':>7} {'sector_rows':>12} {'origin_days':>12}")
    for year, row in report.items():
        print(f"{year:>6} {row['split']:>6} {row['fires']:7d} "
              f"{row['sector_rows']:12d} {row['origin_days']:12d}")
    for split, row in totals.items():
        print(f"{'total':>6} {split:>6} {row['fires']:7d} {'':>12} {row['origin_days']:12d}")


if __name__ == "__main__":
    main()
