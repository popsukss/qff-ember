#!/usr/bin/env python3
"""Time NumPy feature extraction on 100,000 five-day windows."""

from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ember.reservoir import ChainReservoir


def main():
    windows = np.random.default_rng(0).uniform(0, 1, size=(100_000, 5, 5))
    reservoir = ChainReservoir()
    start = perf_counter()
    features = reservoir.feature_matrix(windows, mode="numpy")
    elapsed = perf_counter() - start
    print(f'feature_matrix(mode="numpy"): {len(windows):,} windows, '
          f"shape={features.shape}, {elapsed:.3f} s")
    return 0 if elapsed < 60 else 1


if __name__ == "__main__":
    sys.exit(main())
