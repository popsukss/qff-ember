#!/usr/bin/env python3
"""Run an offline pilot; submit the real case only with typed confirmation."""

import argparse
from pathlib import Path
import sys

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeFez

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ember import hardware


def _case_windows(path):
    """Load scaled windows with an explicit date-and-sector case index."""
    dates = np.repeat(np.arange("2023-08-10", "2023-08-21", dtype="datetime64[D]"), 8)
    sectors = np.tile(np.arange(8), 11)
    with np.load(path, allow_pickle=False) as case:
        windows = case["windows"]
        if windows.shape != (88, 5, 5):
            raise ValueError("case windows must have shape (88, 5, 5)")
        if (not np.array_equal(case["uid"], np.full(88, 242981))
                or not np.array_equal(case["dates"].astype("datetime64[D]"), dates)
                or not np.array_equal(case["sectors"], sectors)):
            raise ValueError("case must be ZF015, 10–20 Aug 2023, ordered by date then sector")
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--windows", type=Path, help="scaled ZF015 case .npz; required with --submit")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if args.submit and args.windows is None:
        parser.error("--submit requires --windows with the real scaled case archive")

    reservoir = hardware.frozen_reservoir(
        Path(__file__).resolve().parents[1] / "results" / "frozen_config.json")
    windows = np.random.default_rng(args.seed).uniform(0, 1, size=(4, 5, 5))
    circuits = hardware.build_circuits(reservoir, windows)
    isa = hardware.transpile_for(circuits, FakeFez(), seed=args.seed)
    print("Offline dry run: 4 random windows on FakeFez", flush=True)
    for index, circuit in enumerate(isa):
        print(f"circuit {index}: CZ={circuit.count_ops().get('cz', 0)}, depth={circuit.depth()}")
    print(f"budget: {hardware.budget(88, args.shots)}", flush=True)
    noisy = hardware.run_sampler(reservoir, windows, mode="fake_fez", shots=args.shots, seed=args.seed)
    exact = reservoir.feature_matrix(windows, mode="exact")[:, :15]
    error = np.abs(noisy - exact)
    for index, row in enumerate(error):
        print(f"window {index}: noisy-vs-exact feature MAE={row.mean():.6f}, max={row.max():.6f}")
    print(f"overall noisy-vs-exact feature MAE={error.mean():.6f}, max={error.max():.6f}")

    if args.submit:
        case = _case_windows(args.windows)
        reservoir._validate(case, 3)
        if not sys.stdin.isatty():
            parser.error("--submit requires an interactive terminal")
        confirmation = input(f"Submit 88 circuits to {args.backend} at {args.shots} shots? Type SUBMIT: ")
        if confirmation != "SUBMIT":
            print("Submission cancelled.")
            return 1
        job_id = hardware.submit(reservoir, case, args.backend, args.shots, confirm=True)
        print(f"job id: {job_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
