#!/usr/bin/env python3
"""E11c (exploratory): offline hardware dry run of an n-qubit reservoir on ZF015.

Same case and protocol as E6 (88 Yellowknife windows, Z-basis only, so the
read-out uses Z and ZZ features), but with an n-qubit chain whose five inputs
are tiled across qubits (qubit i reads input i mod 5). The Z-only ridge
read-out is fit on train with the penalty chosen on val (2022), then applied
unchanged to exact and FakeFez features. Never contacts IBM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember import hardware
from ember.experiments import _windows, clip_prediction, load_frozen
from ember.readout import RidgeReadout, select_lambda
from ember.reservoir import ChainReservoir
from scripts.e11_qubits import chain_features
from scripts.e6_hardware import load_case


def tile(W, n):
    return W[:, :, [i % W.shape[2] for i in range(n)]]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--shots", type=int, default=10000)
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    gamma, seed, L, input_map = frozen["gamma"], frozen["reservoir_seed"], frozen["L"], frozen["input_map"]
    grid = config["readout"]["lambda_grid"]

    years = config["splits"]["train"] + config["splits"]["val"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                   ignore_index=True)
    W5, rows = _windows(df, input_map, L, config)
    case_windows, case_rows = load_case()

    scores, meta = [], {"status": "exploratory offline dry run; no IBM contact", "shots": args.shots,
                        "fake_fez_seed": 0, "qubits": {}}
    for n in args.qubits:
        t0 = perf_counter()
        z_cols = n + n * (n - 1) // 2
        X = chain_features(tile(W5, n), gamma, seed, max(1, 2048 * 2 ** 10 // 2 ** max(n, 10)))[:, :z_cols]
        reservoir = ChainReservoir(n=n, gamma=gamma, seed=seed)
        cw = tile(case_windows, n)
        exact = reservoir.feature_matrix(cw, mode="numpy")[:, :z_cols]
        circuits = hardware.build_circuits(reservoir, cw)
        from qiskit_ibm_runtime.fake_provider import FakeFez
        isa = hardware.transpile_for(circuits, FakeFez(), seed=11)  # raises on SWAPs / extra CZ
        fake = hardware.run_sampler(reservoir, cw, mode="fake_fez", shots=args.shots, seed=0)
        feats = {"exact": exact, "fake_fez": fake}
        b = hardware.budget(len(cw), args.shots)
        cz = (n - 1) * L
        meta["qubits"][n] = {"z_features": z_cols, "cz_per_circuit": cz,
                             "fidelity_2q": (1 - b["eps_2q"]) ** cz, "eps_2q": b["eps_2q"],
                             "isa_depth_max": max(c.depth() for c in isa),
                             "feature_mae_fake_vs_exact": float(np.abs(fake - exact).mean())}
        for h in config["leads"]:
            y = rows[f"y{h}"].to_numpy()
            valid = rows[f"y{h}"].notna().to_numpy()
            train = valid & rows["split"].eq("train").to_numpy()
            val = valid & rows["split"].eq("val").to_numpy()
            lam = select_lambda(X[train], y[train], X[val], y[val], grid)
            readout = RidgeReadout(lam).fit(X[train], y[train])
            keep = case_rows[f"y{h}"].notna().to_numpy()
            yc = case_rows.loc[keep, f"y{h}"].to_numpy()
            for source, F in feats.items():
                pred = clip_prediction(readout.predict(F[keep]))
                scores.append({"n": n, "lead": h, "source": source, "lam": lam,
                               "mae": float(np.abs(yc - pred).mean()), "n_rows": int(keep.sum())})
            pers = h * case_rows.loc[keep, "dd_km"].to_numpy()
            scores.append({"n": n, "lead": h, "source": "persistence", "lam": np.nan,
                           "mae": float(np.abs(yc - pers).mean()), "n_rows": int(keep.sum())})
        meta["qubits"][n]["seconds"] = perf_counter() - t0
        print(f"n={n}: {json.dumps(meta['qubits'][n])}", flush=True)

    scores = pd.DataFrame(scores)
    scores.to_csv(ROOT / "results/e11c_hw_dryrun.csv", index=False)
    (ROOT / "results/e11c_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(scores.pivot_table(index=["n", "source"], columns="lead", values="mae", sort=False).round(4).to_string())


if __name__ == "__main__":
    main()
