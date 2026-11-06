#!/usr/bin/env python3
"""E11 (exploratory, post-hoc): does a larger chain reservoir help?

The frozen study (config.yaml, results/frozen_config.json) is not changed.
Each register size n reuses the frozen L, gamma, input map and seed. The five
inputs are tiled across qubits (qubit i reads input i mod 5); every copy gets
its own random gain, so copies differ. Features: Z_i, all Z_iZ_j, X_i.

A gate-by-gate NumPy simulator is used so that large n does not need the dense
2^n x 2^n layer. It is checked against ChainReservoir for small n.

Selection is on 2022 only. 2023 scores are reported for completeness and are
post-hoc: the locked test result in results/metrics.csv stays the headline.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.baselines import ESN, persistence, raw_window_features
from ember.experiments import _mean_iou, _windows, clip_prediction, load_frozen, mae_growing
from ember.metrics import fire_bootstrap_ci, mae, skill
from ember.readout import RidgeReadout, select_lambda
from ember.reservoir import ChainReservoir, feature_names


def reservoir_params(n, seed):
    """Same random draws, in the same order, as ChainReservoir.__init__."""
    rng = np.random.default_rng(seed)
    gains = rng.uniform(0.5, 1.5, size=n)
    w_angles = rng.uniform(0, 2 * np.pi, size=(n, 2))
    return gains, w_angles


def _apply_1q(state, mats, axis):
    """Apply a 2x2 gate (shared (2,2) or per-window (B,2,2)) on one qubit axis."""
    a = np.moveaxis(state, axis, 1)  # (B, 2, rest...)
    shape = a.shape
    a = a.reshape(shape[0], 2, -1)
    if mats.ndim == 2:
        out = np.einsum("ij,bjk->bik", mats, a)
    else:
        out = np.einsum("bij,bjk->bik", mats, a)
    return np.moveaxis(out.reshape(shape), 1, axis)


def chain_features(W, gamma, seed, batch):
    """Exact noiseless features of the CZ-chain reservoir, any n."""
    count, L, n = W.shape
    gains, w_angles = reservoir_params(n, seed)
    dim = 2 ** n
    idx = np.arange(dim)
    bits = (idx[:, None] >> np.arange(n - 1, -1, -1)) & 1  # qubit 0 = most significant
    signs = 1.0 - 2.0 * bits
    cz_diag = np.prod([np.where(bits[:, i] & bits[:, i + 1], -1.0, 1.0)
                       for i in range(n - 1)], axis=0) if n > 1 else np.ones(dim)
    fixed = []
    for a, b in w_angles:
        c, s = np.cos(a / 2), np.sin(a / 2)
        ry = np.array([[c, -s], [s, c]])
        fixed.append(np.diag(np.exp([-0.5j * b, 0.5j * b])) @ ry)
    pairs = list(combinations(range(n), 2))
    out = np.empty((count, len(feature_names(n))))
    for start in range(0, count, batch):
        Wb = W[start:start + batch]
        B = len(Wb)
        state = np.zeros((B, dim), dtype=complex)
        state[:, 0] = 1.0
        state = state.reshape((B,) + (2,) * n)
        for day in range(L):
            for i in range(n):
                theta = gamma * gains[i] * Wb[:, day, i]
                c, s = np.cos(theta / 2), np.sin(theta / 2)
                ry = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)
                state = _apply_1q(state, ry, i + 1)
            state = (state.reshape(B, dim) * cz_diag).reshape((B,) + (2,) * n)
            for i in range(n):
                state = _apply_1q(state, fixed[i], i + 1)
        flat = state.reshape(B, dim)
        p = flat.real ** 2 + flat.imag ** 2
        z = p @ signs
        zz = np.column_stack([p @ (signs[:, i] * signs[:, j]) for i, j in pairs]) if pairs else np.empty((B, 0))
        x = np.empty((B, n))
        for i in range(n):
            t = np.moveaxis(state, i + 1, 1).reshape(B, 2, -1)
            x[:, i] = 2 * np.sum((t[:, 0].conj() * t[:, 1]).real, axis=1)
        out[start:start + B] = np.column_stack((z, zz, x))
    return out


def self_check(gamma, seed):
    rng = np.random.default_rng(123)
    worst = 0.0
    for n in (2, 3, 5, 7):
        W = rng.uniform(0, 1, size=(40, 5, n))
        ref = ChainReservoir(n=n, gamma=gamma, seed=seed).feature_matrix(W, mode="numpy")
        worst = max(worst, float(np.max(np.abs(chain_features(W, gamma, seed, 16) - ref))))
    if worst > 1e-10:
        raise RuntimeError(f"gate-by-gate simulator disagrees with ChainReservoir: {worst:.3e}")
    return worst


def score(name, X, rows, leads, splits, n_boot):
    records = []
    for h in leads:
        y = rows[f"y{h}"].to_numpy()
        valid = rows[f"y{h}"].notna().to_numpy()
        train = valid & rows["split"].eq("train").to_numpy()
        val = valid & rows["split"].eq("val").to_numpy()
        lam = select_lambda(X[train], y[train], X[val], y[val], LAMBDAS)
        readout = RidgeReadout(lam).fit(X[train], y[train])
        for split in splits:
            keep = valid & rows["split"].eq(split).to_numpy()
            sel = rows.loc[keep]
            pred = clip_prediction(readout.predict(X[keep]))
            errors = pd.DataFrame({"uid": sel.uid.to_numpy(), "abs_err": np.abs(y[keep] - pred)})
            lo, hi = fire_bootstrap_ci(errors, n_boot=n_boot, seed=0)
            err = mae(y[keep], pred)
            ref = mae(y[keep], clip_prediction(persistence(sel.dd_km, h)))
            records.append({"model": name, "split": split, "lead": h, "n_features": X.shape[1],
                            "mae": err, "mae_lo": lo, "mae_hi": hi, "skill": skill(err, ref),
                            "mae_growing": mae_growing(y[keep], pred),
                            "iou": _mean_iou(sel, y[keep], pred), "lam": lam})
    return records


def main():
    global LAMBDAS
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--qubits", type=int, nargs="+", default=[5, 6, 8, 10, 12])
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    LAMBDAS = config["readout"]["lambda_grid"]
    gamma, seed, L, input_map = frozen["gamma"], frozen["reservoir_seed"], frozen["L"], frozen["input_map"]
    print(f"frozen: L={L} gamma={gamma:.4f} seed={seed} map={input_map}", flush=True)
    print(f"simulator check vs ChainReservoir: max |diff| = {self_check(gamma, seed):.2e}", flush=True)

    years = config["splits"]["train"] + config["splits"]["val"] + config["splits"]["test"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                   ignore_index=True)
    W5, rows = _windows(df, input_map, L, config)
    print(f"windows: {len(W5)}", flush=True)

    records, timing = [], {}
    records += score("raw", raw_window_features(W5), rows, config["leads"], ["val", "test"], args.n_boot)
    for n in args.qubits:
        t0 = perf_counter()
        Wn = W5[:, :, [i % W5.shape[2] for i in range(n)]]
        X = chain_features(Wn, gamma, seed, max(1, args.batch * 2 ** 10 // 2 ** max(n, 10)))
        timing[n] = perf_counter() - t0
        records += score(f"qrc{n}", X, rows, config["leads"], ["val", "test"], args.n_boot)
        esn = ESN(X.shape[1], W5.shape[2], frozen["esn_seed"]).features(W5)
        records += score(f"esn{X.shape[1]}", esn, rows, config["leads"], ["val", "test"], args.n_boot)
        print(f"n={n}: {X.shape[1]} features, simulate {timing[n]:.1f} s", flush=True)

    out = pd.DataFrame(records)
    results = ROOT / "results"
    out.to_csv(results / "e11_qubits.csv", index=False)
    (results / "e11_meta.json").write_text(json.dumps({
        "status": "exploratory, post-hoc; selection on val (2022) only; locked test result unchanged",
        "encoding": "qubit i reads input i mod 5; own random gain per qubit (ChainReservoir draws)",
        "frozen": {"L": L, "gamma": gamma, "seed": seed, "input_map": input_map},
        "qubits": args.qubits, "simulate_seconds": timing}, indent=2) + "\n")
    show = out.pivot_table(index="model", columns=["split", "lead"], values="mae", sort=False)
    print(show.round(4).to_string())


if __name__ == "__main__":
    main()
