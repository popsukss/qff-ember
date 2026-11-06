#!/usr/bin/env python3
"""E12 (exploratory): ring reservoir that entangles neighbouring sectors.

One circuit per fire-day origin instead of one per sector. Sector k owns two
qubits, A_k = 2k (wind toward k) and B_k = 2k+1 (today's advance in k); the
16 qubits form a ring A0-B0-A1-B1-...-B7-A0, so information can pass between
neighbouring sectors. Each day, oldest first:

  A_k: RY(gamma*g0*w_k)  RZ(gamma*g1*isi)  RX(gamma*g2*rh)
  B_k: RY(gamma*g3*dd_k) RZ(gamma*g4*bui)
  CZ on all 16 ring edges
  fixed A: RY(a0) RZ(b0); fixed B: RY(a1) RZ(b1)

Every sector uses the same gains and fixed angles, so the circuit is symmetric
under rotating the sectors. The forecast for sector k reads Z_i, all Z_iZ_j and
X_i with qubit indices taken relative to k (152 features), and one ridge
read-out serves all sectors, as in the per-sector model.

Controls on the same rows: per-sector qrc5, raw ridge (own sector), raw ridge
on all eight sectors (relative order), and an ESN fed all eight sectors
(relative order) with 152 units. Selection and scoring use 2018-2021 train and
2022 validation only; this script never loads 2023.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import combinations
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember.baselines import ESN, raw_window_features
from ember.experiments import _windows, clip_prediction, load_frozen
from ember.metrics import fire_bootstrap_ci, mae
from ember.readout import RidgeReadout, select_lambda
from scripts.e11_qubits import chain_features

N_SECTORS, N = 8, 16
PAIRS = list(combinations(range(N), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(PAIRS)}


def ring_params(seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(0.5, 1.5, size=5), rng.uniform(0, 2 * np.pi, size=(2, 2))


def _ry(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)


def _rz(t):
    z = np.zeros(np.shape(t) + (2, 2), dtype=complex)
    z[..., 0, 0], z[..., 1, 1] = np.exp(-0.5j * t), np.exp(0.5j * t)
    return z


def _rx(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.stack([np.stack([c + 0j, -1j * s], -1), np.stack([-1j * s, c + 0j], -1)], -2)


def _apply(state, mats, q):
    a = np.moveaxis(state, q + 1, 1)
    shape = a.shape
    a = a.reshape(shape[0], 2, -1)
    out = np.einsum("ij,bjk->bik", mats, a) if mats.ndim == 2 else np.einsum("bij,bjk->bik", mats, a)
    return np.moveaxis(out.reshape(shape), 1, q + 1)


def ring_features(F, gamma, seed, batch=48):
    """F: (count, L, 8, 5) scaled inputs [w_k, isi, dd_km, bui, rh]. Returns (count, 152) absolute features."""
    g, fixed_angles = ring_params(seed)
    dim = 2 ** N
    bits = (np.arange(dim)[:, None] >> np.arange(N - 1, -1, -1)) & 1
    signs = 1.0 - 2.0 * bits
    cz = np.prod([np.where(bits[:, q] & bits[:, (q + 1) % N], -1.0, 1.0) for q in range(N)], axis=0)
    fixed = [_rz(fixed_angles[j, 1]) @ _ry(fixed_angles[j, 0]) for j in range(2)]
    out = np.empty((len(F), N + len(PAIRS) + N))
    for start in range(0, len(F), batch):
        Fb = F[start:start + batch]
        B = len(Fb)
        state = np.zeros((B, dim), dtype=complex)
        state[:, 0] = 1.0
        state = state.reshape((B,) + (2,) * N)
        for day in range(Fb.shape[1]):
            x = Fb[:, day]  # (B, 8, 5)
            for k in range(N_SECTORS):
                w, isi, dd, bui, rh = (x[:, k, i] for i in range(5))
                enc_a = _rx(gamma * g[2] * rh) @ _rz(gamma * g[1] * isi) @ _ry(gamma * g[0] * w)
                enc_b = _rz(gamma * g[4] * bui) @ _ry(gamma * g[3] * dd)
                state = _apply(state, enc_a, 2 * k)
                state = _apply(state, enc_b, 2 * k + 1)
            state = (state.reshape(B, dim) * cz).reshape((B,) + (2,) * N)
            for q in range(N):
                state = _apply(state, fixed[q % 2], q)
        flat = state.reshape(B, dim)
        p = flat.real ** 2 + flat.imag ** 2
        z = p @ signs
        zz = np.column_stack([p @ (signs[:, i] * signs[:, j]) for i, j in PAIRS])
        xs = np.empty((B, N))
        for q in range(N):
            t = np.moveaxis(state, q + 1, 1).reshape(B, 2, -1)
            xs[:, q] = 2 * np.sum((t[:, 0].conj() * t[:, 1]).real, axis=1)
        out[start:start + B] = np.column_stack((z, zz, xs))
    return out


def ring_features_parallel(F, gamma, seed, workers):
    if workers <= 1:
        return ring_features(F, gamma, seed)
    chunks = np.array_split(F, workers * 4)
    with ProcessPoolExecutor(workers) as pool:
        return np.concatenate(list(pool.map(partial(ring_features, gamma=gamma, seed=seed), chunks)))


def relative_index(k):
    """Column order of the absolute features as seen from sector k."""
    shift = 2 * k
    q = [(j + shift) % N for j in range(N)]
    zz = [PAIR_INDEX[tuple(sorted((q[i], q[j])))] for i, j in PAIRS]
    return np.array(q + [N + i for i in zz] + [N + len(PAIRS) + j for j in q])


def qiskit_check(gamma, seed):
    """Compare the NumPy ring simulator with Qiskit Statevector on two random windows."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector

    g, fa = ring_params(seed)
    F = np.random.default_rng(5).uniform(0, 1, size=(2, 3, N_SECTORS, 5))
    ours = ring_features(F, gamma, seed)
    worst = 0.0
    for f, mine in zip(F, ours):
        qc = QuantumCircuit(N)
        for x in f:
            for k in range(N_SECTORS):
                a, b = 2 * k, 2 * k + 1
                qc.ry(gamma * g[0] * x[k, 0], a); qc.rz(gamma * g[1] * x[k, 1], a); qc.rx(gamma * g[2] * x[k, 4], a)
                qc.ry(gamma * g[3] * x[k, 2], b); qc.rz(gamma * g[4] * x[k, 3], b)
            for q in range(N):
                qc.cz(q, (q + 1) % N)
            for q in range(N):
                qc.ry(fa[q % 2, 0], q); qc.rz(fa[q % 2, 1], q)
        sv = Statevector.from_instruction(qc)
        terms = [("Z", (i,)) for i in range(N)] + [("Z", p) for p in PAIRS] + [("X", (i,)) for i in range(N)]
        ref = []
        for axis, qs in terms:
            label = ["I"] * N
            for i in qs:
                label[N - 1 - i] = axis
            ref.append(sv.expectation_value(Pauli("".join(label))).real)
        worst = max(worst, float(np.max(np.abs(np.array(ref) - mine))))
    if worst > 1e-9:
        raise RuntimeError(f"ring simulator disagrees with Qiskit: {worst:.3e}")
    return worst


def fire_day_tensor(W, rows):
    """Group sector windows into complete fire-days: (F, L, 8, 5) plus row positions (F, 8)."""
    key = rows["uid"].astype(str) + "|" + rows["date"].astype(str)
    order = pd.DataFrame({"key": key, "sector": rows["sector"].to_numpy(), "pos": np.arange(len(rows))})
    complete = order.groupby("key")["sector"].transform("nunique").eq(N_SECTORS)
    order = order.loc[complete].sort_values(["key", "sector"])
    pos = order["pos"].to_numpy().reshape(-1, N_SECTORS)
    return np.transpose(W[pos], (0, 2, 1, 3)), pos  # (F, L, 8, 5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gammas", type=float, nargs="+", default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    config_path = ROOT / "config.yaml"
    config = yaml.safe_load(config_path.read_text())
    frozen = load_frozen(config_path=config_path, frozen_path=ROOT / "results/frozen_config.json")
    grid, leads, L = config["readout"]["lambda_grid"], config["leads"], frozen["L"]
    gammas = args.gammas or config["experiments"]["e3_gamma"]
    if frozen["input_map"] != ["w_k", "isi", "dd_km", "bui", "rh"]:
        raise SystemExit("ring encoding assumes the frozen input map [w_k, isi, dd_km, bui, rh]")
    print(f"ring vs Qiskit Statevector: max |diff| = {qiskit_check(frozen['gamma'], 7):.2e}", flush=True)

    years = config["splits"]["train"] + config["splits"]["val"]
    df = pd.concat([pd.read_parquet(ROOT / f"data/processed/sectors_{y}.parquet") for y in years],
                   ignore_index=True)
    W, rows = _windows(df, frozen["input_map"], L, config)
    assert not rows["split"].eq("test").any()
    F, pos = fire_day_tensor(W, rows)
    flat_pos = pos.ravel()  # fire-day-major, sector-minor
    sel = rows.iloc[flat_pos].reset_index(drop=True)
    print(f"complete fire-days: {len(F)} ({len(flat_pos)} of {len(rows)} sector rows)", flush=True)

    def expand(abs_feats):
        """(F, d_abs) -> (F*8, d) features relative to each target sector."""
        return np.concatenate([abs_feats[:, relative_index(k)][:, None, :] for k in range(N_SECTORS)],
                              axis=1).reshape(-1, abs_feats.shape[1])

    def relative_inputs(Fx):
        """(F, L, 8, 5) -> (F*8, L, 40): all sectors' inputs, starting from the target sector."""
        parts = [np.roll(Fx, -k, axis=2).reshape(len(Fx), Fx.shape[1], -1)[:, None] for k in range(N_SECTORS)]
        return np.concatenate(parts, axis=1).reshape(-1, Fx.shape[1], N_SECTORS * 5)

    def val_fit(X, h):
        y = sel[f"y{h}"].to_numpy()
        valid = sel[f"y{h}"].notna().to_numpy()
        train = valid & sel["split"].eq("train").to_numpy()
        val = valid & sel["split"].eq("val").to_numpy()
        lam = select_lambda(X[train], y[train], X[val], y[val], grid)
        pred = clip_prediction(RidgeReadout(lam).fit(X[train], y[train]).predict(X[val]))
        return np.abs(y[val] - pred), sel.loc[val, "uid"].to_numpy(), lam

    W_sel = W[flat_pos]
    Rin = relative_inputs(F)
    controls = {
        "qrc5": chain_features(W_sel, frozen["gamma"], frozen["reservoir_seed"], 2048),
        "raw": raw_window_features(W_sel),
        "raw_ring": raw_window_features(Rin),
        "esn_ring152": ESN(152, N_SECTORS * 5, frozen["esn_seed"]).features(Rin),
    }
    errs, uids, records = {}, {}, []
    for name, X in controls.items():
        for h in leads:
            e, uids[h], lam = val_fit(X, h)
            errs[name, h] = e
            records.append({"model": name, "gamma": np.nan, "seed": np.nan, "lead": h,
                            "n_features": X.shape[1], "val_mae": float(e.mean()), "lam": lam})
        print(f"{name}: " + ", ".join(f"+{24*h} h {errs[name, h].mean():.4f}" for h in leads), flush=True)

    timing = {}
    for gamma in gammas:
        for seed in args.seeds:
            t0 = perf_counter()
            X = expand(ring_features_parallel(F, gamma, seed, args.workers))
            timing[f"{gamma:.4f}/{seed}"] = perf_counter() - t0
            name = f"ring_g{gamma:.3f}_s{seed}"
            for h in leads:
                e, _, lam = val_fit(X, h)
                errs[name, h] = e
                records.append({"model": "ring", "gamma": gamma, "seed": seed, "lead": h,
                                "n_features": X.shape[1], "val_mae": float(e.mean()), "lam": lam})
            print(f"{name} ({timing[f'{gamma:.4f}/{seed}']:.0f} s): "
                  + ", ".join(f"+{24*h} h {errs[name, h].mean():.4f}" for h in leads), flush=True)

    table = pd.DataFrame(records)
    ring = table.loc[table.model.eq("ring")]
    best = ring.groupby(["gamma", "seed"]).val_mae.mean().idxmin()
    best_name = f"ring_g{best[0]:.3f}_s{int(best[1])}"
    paired = []
    for ref in controls:
        for h in leads:
            delta = errs[best_name, h] - errs[ref, h]
            lo, hi = fire_bootstrap_ci(pd.DataFrame({"uid": uids[h], "abs_err": delta}), n_boot=args.n_boot, seed=0)
            paired.append({"ring": best_name, "reference": ref, "lead": h,
                           "diff": float(delta.mean()), "lo": lo, "hi": hi})
    paired = pd.DataFrame(paired)
    results = ROOT / "results"
    table.to_csv(results / f"e12_ring_val{args.tag}.csv", index=False)
    paired.to_csv(results / f"e12_ring_paired{args.tag}.csv", index=False)
    (results / f"e12_meta{args.tag}.json").write_text(json.dumps({
        "status": "exploratory; train 2018-2021, val 2022 only; 2023 never loaded",
        "fire_days": int(len(F)), "sector_rows": int(len(flat_pos)), "best": best_name,
        "seconds": timing}, indent=2) + "\n")
    print("\nVal MAE (km):")
    print(table.pivot_table(index=["model", "gamma", "seed"], columns="lead", values="val_mae",
                            dropna=False, sort=False).round(4).to_string())
    print(f"\n{best_name} minus reference, val, paired 95% fire-bootstrap (negative favours ring):")
    print(paired.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
