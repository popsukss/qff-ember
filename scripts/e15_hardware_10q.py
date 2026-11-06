#!/usr/bin/env python3
"""E15 (exploratory): the 10-qubit reservoir (+72 h model) on ibm_fez for Yellowknife.

Same protocol as E6, scaled up: 88 ZF015 windows (10-20 Aug 2023), Z-basis only, so
the read-out uses the 55 Z/ZZ features. The Z-only ridge read-out is fit on exact
features of the 2018-21 training windows, penalty chosen on 2022, then applied
unchanged to exact, FakeFez and hardware features. Inputs come from
results/qubit_windows.npz (scripts/export_qubit_windows.py). Post-hoc: the 10-qubit
version uses observed later weather (a perfect forecast).

  --dry-run           FakeFez on all 88 circuits (offline)
  --submit --yes      one job on ibm_fez; logged to hardware/jobs_10q.json
  --fetch JOB_ID      score a finished job; writes results/e15_10q_hardware.json
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ember import hardware
from ember.readout import RidgeReadout, select_lambda
from ember.reservoir import ChainReservoir

LEAD = 3
N = 10
NZ = N * (N + 1) // 2  # 10 Z + 45 ZZ
GRID = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
JOBS = ROOT / "hardware" / "jobs_10q.json"
OPTIONS = {"dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
           "twirling": {"enable_gates": True}}


def setup():
    frozen = json.loads((ROOT / "results" / "frozen_config.json").read_text())
    with np.load(ROOT / "results" / "qubit_windows.npz") as z:
        daily, static, part = z["daily"].astype(float), z["static"].astype(float), z["part"]
        y = z[f"y{LEAD}"].astype(float)
        dates, sectors = z["dates"], z["sectors"]
    W = np.concatenate([daily, np.repeat(static[:, None, 3:], daily.shape[1], axis=1)], axis=2)
    res = ChainReservoir(n=N, gamma=frozen["gamma"], seed=frozen["reservoir_seed"])
    X = res.feature_matrix(W, mode="numpy")[:, :NZ]
    ok = ~np.isnan(y)
    tr, va = ok & (part == 0), ok & (part == 1)
    lam = select_lambda(X[tr], y[tr], X[va], y[va], GRID)
    readout = RidgeReadout(lam).fit(X[tr], y[tr])
    case = np.flatnonzero(part == 2)
    case = case[np.lexsort((sectors[case], dates[case]))]  # date, then sector (as E6)
    if len(case) != 88 or np.isnan(y[case]).any():
        raise RuntimeError("expected 88 scored ZF015 windows")
    return res, readout, lam, W[case], X[case], y[case]


def mae(y, X, readout):
    return float(np.mean(np.abs(y - np.maximum(readout.predict(X), 0.0))))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--dry-shots", type=int, default=2000)
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--backend", default="ibm_fez")
    parser.add_argument("--fetch")
    args = parser.parse_args()

    res, readout, lam, Wc, Xc, yc = setup()
    print(f"10 q, +72 h, Z/ZZ read-out ({NZ} features), lambda {lam:g}; "
          f"Yellowknife exact MAE {mae(yc, Xc, readout):.3f} km", flush=True)

    if args.dry_run:
        isa = hardware.transpile_for(hardware.build_circuits(res, Wc), hardware.FakeFez(), seed=11)
        print(f"FakeFez transpile: CZ {sorted({c.count_ops().get('cz', 0) for c in isa})}, "
              f"depth {min(c.depth() for c in isa)}-{max(c.depth() for c in isa)}, no SWAPs", flush=True)
        noisy = hardware.run_sampler(res, Wc, mode="fake_fez", shots=args.dry_shots, seed=11)
        print(f"FakeFez {args.dry_shots} shots: MAE {mae(yc, noisy, readout):.3f} km, "
              f"feature gap {np.abs(noisy - Xc).mean():.4f}", flush=True)

    if args.submit:
        if not args.yes:
            parser.error("--submit needs --yes")
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        service = QiskitRuntimeService()
        backend = service.backend(args.backend)
        circuits = hardware.build_circuits(res, Wc)
        isa = hardware.transpile_for(circuits, backend, seed=11)
        job = SamplerV2(mode=backend, options=OPTIONS).run(isa, shots=args.shots)
        jobs = json.loads(JOBS.read_text()) if JOBS.exists() else []
        jobs.append({"job_id": job.job_id(), "backend": args.backend, "lead": LEAD, "n_qubits": N,
                     "submitted_at": datetime.now(timezone.utc).isoformat(), "shots": args.shots,
                     "n_circuits": len(isa), "seed_transpiler": 11, "lambda": lam,
                     "windows_sha256": hashlib.sha256(np.asarray(Wc, "<f8").tobytes()).hexdigest(),
                     "cz_counts": [int(c.count_ops().get("cz", 0)) for c in isa],
                     "depths": [c.depth() for c in isa], "options": OPTIONS})
        JOBS.write_text(json.dumps(jobs, indent=2) + "\n")
        print(f"submitted job {job.job_id()} ({len(isa)} circuits x {args.shots} shots)", flush=True)

    if args.fetch:
        from qiskit_ibm_runtime import QiskitRuntimeService
        job = QiskitRuntimeService().job(args.fetch)
        status = str(job.status())
        print(f"job {args.fetch}: {status}", flush=True)
        if "DONE" not in status.upper():
            return 0
        result = job.result()
        import gzip
        counts = [pub.data.meas.get_counts() for pub in result]  # raw bit strings per circuit
        with gzip.open(ROOT / "hardware" / "e15_counts.json.gz", "wt") as fh:
            json.dump(counts, fh)
        hw = hardware._features_from_result(result, N)
        np.save(ROOT / "hardware" / "e15_hw_features.npy", hw)
        metrics = result.metadata.get("execution", {}) if hasattr(result, "metadata") else {}
        out = {"job_id": args.fetch, "lead_h": 24 * LEAD, "n_qubits": N, "n_windows": len(yc),
               "lambda": lam, "mae_exact": mae(yc, Xc, readout), "mae_hardware": mae(yc, hw, readout),
               "feature_gap_mean": float(np.abs(hw - Xc).mean()),
               "forecast_gap_mean": float(np.mean(np.abs(np.maximum(readout.predict(hw), 0)
                                                         - np.maximum(readout.predict(Xc), 0)))),
               "mean_forecast_exact": float(np.maximum(readout.predict(Xc), 0).mean()),
               "mean_forecast_hardware": float(np.maximum(readout.predict(hw), 0).mean()),
               "share_hardware_forecasts_zero": float(np.mean(np.maximum(readout.predict(hw), 0) == 0)),
               "mae_predict_zero": float(np.mean(np.abs(yc))),
               "mean_abs_feature_exact": float(np.abs(Xc).mean()),
               "mean_abs_feature_hardware": float(np.abs(hw).mean()),
               "usage": dict(job.usage_estimation) if hasattr(job, "usage_estimation") else None,
               "status": "exploratory, post-hoc; one fire; perfect-weather inputs"}
        try:
            out["quantum_seconds"] = job.metrics().get("usage", {}).get("quantum_seconds")
        except Exception:
            pass
        (ROOT / "results" / "e15_10q_hardware.json").write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
