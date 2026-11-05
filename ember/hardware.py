"""Local hardware preparation and explicitly confirmed Runtime jobs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
from qiskit_ibm_runtime.fake_provider import FakeFez

from ember.reservoir import counts_to_features

JOBS_PATH = Path(__file__).resolve().parents[1] / "hardware" / "jobs.json"
_runtime_service = None


def _positive_int(value, name):
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < 1):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def build_circuits(reservoir, windows):
    """Build Z-basis measurements in input-window order."""
    windows = reservoir._validate(windows, 3)
    if reservoir.entangler != "cz_chain":
        raise ValueError("hardware preparation requires the CZ chain reservoir")
    circuits = []
    for window in windows:
        circuit = reservoir.measured_circuit(window, basis="Z")
        circuit.metadata = {"n": reservoir.n, "window_days": len(window),
                            "expected_cz": len(window) * (reservoir.n - 1)}
        circuits.append(circuit)
    return circuits


def transpile_for(circuits, backend, seed):
    """Compile ISA circuits and reject extra routing entanglers."""
    circuits = list(circuits)
    if not circuits:
        return []
    manager = generate_preset_pass_manager(
        optimization_level=1, backend=backend, seed_transpiler=seed)
    isa = manager.run(circuits)
    for index, (source, compiled) in enumerate(zip(circuits, isa)):
        expected = (source.metadata or {}).get("expected_cz", source.count_ops().get("cz", 0))
        ops = compiled.count_ops()
        actual = ops.get("cz", 0)
        if actual != expected or ops.get("swap", 0):
            raise ValueError(f"circuit {index}: expected {expected} CZ and no SWAPs; got {dict(ops)}")
    return isa


def _features_from_result(result, n):
    """Decode the measured PUB register through the shared count converter."""
    return np.asarray([counts_to_features(pub.data.meas.get_counts(), n)
                       for pub in result], dtype=float).reshape(-1, n * (n + 1) // 2)


def run_sampler(reservoir, windows, mode, shots, seed):
    """Sample locally with ideal Aer or Runtime's noisy FakeFez mode."""
    if mode not in ("aer_ideal", "fake_fez"):
        raise ValueError("mode must be 'aer_ideal' or 'fake_fez'")
    shots = _positive_int(shots, "shots")
    circuits = build_circuits(reservoir, windows)
    if not circuits:
        return np.empty((0, reservoir.n * (reservoir.n + 1) // 2))
    if mode == "aer_ideal":
        backend = AerSimulator(max_parallel_threads=1, seed_simulator=seed)
    else:
        backend = FakeFez()
        backend.set_options(max_parallel_threads=1, seed_simulator=seed)
    isa = transpile_for(circuits, backend, seed)
    sampler = SamplerV2(mode=backend, options={"simulator": {"seed_simulator": seed}})
    result = sampler.run(isa, shots=shots).result()
    return _features_from_result(result, reservoir.n)


def budget(n_circuits, shots):
    """Estimate shots and two-qubit survival from bundled FakeFez errors."""
    n_circuits = _positive_int(n_circuits, "n_circuits")
    shots = _positive_int(shots, "shots")
    errors = [properties.error for properties in FakeFez().target["cz"].values()
              if properties is not None and properties.error is not None]
    eps_2q = float(np.median(errors))
    return {"n_circuits": n_circuits, "shots": shots,
            "total_shots": n_circuits * shots, "cz_per_circuit": 20,
            "eps_2q": eps_2q, "fidelity_2q": (1 - eps_2q) ** 20,
            "shot_se": float(1 / np.sqrt(shots))}


def submit(reservoir, windows, backend_name, shots, confirm):
    """Submit one job only after explicit confirmation; record its identifier."""
    if confirm is not True:
        raise PermissionError("hardware submission requires confirm=True")

    global _runtime_service
    shots = _positive_int(shots, "shots")
    windows = reservoir._validate(windows, 3)
    if reservoir.n != 5 or windows.shape[1] != 5 or not len(windows):
        raise ValueError("submission requires nonempty five-day, five-qubit windows")
    circuits = build_circuits(reservoir, windows)
    # Check the log before submitting so a malformed log cannot lose a job ID.
    jobs = json.loads(JOBS_PATH.read_text()) if JOBS_PATH.exists() else []
    if not isinstance(jobs, list):
        raise ValueError("hardware/jobs.json must contain a list of job records")

    _runtime_service = QiskitRuntimeService()
    backend = _runtime_service.backend(backend_name)
    seed = 11
    isa = transpile_for(circuits, backend, seed)
    options = {"dynamical_decoupling": {"enable": True, "sequence_type": "XY4"},
               "twirling": {"enable_gates": True}}
    job = SamplerV2(mode=backend, options=options).run(isa, shots=shots)
    job_id = job.job_id()
    jobs.append({"job_id": job_id, "backend": backend_name, "mode": "job",
                 "submitted_at": datetime.now(timezone.utc).isoformat(),
                 "budget": budget(len(circuits), shots), "seed_transpiler": seed,
                 "n_qubits": reservoir.n, "window_days": windows.shape[1],
                 "gamma": reservoir.gamma, "gains": reservoir.gains.tolist(),
                 "w_angles": reservoir.w_angles.tolist(),
                 "windows_sha256": hashlib.sha256(
                     np.asarray(windows, dtype="<f8").tobytes()).hexdigest(),
                 "cz_counts": [int(c.count_ops().get("cz", 0)) for c in isa],
                 "depths": [c.depth() for c in isa], "options": options})
    JOBS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = JOBS_PATH.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(jobs, indent=2) + "\n")
    temporary.replace(JOBS_PATH)
    return job_id


def fetch(job_id, *, service=None):
    """Fetch a finished job using an explicitly supplied or reused service."""
    service = service if service is not None else _runtime_service
    if service is None:
        raise RuntimeError("fetch requires an explicit service or one from a confirmed submission")
    job = service.job(job_id)
    if not job.done():
        raise ValueError(f"job {job_id} has not finished successfully")
    return _features_from_result(job.result(), 5)
