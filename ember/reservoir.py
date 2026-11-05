"""Fixed chain reservoir with independent Qiskit and NumPy simulators."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def feature_names(n):
    """Names in read-out order: Z, lexicographic ZZ, then X."""
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    return ([f"Z{i}" for i in range(n)]
            + [f"Z{i}Z{j}" for i, j in combinations(range(n), 2)]
            + [f"X{i}" for i in range(n)])


def counts_to_features(counts: dict[str, int], n):
    """Convert Qiskit Z-basis counts to Z and ZZ expectations."""
    size = len(feature_names(n)) - n
    pairs = list(combinations(range(n), 2))
    features = np.zeros(size)
    total = 0
    for key, count in counts.items():
        bits = key.replace(" ", "")[::-1]  # The only count-key reversal.
        if len(bits) != n or any(bit not in "01" for bit in bits):
            raise ValueError(f"expected a {n}-bit count key, got {key!r}")
        if count < 0:
            raise ValueError("counts must be nonnegative")
        signs = np.array([1.0 - 2.0 * int(bit) for bit in bits])
        features[:n] += count * signs
        features[n:] += count * np.array([signs[i] * signs[j] for i, j in pairs])
        total += count
    if total <= 0:
        raise ValueError("counts must contain at least one shot")
    return features / total


def _ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]])


class ChainReservoir:
    """Restart at zero per window; carry the state through all its days."""

    def __init__(self, n=5, gamma=np.pi / 2, seed=7,
                 entangler="cz_chain", ent_seed=None):
        feature_names(n)  # Validate the register size.
        if not np.isfinite(gamma):
            raise ValueError("gamma must be finite")
        if entangler not in ("cz_chain", "haar"):
            raise ValueError("entangler must be 'cz_chain' or 'haar'")
        self.n, self.gamma = n, float(gamma)
        self.entangler = entangler
        rng = np.random.default_rng(seed)
        self.gains = rng.uniform(0.5, 1.5, size=n)
        self.w_angles = rng.uniform(0, 2 * np.pi, size=(n, 2))
        self.gains.flags.writeable = False
        self.w_angles.flags.writeable = False

        ent_rng = np.random.default_rng(ent_seed)
        gates = []
        for _ in range(n - 1):
            if entangler == "cz_chain":
                gate = np.diag([1, 1, 1, -1]).astype(complex)
            else:
                z = (ent_rng.normal(size=(4, 4))
                     + 1j * ent_rng.normal(size=(4, 4)))
                q, r = np.linalg.qr(z)
                diagonal = np.diag(r)
                gate = q * (diagonal / np.abs(diagonal))
            gate.flags.writeable = False
            gates.append(gate)
        self._entanglers = tuple(gates)

        # Compile the fixed layer entirely in NumPy, qubit 0 leftmost.
        dim = 2 ** n
        layer = np.eye(dim, dtype=complex)
        for i, gate in enumerate(self._entanglers):
            operator = np.kron(np.kron(np.eye(2 ** i), gate),
                               np.eye(2 ** (n - i - 2)))
            layer = operator @ layer
        for i, (a, b) in enumerate(self.w_angles):
            gate = np.diag(np.exp([-0.5j * b, 0.5j * b])) @ _ry(a)
            operator = np.kron(np.kron(np.eye(2 ** i), gate),
                               np.eye(2 ** (n - i - 1)))
            layer = operator @ layer
        self._layer = layer
        indices = np.arange(dim)
        signs = 1.0 - 2.0 * ((indices[:, None] >> np.arange(n - 1, -1, -1)) & 1)
        self._z_signs = np.column_stack(
            [signs] + [signs[:, i] * signs[:, j]
                       for i, j in combinations(range(n), 2)])

    def _validate(self, values, ndim):
        values = np.asarray(values, dtype=float)
        if values.ndim != ndim or values.shape[-1] != self.n:
            shape = "(L, n)" if ndim == 2 else "(N, L, n)"
            raise ValueError(f"expected input shape {shape} with n={self.n}")
        if (not np.isfinite(values).all() or np.any(values < -1e-12)
                or np.any(values > 1 + 1e-12)):
            raise ValueError("inputs must be finite and scaled to [0, 1]")
        return values

    def circuit(self, window):
        """Build an unmeasured circuit, oldest day first."""
        from qiskit import QuantumCircuit

        window = self._validate(window, 2)
        qc = QuantumCircuit(self.n)
        for day in window:
            for i in range(self.n):
                qc.ry(self.gamma * self.gains[i] * day[i], i)
            for i, gate in enumerate(self._entanglers):
                if self.entangler == "cz_chain":
                    qc.cz(i, i + 1)
                else:
                    # Qiskit's first qarg is the low bit of this local gate.
                    qc.unitary(gate, [i + 1, i])
            for i, (a, b) in enumerate(self.w_angles):
                qc.ry(a, i)
                qc.rz(b, i)
        return qc

    def measured_circuit(self, window, basis="Z"):
        """Measure every qubit in the Z or X basis."""
        if basis not in ("Z", "X"):
            raise ValueError("basis must be 'Z' or 'X'")
        qc = self.circuit(window)
        if basis == "X":
            qc.h(range(self.n))
        qc.measure_all()
        return qc

    def features_exact(self, window):
        """Evaluate Pauli expectations using Qiskit's Statevector."""
        from qiskit.quantum_info import Pauli, Statevector

        state = Statevector.from_instruction(self.circuit(window))
        terms = ([('Z', (i,)) for i in range(self.n)]
                 + [('Z', pair) for pair in combinations(range(self.n), 2)]
                 + [('X', (i,)) for i in range(self.n)])
        features = []
        for axis, qubits in terms:
            label = ["I"] * self.n
            for i in qubits:
                label[self.n - 1 - i] = axis
            features.append(state.expectation_value(Pauli("".join(label))).real)
        return np.array(features)

    def _numpy_batch(self, windows):
        """Propagate a batch of big-endian statevectors without Qiskit."""
        count = len(windows)
        state = np.zeros((count, 2 ** self.n), dtype=complex)
        state[:, 0] = 1.0
        for day in range(windows.shape[1]):
            for i in range(self.n):
                theta = self.gamma * self.gains[i] * windows[:, day, i]
                c = np.cos(theta / 2)[:, None, None]
                s = np.sin(theta / 2)[:, None, None]
                tensor = state.reshape(count, 2 ** i, 2, 2 ** (self.n - i - 1))
                zero, one = tensor[:, :, 0, :].copy(), tensor[:, :, 1, :].copy()
                tensor[:, :, 0, :] = c * zero - s * one
                tensor[:, :, 1, :] = s * zero + c * one
            state = state @ self._layer.T
        features = np.empty((count, len(feature_names(self.n))))
        z_count = self._z_signs.shape[1]
        features[:, :z_count] = (state.real ** 2 + state.imag ** 2) @ self._z_signs
        for i in range(self.n):
            tensor = state.reshape(count, 2 ** i, 2, 2 ** (self.n - i - 1))
            features[:, z_count + i] = 2 * np.sum(
                (tensor[:, :, 0, :].conj() * tensor[:, :, 1, :]).real, axis=(1, 2))
        return features

    def features_numpy(self, window):
        """Compute all observables with the pure-NumPy validation twin."""
        window = self._validate(window, 2)
        return self._numpy_batch(window[None, ...])[0]

    def _sampled_matrix(self, windows, shots, seed, noise_model=None):
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        if not isinstance(shots, (int, np.integer)) or shots <= 0:
            raise ValueError("shots must be a positive integer")
        size = len(feature_names(self.n)) - self.n
        if not len(windows):
            return np.empty((0, size))
        simulator = AerSimulator(noise_model=noise_model, max_parallel_threads=1)
        circuits = [self.measured_circuit(w) for w in windows]
        circuits = transpile(circuits, simulator, seed_transpiler=seed)
        result = simulator.run(circuits, shots=int(shots), seed_simulator=seed).result()
        return np.array([counts_to_features(result.get_counts(i), self.n)
                         for i in range(len(windows))])

    def features_sampled(self, window, shots, seed, noise_model=None):
        """Estimate Z and ZZ features from local Aer counts."""
        window = self._validate(window, 2)
        return self._sampled_matrix(window[None, ...], shots, seed, noise_model)[0]

    def feature_matrix(self, windows, mode="exact", shots=None, seed=None):
        """Return one feature row per window; NumPy batches use bounded memory."""
        if mode not in ("exact", "numpy", "sampled"):
            raise ValueError("mode must be 'exact', 'numpy', or 'sampled'")
        windows = self._validate(windows, 3)
        if mode == "sampled":
            return self._sampled_matrix(windows, shots, seed)
        features = np.empty((len(windows), len(feature_names(self.n))))
        if mode == "exact":
            for i, window in enumerate(windows):
                features[i] = self.features_exact(window)
        else:
            for start in range(0, len(windows), 4096):
                stop = start + 4096
                features[start:stop] = self._numpy_batch(windows[start:stop])
        return features
