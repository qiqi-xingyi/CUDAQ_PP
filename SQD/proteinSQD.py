# --*-- conding:utf-8 --*--
# @time:9/24/25 21:50
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:proteinSQD.py

# protein_sqd.py
from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
import os, json, time, csv
import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.compiler import transpile
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit_ibm_runtime.options import EstimatorOptions


# ---------- Utilities: QWC grouping ----------
def qwc_commute(p1: Pauli, p2: Pauli) -> bool:
    """Check qubit-wise commutativity between two Pauli operators."""
    z1, x1 = p1.z, p1.x
    z2, x2 = p2.z, p2.x
    same_axis = np.logical_and(
        np.logical_or(z1 == z2, np.logical_or(z1 == False, z2 == False)),
        np.logical_or(x1 == x2, np.logical_or(x1 == False, x2 == False)),
    )
    return bool(np.all(same_axis))


def group_qwc(paulis: List[Pauli]) -> List[List[int]]:
    """Greedy QWC grouping: return a list of index groups."""
    groups: List[List[int]] = []
    for idx, p in enumerate(paulis):
        placed = False
        for g in groups:
            if all(qwc_commute(p, paulis[j]) for j in g):
                g.append(idx)
                placed = True
                break
        if not placed:
            groups.append([idx])
    return groups


# ---------- Data container ----------
@dataclass
class EnergyReport:
    backend_name: str
    num_qubits: int
    total_shots: int
    resilience_level: int
    groups: int
    energy: float
    energy_terms: List[float]
    exp_values: List[float]
    coeffs: List[float]
    group_shots: List[int]
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


# ---------- Main class ----------
class proteinSQD:
    def __init__(
        self,
        H: SparsePauliOp,
        workdir: str = "sqd_protein",
        service: Optional[QiskitRuntimeService] = None,
        backend_name: Optional[str] = None,
    ):
        assert isinstance(H, SparsePauliOp), "H must be SparsePauliOp"
        self.H = H.simplify()
        self.n = H.num_qubits
        self.workdir = workdir
        os.makedirs(workdir, exist_ok=True)
        for d in ["configs", "circuits", "jobs", "results", "logs"]:
            os.makedirs(os.path.join(workdir, d), exist_ok=True)

        # Runtime and backend
        self.service = service or QiskitRuntimeService()
        if backend_name is None:
            backend_name = self.service.backends(simulator=False)[0].name
        self.backend = self.service.backend(backend_name)
        self.backend_name = backend_name

        # State preparation circuit placeholder
        self.stateprep = QuantumCircuit(self.n, name="stateprep")

        # Grouping placeholders
        self.groups: List[List[int]] = []
        self.paulis: List[Pauli] = [Pauli(p) for p in self.H.paulis]
        self.coeffs: np.ndarray = np.array(self.H.coeffs, dtype=complex)

    # ---------- State preparation ----------
    def set_stateprep_hardware_efficient(self, layers: int = 1, theta_seed: int = 7):
        rng = np.random.default_rng(theta_seed)
        qc = QuantumCircuit(self.n, name="he_stateprep")
        for _ in range(layers):
            for q in range(self.n):
                qc.ry(float(rng.uniform(-0.4, 0.4)), q)
            for q in range(self.n - 1):
                qc.cx(q, q + 1)
        self.stateprep = qc

    def set_stateprep_problem_inspired(
        self, angles: Optional[List[float]] = None, entangle: bool = True
    ):
        qc = QuantumCircuit(self.n, name="pi_stateprep")
        if angles is None:
            angles = [0.2] * self.n
        assert len(angles) == self.n
        for q, a in enumerate(angles):
            qc.ry(float(a), q)
        if entangle:
            for q in range(0, self.n - 1, 2):
                qc.cx(q, q + 1)
        self.stateprep = qc

    def set_stateprep_custom(self, qc: QuantumCircuit):
        assert qc.num_qubits == self.n
        self.stateprep = qc

    # ---------- Grouping ----------
    def build_groups(self, method: str = "qwc"):
        if method != "qwc":
            raise NotImplementedError("Only QWC grouping is implemented")
        self.groups = group_qwc(self.paulis)
        return self.groups

    # ---------- Basis change ----------
    @staticmethod
    def _basis_change_for_pauli(p: Pauli) -> List[Tuple[str, int]]:
        """Generate basis change gates for a single Pauli operator."""
        ops: List[Tuple[str, int]] = []
        label = p.to_label()
        for qubit, ch in enumerate(label[::-1]):  # reversed to match qiskit ordering
            if ch == "X":
                ops.append(("h", qubit))
            elif ch == "Y":
                ops.append(("sdg", qubit))
                ops.append(("h", qubit))
        return ops

    def _basis_change_for_group(self, indices: List[int]) -> List[Tuple[str, int]]:
        p0 = self.paulis[indices[0]]
        return self._basis_change_for_pauli(p0)

    # ---------- Shot allocation ----------
    def allocate_shots(
        self, total_shots: int = 40_000, per_group_min: int = 200
    ) -> Tuple[List[int], List[int]]:
        if not self.groups:
            self.build_groups()
        w_abs = np.abs(self.coeffs.real) + np.abs(self.coeffs.imag)
        w_abs = np.maximum(w_abs, 1e-16)
        term_shots = (w_abs / w_abs.sum() * total_shots).astype(int)
        term_shots = np.maximum(term_shots, 10)
        group_shots = []
        for g in self.groups:
            s = int(np.sum(term_shots[g]))
            group_shots.append(max(s, per_group_min))
        scale = total_shots / max(int(np.sum(group_shots)), 1)
        group_shots = [max(int(s * scale), per_group_min) for s in group_shots]
        return group_shots, term_shots.tolist()

    # ---------- Circuit building ----------
    def build_group_circuit(self, group_idx: int) -> QuantumCircuit:
        assert self.groups, "Call build_groups() first"
        g = self.groups[group_idx]
        qc = QuantumCircuit(self.n, name=f"group{group_idx}_meas")
        qc.compose(self.stateprep, range(self.n), inplace=True)
        for name, q in self._basis_change_for_group(g):
            getattr(qc, name)(q)
        return qc

    # ---------- Execution ----------
    def run(
        self,
        total_shots: int = 40_000,
        resilience_level: int = 1,
        optimization_level: int = 2,
        layout: Optional[List[int]] = None,
        seed_transpile: int = 42,
    ) -> EnergyReport:
        if not self.groups:
            self.build_groups()

        group_shots, _ = self.allocate_shots(total_shots=total_shots)

        circuits: List[QuantumCircuit] = []
        observables: List[List[SparsePauliOp]] = []
        shots_list: List[int] = []

        for gi, g in enumerate(self.groups):
            qc = self.build_group_circuit(gi)
            obs_list = []
            for idx in g:
                obs_list.append(SparsePauliOp(self.paulis[idx], coeffs=[1.0]))
            circuits.append(qc)
            observables.append(obs_list)
            shots_list.append(group_shots[gi])

        circuits_tr = transpile(
            circuits,
            backend=self.backend,
            optimization_level=optimization_level,
            initial_layout=layout,
            seed_transpiler=seed_transpile,
        )

        options = EstimatorOptions(resilience_level=resilience_level)
        estimator = Estimator(backend=self.backend, options=options)

        exp_values: List[float] = [0.0] * len(self.paulis)
        energy_terms: List[float] = [0.0] * len(self.paulis)

        for gi, (qc_tr, obs_list) in enumerate(zip(circuits_tr, observables)):
            job = estimator.run([(qc_tr, o) for o in obs_list], shots=shots_list[gi])
            res = job.result()
            vals = res.values
            for loc, idx in enumerate(self.groups[gi]):
                v = float(np.real_if_close(vals[loc]))
                exp_values[idx] = v
                energy_terms[idx] = float(np.real_if_close(self.coeffs[idx] * v))

            meta = {
                "backend": self.backend_name,
                "group_index": gi,
                "shots": shots_list[gi],
                "pauli_indices": self.groups[gi],
                "job_id": job.job_id(),
                "values": [float(np.real_if_close(x)) for x in vals],
            }
            with open(
                os.path.join(self.workdir, "jobs", f"group_{gi}_result.json"), "w"
            ) as f:
                json.dump(meta, f, indent=2)

            with open(
                os.path.join(self.workdir, "circuits", f"group_{gi}_qasm.qasm"), "w"
            ) as f:
                f.write(qc_tr.qasm())

        energy = float(np.sum(energy_terms))

        report = EnergyReport(
            backend_name=self.backend_name,
            num_qubits=self.n,
            total_shots=int(np.sum(shots_list)),
            resilience_level=resilience_level,
            groups=len(self.groups),
            energy=energy,
            energy_terms=energy_terms,
            exp_values=exp_values,
            coeffs=[float(np.real_if_close(c)) for c in self.coeffs],
            group_shots=shots_list,
            timestamp=time.time(),
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(
            self.workdir, "results", f"energy_report_{stamp}.json"
        )
        with open(json_path, "w") as f:
            f.write(report.to_json())

        csv_path = os.path.join(
            self.workdir, "results", f"energy_terms_{stamp}.csv"
        )
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "coeff", "exp_value", "term_contrib"])
            for i, (c, v, t) in enumerate(
                zip(report.coeffs, report.exp_values, report.energy_terms)
            ):
                w.writerow([i, c, v, t])

        return report
