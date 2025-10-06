# --*-- conding:utf-8 --*--
# @time:9/25/25 20:22
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:quantum.py

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import os, json, time, csv
import numpy as np

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import EstimatorV2 as Estimator, SamplerV2 as Sampler
from qiskit_ibm_runtime.options import EstimatorOptions, SamplerOptions

@dataclass
class QuantumConfig:
    workdir: str = "sqd_work"
    backend_name: Optional[str] = None
    optimization_level: int = 2
    seed_transpile: int = 42
    resilience_level: int = 1  # readout mitigation level
    total_shots: int = 40_000
    per_group_min: int = 200
    sampler_shots: int = 100_000

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


@dataclass
class SampleReport:
    backend_name: str
    num_qubits: int
    shots: int
    counts: Dict[str, int]
    timestamp: float

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

class QuantumExecutor:
    """
    Layer 2: Execute on IBM QPU. Provides:
      - energy_estimation(): Estimator-based Pauli expectation aggregation
      - sample_bitstrings(): Sampler-based bitstring sampling
    """
    def __init__(
        self,
        workdir: str,
        backend_name: Optional[str] = None,
        service: Optional[QiskitRuntimeService] = None,
    ):
        self.workdir = workdir
        self.service = service or QiskitRuntimeService()
        if backend_name is None:
            backend_name = self.service.backends(simulator=False)[0].name
        self.backend = self.service.backend(backend_name)
        self.backend_name = backend_name

    # -------- shot allocation --------
    @staticmethod
    def allocate_group_shots(
        coeffs: np.ndarray, groups: list[list[int]], total_shots: int, per_group_min: int
    ) -> list[int]:
        """Importance allocation: sum of |w_j| per group."""
        w_abs = np.abs(coeffs.real) + np.abs(coeffs.imag)
        w_abs = np.maximum(w_abs, 1e-16)
        group_weight = np.array([np.sum(w_abs[g]) for g in groups], dtype=float)
        group_shots = (group_weight / group_weight.sum() * total_shots).astype(int)
        group_shots = np.maximum(group_shots, per_group_min)
        scale = total_shots / max(int(np.sum(group_shots)), 1)
        return [max(int(s * scale), per_group_min) for s in group_shots]

    # -------- energy estimation --------
    def energy_estimation(
        self,
        grouped_circuits: list[QuantumCircuit],
        grouped_observables: list[list[SparsePauliOp]],
        coeffs: np.ndarray,
        cfg: QuantumConfig,
        initial_layout: Optional[List[int]] = None,
    ) -> EnergyReport:
        """Run EstimatorV2 per group; aggregate E = sum w_j <P_j>."""
        # Transpile
        circuits_tr = transpile(
            grouped_circuits,
            backend=self.backend,
            optimization_level=cfg.optimization_level,
            initial_layout=initial_layout,
            seed_transpiler=cfg.seed_transpile,
        )

        # Options and primitive
        est = Estimator(
            backend=self.backend,
            options=EstimatorOptions(resilience_level=cfg.resilience_level),
        )

        exp_values = [0.0] * sum(len(obs) for obs in grouped_observables)  # flat index will not be used directly
        # We'll store per-term results by mapping indices provided by caller.

        energy_terms: List[float] = []
        all_exp_values: List[float] = []
        flat_coeffs: List[float] = []

        # shots per group
        # caller should provide group mapping; here assume 1 circuit : 1 group
        # and observables[k] are the Pauli terms in that group (coeffs must be mapped accordingly by caller)
        # For simplicity, we re-allocate locally by |coeff| of observables in each group.
        group_weights = [np.sum(np.abs([complex(*o.coeffs.view(float).reshape(-1,2)[0]) for o in obs])) for obs in grouped_observables]  # rough weight
        # Using the config allocator:
        group_shots = self.allocate_group_shots(coeffs=np.array([c for c in coeffs]), groups=[list(range(len(obs))) for obs in grouped_observables], total_shots=cfg.total_shots, per_group_min=cfg.per_group_min)

        # Execute per group
        for gi, (qc_tr, obs_list, shots) in enumerate(zip(circuits_tr, grouped_observables, group_shots)):
            job = est.run([(qc_tr, o) for o in obs_list], shots=shots)
            res = job.result()
            vals = [float(np.real_if_close(v)) for v in res.values]

            # Persist raw group result
            meta = {
                "backend": self.backend_name,
                "group_index": gi,
                "shots": shots,
                "job_id": job.job_id(),
                "values": vals,
            }
            with open(os.path.join(self.workdir, "jobs", f"group_{gi}_estimator.json"), "w") as f:
                json.dump(meta, f, indent=2)
            with open(os.path.join(self.workdir, "circuits", f"group_{gi}_qasm.qasm"), "w") as f:
                f.write(qc_tr.qasm())

            all_exp_values.extend(vals)

        # Aggregate energy with provided coeffs order
        flat_coeffs = [float(np.real_if_close(c)) for c in coeffs]
        assert len(flat_coeffs) == len(all_exp_values), "Coeff and observable value length mismatch."
        energy_terms = [c * v for c, v in zip(flat_coeffs, all_exp_values)]
        energy = float(np.sum(energy_terms))

        report = EnergyReport(
            backend_name=self.backend_name,
            num_qubits=grouped_circuits[0].num_qubits if grouped_circuits else 0,
            total_shots=int(np.sum(group_shots)),
            resilience_level=cfg.resilience_level,
            groups=len(grouped_circuits),
            energy=energy,
            energy_terms=energy_terms,
            exp_values=all_exp_values,
            coeffs=flat_coeffs,
            group_shots=group_shots,
            timestamp=time.time(),
        )

        stamp = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.workdir, "results", f"energy_report_{stamp}.json"), "w") as f:
            f.write(report.to_json())
        with open(os.path.join(self.workdir, "results", f"energy_terms_{stamp}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["index", "coeff", "exp_value", "term_contrib"])
            for i, (c, v, t) in enumerate(zip(report.coeffs, report.exp_values, report.energy_terms)):
                w.writerow([i, c, v, t])

        return report

    # -------- bitstring sampling --------
    def sample_bitstrings(
        self,
        stateprep_circuit: QuantumCircuit,
        cfg: QuantumConfig,
        initial_layout: Optional[List[int]] = None,
    ) -> SampleReport:
        """Run SamplerV2 on the given stateprep circuit to collect raw counts."""
        circ = stateprep_circuit.copy()
        circ.measure_all()  # SamplerV2 can infer classical mapping; measure explicitly for clarity

        circ_tr = transpile(
            circ,
            backend=self.backend,
            optimization_level=cfg.optimization_level,
            initial_layout=initial_layout,
            seed_transpiler=cfg.seed_transpile,
        )

        sampler = Sampler(
            backend=self.backend,
            options=SamplerOptions(resilience_level=cfg.resilience_level),
        )
        job = sampler.run([circ_tr], shots=cfg.sampler_shots)
        res = job.result()
        qd = res.quasi_dists[0]
        # Convert to int counts by multiplying shots (best-effort)
        counts = {format(bit, f"0{circ_tr.num_qubits}b"): int(round(prob * cfg.sampler_shots)) for bit, prob in qd.items()}

        meta = {
            "backend": self.backend_name,
            "shots": cfg.sampler_shots,
            "job_id": job.job_id(),
        }
        with open(os.path.join(self.workdir, "jobs", "sampler_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        with open(os.path.join(self.workdir, "circuits", "sampler_qasm.qasm"), "w") as f:
            f.write(circ_tr.qasm())

        report = SampleReport(
            backend_name=self.backend_name,
            num_qubits=circ_tr.num_qubits,
            shots=cfg.sampler_shots,
            counts=counts,
            timestamp=time.time(),
        )
        stamp = time.strftime("%Y%m%d_%H%M%S")
        with open(os.path.join(self.workdir, "results", f"sampler_counts_{stamp}.json"), "w") as f:
            f.write(report.to_json())
        return report
