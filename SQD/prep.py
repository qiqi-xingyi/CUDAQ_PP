# --*-- conding:utf-8 --*--
# @time:9/25/25 20:16
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:prep.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, List
import os

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp, Pauli

@dataclass
class PrepConfig:
    """Configuration for state preparation and grouping."""
    workdir: str = "sqd_work"
    grouping: str = "qwc"           # "qwc" or future options
    he_layers: int = 1              # hardware-efficient layers
    he_theta_seed: int = 7          # RNG seed for HE angles
    use_problem_inspired: bool = False
    problem_angles: Optional[List[float]] = None
    problem_entangle: bool = True

class PrepBuilder:
    """
    Layer 1: Handle input SparsePauliOp, build state-preparation circuits,
    and prepare per-group measurement circuits (basis changes X->H, Y->Sdg+H).
    """
    def __init__(self, H: SparsePauliOp, cfg: PrepConfig):
        assert isinstance(H, SparsePauliOp), "H must be SparsePauliOp"
        self.H = H.simplify()
        self.n = self.H.num_qubits
        self.cfg = cfg

        os.makedirs(self.cfg.workdir, exist_ok=True)
        for d in ["configs", "circuits", "jobs", "results", "logs"]:
            os.makedirs(os.path.join(self.cfg.workdir, d), exist_ok=True)

        # default stateprep: |0...0>
        self.stateprep = QuantumCircuit(self.n, name="stateprep")

        # cache
        self._paulis = [Pauli(p) for p in self.H.paulis]
        self._coeffs = self.H.coeffs
        self.groups: list[list[int]] = []

    # -------- state preparation families --------
    def set_stateprep_hardware_efficient(self, layers: Optional[int] = None, theta_seed: Optional[int] = None):
        """Hardware-efficient: random small RY rotations + CX ladder."""
        import numpy as np
        rng = np.random.default_rng(theta_seed or self.cfg.he_theta_seed)
        L = layers if layers is not None else self.cfg.he_layers

        qc = QuantumCircuit(self.n, name="he_stateprep")
        for _ in range(L):
            for q in range(self.n):
                qc.ry(float(rng.uniform(-0.4, 0.4)), q)
            for q in range(self.n - 1):
                qc.cx(q, q + 1)
        self.stateprep = qc

    def set_stateprep_problem_inspired(self, angles: Optional[List[float]] = None, entangle: Optional[bool] = None):
        """Problem-inspired: per-qubit RY with optional sparse entanglement."""
        if angles is None:
            angles = self.cfg.problem_angles or [0.2] * self.n
        if entangle is None:
            entangle = self.cfg.problem_entangle

        qc = QuantumCircuit(self.n, name="pi_stateprep")
        for q, a in enumerate(angles):
            qc.ry(float(a), q)
        if entangle:
            for q in range(0, self.n - 1, 2):
                qc.cx(q, q + 1)
        self.stateprep = qc

    def set_stateprep_custom(self, qc: QuantumCircuit):
        assert qc.num_qubits == self.n
        self.stateprep = qc

    # -------- grouping and basis changes --------
    def build_groups(self, method: Optional[str] = None) -> list[list[int]]:
        """Group Pauli terms for shared measurement bases; default QWC greedy."""
        method = method or self.cfg.grouping
        if method != "qwc":
            raise NotImplementedError("Only QWC grouping is implemented yet.")
        self.groups = self._group_qwc(self._paulis)
        return self.groups

    @staticmethod
    def _qwc_commute(p1: Pauli, p2: Pauli) -> bool:
        import numpy as np
        z1, x1 = p1.z, p1.x
        z2, x2 = p2.z, p2.x
        same_axis = np.logical_and(
            (z1 == z2) | (~z1) | (~z2),
            (x1 == x2) | (~x1) | (~x2),
        )
        return bool(same_axis.all())

    def _group_qwc(self, paulis: list[Pauli]) -> list[list[int]]:
        groups: list[list[int]] = []
        for idx, p in enumerate(paulis):
            placed = False
            for g in groups:
                if all(self._qwc_commute(p, paulis[j]) for j in g):
                    g.append(idx)
                    placed = True
                    break
            if not placed:
                groups.append([idx])
        return groups

    @staticmethod
    def _basis_change_ops_for_pauli(p: Pauli) -> list[tuple[str, int]]:
        """Return list of (gate_name, qubit) to rotate X/Y into Z basis."""
        ops: list[tuple[str, int]] = []
        label = p.to_label()
        for qubit, ch in enumerate(label[::-1]):  # reversed index order
            if ch == "X":
                ops.append(("h", qubit))
            elif ch == "Y":
                ops.append(("sdg", qubit))
                ops.append(("h", qubit))
        return ops

    def build_group_circuit(self, group_idx: int) -> QuantumCircuit:
        """Compose stateprep + basis change for a given group."""
        assert self.groups, "Call build_groups() first."
        g = self.groups[group_idx]
        qc = QuantumCircuit(self.n, name=f"group{group_idx}_meas")
        qc.compose(self.stateprep, range(self.n), inplace=True)
        # Use first term as representative due to QWC
        for name, q in self._basis_change_ops_for_pauli(self._paulis[g[0]]):
            getattr(qc, name)(q)
        return qc

    # convenience getters
    @property
    def paulis(self) -> list[Pauli]:
        return self._paulis

    @property
    def coeffs(self):
        return self._coeffs
