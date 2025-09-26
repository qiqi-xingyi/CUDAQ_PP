# --*-- conding:utf-8 --*--
# @time:9/25/25 20:21
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:classical.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os, json, csv
import numpy as np

from qiskit.quantum_info import SparsePauliOp

@dataclass
class SubspaceResult:
    E0: float
    evals: List[float]
    evecs: List[List[complex]]      # columns are eigenvectors in the chosen basis
    basis: List[str]                # bitstring basis order used to build H_sub

@dataclass
class DecodeResult:
    bitstring: str
    conformation: Dict              # user-defined structure fields
    notes: Optional[str] = None

@dataclass
class EvalReport:
    metrics: Dict[str, float]
    table_path: Optional[str] = None

class ClassicalPostProcessor:
    """
    Layer 3: Classical post-processing, including:
      - Subspace SQD: select top-K bitstrings by probability, build H_sub, diagonalize.
      - Decoding: bitstring -> protein conformation (user supplies mapping).
      - Evaluation: compute RMSD/contact metrics if references exist; export results.
    """
    def __init__(self, workdir: str = "sqd_work"):
        self.workdir = workdir

    # -------- probability utilities --------
    @staticmethod
    def normalize_counts(counts: Dict[str, int]) -> List[Tuple[str, float]]:
        total = sum(counts.values())
        if total == 0:
            return []
        items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
        return [(b, c / total) for b, c in items]

    @staticmethod
    def select_topk(prob_list: List[Tuple[str, float]], top_k: int = 500, mass_threshold: Optional[float] = 0.95) -> List[str]:
        chosen: List[str] = []
        mass = 0.0
        for b, p in prob_list:
            if top_k is not None and len(chosen) >= top_k:
                break
            chosen.append(b)
            mass += p
            if mass_threshold is not None and mass >= mass_threshold:
                break
        return chosen

    # -------- build H_sub and diagonalize --------
    def build_subspace_and_diagonalize(
        self,
        H: SparsePauliOp,
        basis: List[str],
    ) -> SubspaceResult:
        """
        Construct H_sub in computational basis defined by `basis` bitstrings and diagonalize it.
        H_sub[i,j] = <basis[i]| H |basis[j]>.
        """
        import numpy.linalg as LA
        size = len(basis)
        if size == 0:
            return SubspaceResult(E0=float("nan"), evals=[], evecs=[], basis=[])

        # Precompute Pauli action maps to accelerate <z_i|P|z_j>
        # For SparsePauliOp acting on |z>, only terms that flip bits according to X/Y contribute off-diagonals.
        # For simplicity, we build dense H_sub here; for large K, consider sparse.
        basis_index = {b: i for i, b in enumerate(basis)}
        H_sub = np.zeros((size, size), dtype=np.complex128)

        # Iterate each Pauli term
        for P, w in zip(H.paulis, H.coeffs):
            # Compute action of P on each basis vector
            # Strategy: apply X/Y flips to get target bitstring; multiply by phase from Y/Z.
            # For clarity, we use qiskit's pauli API to act on a basis ket:
            z_mask = np.array(list(basis), dtype=f"U{H.num_qubits}")  # not memory-efficient; replace for performance in production
            # Naive implementation (readable placeholder):
            label = P.to_label()[::-1]  # align qubit index
            # Build bit-flip and phase for each basis vector
            for i, b in enumerate(basis):
                # apply X/Y flips
                b_list = list(b)
                phase = 1.0 + 0.0j
                for q, ch in enumerate(label):
                    if ch == "X":
                        b_list[q] = "1" if b_list[q] == "0" else "0"
                    elif ch == "Y":
                        # Y = i X Z. On computational basis, effect equals X plus a ±i phase from Z.
                        # We flip like X and multiply by phase from Z eigenvalue of original bit.
                        phase *= (1j if b_list[q] == "0" else -1j)
                        b_list[q] = "1" if b_list[q] == "0" else "0"
                    elif ch == "Z":
                        phase *= (1.0 if b_list[q] == "0" else -1.0)
                b_target = "".join(b_list)
                j = basis_index.get(b_target, None)
                if j is not None:
                    H_sub[i, j] += w * phase

        evals, evecs = LA.eigh(H_sub)
        E0 = float(np.real_if_close(evals[0]))
        evecs_list = [list(vec) for vec in evecs.T]  # columns as eigenvectors
        return SubspaceResult(E0=E0, evals=[float(np.real_if_close(x)) for x in evals], evecs=evecs_list, basis=basis)

    # -------- decoding (placeholder, user to implement mapping) --------
    def decode_bitstring(self, bitstring: str) -> DecodeResult:
        """
        Map a bitstring to a protein conformation representation.
        TODO: implement your encoding map here (torsion states, lattice steps, contact flags, etc.)
        """
        conf = {"placeholder": True, "bitstring": bitstring}
        return DecodeResult(bitstring=bitstring, conformation=conf, notes="Implement your decoding logic.")

    # -------- evaluation (RMSD/contact etc.) --------
    def evaluate_candidates(
        self,
        decoded: List[DecodeResult],
        reference: Optional[object] = None,
        out_csv: Optional[str] = None,
    ) -> EvalReport:
        """
        Compute evaluation metrics for decoded conformations.
        If reference is provided (e.g., target structure), compute RMSD / contact F1, etc.
        """
        metrics = {}
        # TODO: fill with actual computations; placeholders below
        metrics["num_candidates"] = float(len(decoded))
        metrics["placeholder_metric"] = 0.0

        table_path = None
        if out_csv:
            table_path = out_csv
            with open(out_csv, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["bitstring", "notes"])
                for d in decoded:
                    w.writerow([d.bitstring, d.notes or ""])
        return EvalReport(metrics=metrics, table_path=table_path)

    # -------- convenience: full subspace pipeline --------
    def subspace_pipeline(
        self,
        H: SparsePauliOp,
        counts: Dict[str, int],
        top_k: int = 500,
        mass_threshold: float = 0.95,
    ) -> SubspaceResult:
        probs = self.normalize_counts(counts)
        basis = self.select_topk(probs, top_k=top_k, mass_threshold=mass_threshold)
        res = self.build_subspace_and_diagonalize(H, basis)
        # persist
        os.makedirs(self.workdir, exist_ok=True)
        with open(os.path.join(self.workdir, "results", "subspace_result.json"), "w") as f:
            json.dump(
                {
                    "E0": res.E0,
                    "evals": res.evals,
                    "basis": res.basis[:50],  # preview
                },
                f,
                indent=2,
            )
        return res
