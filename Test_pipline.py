# --*-- conding:utf-8 --*--
# @time:9/24/25 18:30
# @Author : Yuqi Zhang
# @Email : yzhan135@kent.edu
# @File:Test_pipline.py

from qiskit.quantum_info import SparsePauliOp
from SQD import PrepBuilder, PrepConfig, QuantumExecutor, QuantumConfig, ClassicalPostProcessor
from qiskit_ibm_runtime import QiskitRuntimeService
from Protein_Folding import Peptide
from Protein_Folding.interactions.miyazawa_jernigan_interaction import MiyazawaJerniganInteraction
from Protein_Folding.penalty_parameters import PenaltyParameters
from Protein_Folding.protein_folding_problem import ProteinFoldingProblem

main_chain_residue_seq = "YAGYS"
side_chain_residue_sequences = ['' for _ in range(len(main_chain_residue_seq))]
protein_name = '6mu3'


if __name__ == '__main__':

    char_count = len(main_chain_residue_seq)
    print(f'Num of Acid:{char_count}')

    side_site = len(side_chain_residue_sequences)
    print(side_chain_residue_sequences)
    print(f'Num of Side cite:{side_site}')

    # create Peptide
    peptide = Peptide(main_chain_residue_seq , side_chain_residue_sequences)

    # Interaction definition (e.g. Miyazawa-Jernigan)
    mj_interaction = MiyazawaJerniganInteraction()

    # Penalty Parameters Definition
    penalty_terms = PenaltyParameters(10, 10, 10)

    # Create Protein Folding case
    protein_folding_problem = ProteinFoldingProblem(peptide, mj_interaction, penalty_terms)

    # create quantum Op
    hamiltonian = protein_folding_problem.qubit_op()

    print(type(hamiltonian))

    H: SparsePauliOp = hamiltonian

    # 1) Prep layer: grouping + circuits
    prep = PrepBuilder(H, PrepConfig(workdir="sqd_run", he_layers=1))
    prep.set_stateprep_hardware_efficient()  # or prep.set_stateprep_problem_inspired(...)
    groups = prep.build_groups()
    grouped_circuits = [prep.build_group_circuit(i) for i in range(len(groups))]
    # Flatten observables with the same global order as H.coeffs:
    grouped_observables = []
    offset = 0
    for g in groups:
        obs = [SparsePauliOp(prep.paulis[idx], coeffs=[1.0]) for idx in g]
        grouped_observables.append(obs)

    # 2) Quantum layer: run on IBM QPU

    service = QiskitRuntimeService(
        channel="ibm_quantum_platform",
        instance="ibm_cleveland",
        token="TOKEN"
    )

    qexec = QuantumExecutor(workdir="sqd_run", backend_name="ibm_<device>")
    qcfg = QuantumConfig(workdir="sqd_run", total_shots=60000, sampler_shots=100000, resilience_level=1)

    energy_report = qexec.energy_estimation(
        grouped_circuits=grouped_circuits,
        grouped_observables=grouped_observables,
        coeffs=prep.coeffs,
        cfg=qcfg,
        initial_layout=None,
    )
    print("Energy =", energy_report.energy)

    sample_report = qexec.sample_bitstrings(
        stateprep_circuit=prep.stateprep,
        cfg=qcfg,
        initial_layout=None,
    )
    counts = sample_report.counts


    # 3) Classical layer: subspace + decode + evaluate
    post = ClassicalPostProcessor(workdir="sqd_run")
    subspace = post.subspace_pipeline(H, counts, top_k=500, mass_threshold=0.95)
    print("Subspace E0 =", subspace.E0)

    decoded = [post.decode_bitstring(b) for b in subspace.basis[:20]]  # decode top candidates
    eval_report = post.evaluate_candidates(decoded, reference=None, out_csv="sqd_run/results/candidates.csv")
    print(eval_report.metrics)