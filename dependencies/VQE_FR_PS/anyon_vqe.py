from __future__ import annotations

import json
import numbers
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import OptimizeResult, minimize

import qiskit.quantum_info as qi
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector


@dataclass(frozen=True)
class HamiltonianSpec:
    num_qubits: int = 7
    j_f3: float = 1.0
    j_bf4: float = 0.5


@dataclass(frozen=True)
class EstimatorBudget:
    fr_shots_per_term: int = 8000
    ps_shots_per_term_total: int = 20000


@dataclass
class OptimizationRecord:
    iteration: int
    energy: float
    method: str


def normalize_ansatz_kind(ansatz_kind: str) -> str:
    normalized = str(ansatz_kind).strip().lower().replace("-", "_")
    aliases = {
        "hea": "hardware_efficient",
        "he": "hardware_efficient",
        "hardware": "hardware_efficient",
        "hardware_efficient_nnn": "hardware_efficient_nnn",
        "hea_nnn": "hardware_efficient_nnn",
        "problem": "problem_inspired",
        "hamiltonian": "problem_inspired",
        "qaoa_like": "problem_inspired",
        "charge": "charge_conserving",
        "charge_conserving": "charge_conserving",
    }
    normalized = aliases.get(normalized, normalized)
    allowed = {
        "hardware_efficient",
        "hardware_efficient_nnn",
        "problem_inspired",
        "charge_conserving",
    }
    if normalized not in allowed:
        raise ValueError(f"Unknown ansatz kind {ansatz_kind}")
    return normalized


def fibonacci_fr_matrices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    f_matrix = np.array(
        [[1.0 / phi, 1.0 / np.sqrt(phi)], [1.0 / np.sqrt(phi), -1.0 / phi]],
        dtype=complex,
    )
    r_matrix = np.diag([np.exp(-4j * np.pi / 5), np.exp(3j * np.pi / 5)]).astype(complex)
    b_matrix = f_matrix.conj().T @ r_matrix @ f_matrix
    return f_matrix, r_matrix, b_matrix


def r_move_gate_compact_1q():
    _, r_matrix, _ = fibonacci_fr_matrices()
    gate = qi.Operator(r_matrix).to_instruction()
    gate.label = "R"
    return gate


def f_gate_optimal():
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    f_matrix = np.array(
        [[1.0 / phi, 1.0 / np.sqrt(phi)], [1.0 / np.sqrt(phi), -1.0 / phi]],
        dtype=complex,
    )
    f_gate = qi.Operator(f_matrix).to_instruction()
    f_gate.label = "F"

    circuit = QuantumCircuit(3)
    controlled_f = f_gate.control(2)
    circuit.append(controlled_f, [0, 2, 1])
    circuit.x(0)
    circuit.x(2)
    circuit.mcx([0, 2], 1)
    circuit.x(0)
    circuit.x(2)
    return circuit.to_gate(label="F_opt")


def b_move_gate_3q():
    f_gate = f_gate_optimal()
    r_gate = r_move_gate_compact_1q()
    circuit = QuantumCircuit(3)
    circuit.append(f_gate, [0, 1, 2])
    circuit.append(r_gate, [1])
    circuit.append(f_gate.inverse(), [0, 1, 2])
    return circuit.to_gate(label="B")


def create_fusion_valid_reference(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits, name="init")
    for qubit in range(num_qubits):
        circuit.x(qubit)
    return circuit


def create_reference_state(num_qubits: int, reference: str = "fusion_valid") -> QuantumCircuit:
    normalized = str(reference).strip().lower().replace("-", "_")
    if normalized in {"fusion_valid", "fusion", "all_ones"}:
        return create_fusion_valid_reference(num_qubits)
    if normalized in {"zero", "all_zero"}:
        return QuantumCircuit(num_qubits, name="init")
    if normalized in {"alternating", "neel"}:
        circuit = QuantumCircuit(num_qubits, name="init")
        for qubit in range(0, num_qubits, 2):
            circuit.x(qubit)
        return circuit
    raise ValueError(f"Unknown reference state {reference}")


def charge_conserving_block(theta: float, phi: float):
    circuit = QuantumCircuit(2)
    circuit.cx(1, 0)
    circuit.rz(-phi, 1)
    circuit.ry(-theta, 1)
    circuit.cx(0, 1)
    circuit.ry(theta, 1)
    circuit.rz(phi, 1)
    circuit.cx(1, 0)
    return circuit.to_gate(label="A")


def brickwall_pairs(num_qubits: int) -> list[tuple[int, int]]:
    even_pairs = [(index, index + 1) for index in range(0, num_qubits - 1, 2)]
    odd_pairs = [(index, index + 1) for index in range(1, num_qubits - 1, 2)]
    return even_pairs + odd_pairs


def num_charge_conserving_parameters(num_qubits: int, depth: int) -> int:
    return 2 * depth * len(brickwall_pairs(num_qubits))


def build_charge_conserving_ansatz(
    num_qubits: int,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    include_reference: bool = True,
) -> QuantumCircuit:
    flat_parameters = np.asarray(parameters, dtype=float).ravel()
    expected = num_charge_conserving_parameters(num_qubits, depth)
    if flat_parameters.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {flat_parameters.size}")

    circuit = create_fusion_valid_reference(num_qubits) if include_reference else QuantumCircuit(num_qubits)
    pairs = brickwall_pairs(num_qubits)
    cursor = 0
    for _ in range(depth):
        for q0, q1 in pairs:
            theta = float(flat_parameters[cursor])
            phi = float(flat_parameters[cursor + 1])
            cursor += 2
            circuit.append(charge_conserving_block(theta, phi), [q0, q1])
    return circuit


def num_ansatz_parameters(num_qubits: int, depth: int) -> int:
    return num_ansatz_parameters_for_kind(num_qubits, depth, "hardware_efficient")


def num_ansatz_parameters_for_kind(num_qubits: int, depth: int, ansatz_kind: str) -> int:
    normalized = normalize_ansatz_kind(ansatz_kind)
    if normalized == "charge_conserving":
        return num_charge_conserving_parameters(num_qubits, depth)
    if normalized in {"hardware_efficient", "hardware_efficient_nnn"}:
        return 2 * num_qubits * (depth + 1)
    if normalized == "problem_inspired":
        return depth * (3 * num_qubits - 5)
    raise ValueError(f"Unhandled ansatz kind {ansatz_kind}")


def _apply_nearest_neighbor_cz(circuit: QuantumCircuit, num_qubits: int) -> None:
    for start in range(0, num_qubits - 1, 2):
        circuit.cz(start, start + 1)
    for start in range(1, num_qubits - 1, 2):
        circuit.cz(start, start + 1)


def _apply_next_nearest_neighbor_cz(circuit: QuantumCircuit, num_qubits: int) -> None:
    for start in range(0, num_qubits - 2):
        circuit.cz(start, start + 2)


def _append_rotation_layer(circuit: QuantumCircuit, num_qubits: int, flat_parameters: np.ndarray, cursor: int) -> int:
    for qubit in range(num_qubits):
        circuit.ry(float(flat_parameters[cursor]), qubit)
        circuit.rz(float(flat_parameters[cursor + 1]), qubit)
        cursor += 2
    return cursor


def build_hardware_efficient_ansatz(
    num_qubits: int,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> QuantumCircuit:
    flat_parameters = np.asarray(parameters, dtype=float).ravel()
    expected = num_ansatz_parameters_for_kind(num_qubits, depth, "hardware_efficient")
    if flat_parameters.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {flat_parameters.size}")

    circuit = create_reference_state(num_qubits, reference=reference) if include_reference else QuantumCircuit(num_qubits)
    cursor = _append_rotation_layer(circuit, num_qubits, flat_parameters, 0)
    for _ in range(depth):
        _apply_nearest_neighbor_cz(circuit, num_qubits)
        cursor = _append_rotation_layer(circuit, num_qubits, flat_parameters, cursor)
    return circuit


def build_hardware_efficient_nnn_ansatz(
    num_qubits: int,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> QuantumCircuit:
    flat_parameters = np.asarray(parameters, dtype=float).ravel()
    expected = num_ansatz_parameters_for_kind(num_qubits, depth, "hardware_efficient_nnn")
    if flat_parameters.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {flat_parameters.size}")

    circuit = create_reference_state(num_qubits, reference=reference) if include_reference else QuantumCircuit(num_qubits)
    cursor = _append_rotation_layer(circuit, num_qubits, flat_parameters, 0)
    for _ in range(depth):
        _apply_nearest_neighbor_cz(circuit, num_qubits)
        _apply_next_nearest_neighbor_cz(circuit, num_qubits)
        cursor = _append_rotation_layer(circuit, num_qubits, flat_parameters, cursor)
    return circuit


def rz_for_exp_minus_i_t_z(circuit: QuantumCircuit, qubit: int, angle: float) -> None:
    circuit.rz(2.0 * angle, qubit)


def apply_f_conjugated_z_evolution(circuit: QuantumCircuit, qubits_3: list[int], angle: float) -> None:
    f_gate = f_gate_optimal()
    circuit.append(f_gate, qubits_3)
    rz_for_exp_minus_i_t_z(circuit, qubits_3[1], angle)
    circuit.append(f_gate.inverse(), qubits_3)


def apply_bf_conjugated_z_evolution(circuit: QuantumCircuit, qubits_4: list[int], angle: float) -> None:
    b_gate = b_move_gate_3q()
    f_gate = f_gate_optimal()
    circuit.append(b_gate, qubits_4[0:3])
    circuit.append(f_gate, qubits_4[1:4])
    rz_for_exp_minus_i_t_z(circuit, qubits_4[2], angle)
    circuit.append(f_gate.inverse(), qubits_4[1:4])
    circuit.append(b_gate.inverse(), qubits_4[0:3])


def build_problem_inspired_ansatz(
    num_qubits: int,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> QuantumCircuit:
    flat_parameters = np.asarray(parameters, dtype=float).ravel()
    expected = num_ansatz_parameters_for_kind(num_qubits, depth, "problem_inspired")
    if flat_parameters.size != expected:
        raise ValueError(f"Expected {expected} parameters, received {flat_parameters.size}")

    circuit = create_reference_state(num_qubits, reference=reference) if include_reference else QuantumCircuit(num_qubits)
    cursor = 0
    for _ in range(depth):
        for qubit in range(num_qubits):
            circuit.ry(float(flat_parameters[cursor]), qubit)
            cursor += 1
        for start in range(0, num_qubits - 2):
            apply_f_conjugated_z_evolution(circuit, [start, start + 1, start + 2], float(flat_parameters[cursor]))
            cursor += 1
        for start in range(0, num_qubits - 3):
            apply_bf_conjugated_z_evolution(circuit, [start, start + 1, start + 2, start + 3], float(flat_parameters[cursor]))
            cursor += 1
    return circuit


def build_ansatz(
    num_qubits: int,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    ansatz_kind: str = "hardware_efficient",
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> QuantumCircuit:
    normalized = normalize_ansatz_kind(ansatz_kind)
    if normalized == "charge_conserving":
        return build_charge_conserving_ansatz(
            num_qubits,
            depth,
            parameters,
            include_reference=include_reference,
        )
    if normalized == "hardware_efficient":
        return build_hardware_efficient_ansatz(
            num_qubits,
            depth,
            parameters,
            include_reference=include_reference,
            reference=reference,
        )
    if normalized == "hardware_efficient_nnn":
        return build_hardware_efficient_nnn_ansatz(
            num_qubits,
            depth,
            parameters,
            include_reference=include_reference,
            reference=reference,
        )
    if normalized == "problem_inspired":
        return build_problem_inspired_ansatz(
            num_qubits,
            depth,
            parameters,
            include_reference=include_reference,
            reference=reference,
        )
    raise ValueError(f"Unhandled ansatz kind {ansatz_kind}")


def term_f_conjugated_z_3q(coeff: float = 1.0) -> SparsePauliOp:
    circuit = QuantumCircuit(3)
    f_gate = f_gate_optimal()
    circuit.append(f_gate, [0, 1, 2])
    circuit.z(1)
    circuit.append(f_gate.inverse(), [0, 1, 2])
    operator = SparsePauliOp.from_operator(Operator(circuit)).simplify(atol=1e-12)
    return SparsePauliOp(operator.paulis, np.real(operator.coeffs) * float(coeff))


def term_b_f_conjugated_z_4q(coeff: float = 1.0) -> SparsePauliOp:
    circuit = QuantumCircuit(4)
    b_gate = b_move_gate_3q()
    f_gate = f_gate_optimal()
    circuit.append(b_gate, [0, 1, 2])
    circuit.append(f_gate, [1, 2, 3])
    circuit.z(2)
    circuit.append(f_gate.inverse(), [1, 2, 3])
    circuit.append(b_gate.inverse(), [0, 1, 2])
    operator = SparsePauliOp.from_operator(Operator(circuit)).simplify(atol=1e-12)
    return SparsePauliOp(operator.paulis, np.real(operator.coeffs) * float(coeff))


def embed_local_operator(local_operator: SparsePauliOp, window: list[int], num_qubits: int) -> SparsePauliOp:
    if any(qubit < 0 or qubit >= num_qubits for qubit in window):
        raise ValueError("Window exceeds system size")
    if len(window) != len(set(window)):
        raise ValueError("Window contains duplicate qubits")

    padded_labels: list[str] = []
    padded_coeffs: list[complex] = []
    for local_label, coeff in local_operator.to_list():
        per_qubit = ["I"] * num_qubits
        local_per_qubit = list(local_label[::-1])
        for offset, qubit in enumerate(window):
            per_qubit[qubit] = local_per_qubit[offset]
        padded_labels.append("".join(per_qubit[::-1]))
        padded_coeffs.append(coeff)
    return SparsePauliOp(padded_labels, coeffs=np.asarray(padded_coeffs, dtype=complex)).simplify(atol=1e-12)


def hamiltonian_terms(spec: HamiltonianSpec) -> list[dict[str, Any]]:
    terms: list[dict[str, Any]] = []
    for start in range(0, spec.num_qubits - 2):
        window = [start, start + 1, start + 2]
        local_operator = term_f_conjugated_z_3q(coeff=spec.j_f3)
        terms.append({"kind": "F3", "window": window, "operator": local_operator})
    for start in range(0, spec.num_qubits - 3):
        window = [start, start + 1, start + 2, start + 3]
        local_operator = term_b_f_conjugated_z_4q(coeff=spec.j_bf4)
        terms.append({"kind": "BF4", "window": window, "operator": local_operator})
    return terms


def build_chain_hamiltonian(spec: HamiltonianSpec) -> SparsePauliOp:
    embedded_terms = [embed_local_operator(term["operator"], term["window"], spec.num_qubits) for term in hamiltonian_terms(spec)]
    if not embedded_terms:
        raise ValueError("No Hamiltonian terms were generated")
    total = embedded_terms[0]
    for operator in embedded_terms[1:]:
        total = total + operator
    return total.simplify(atol=1e-12)


def exact_ground_state(spec: HamiltonianSpec) -> dict[str, Any]:
    hamiltonian = build_chain_hamiltonian(spec)
    dense = hamiltonian.to_matrix()
    eigenvalues, eigenvectors = eigh(dense)
    ground_index = int(np.argmin(eigenvalues))
    return {
        "hamiltonian": hamiltonian,
        "eigenvalues": eigenvalues,
        "ground_energy": float(np.real(eigenvalues[ground_index])),
        "ground_state": eigenvectors[:, ground_index],
    }


def statevector_from_parameters(
    spec: HamiltonianSpec,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    ansatz_kind: str = "hardware_efficient",
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> Statevector:
    circuit = build_ansatz(
        spec.num_qubits,
        depth,
        parameters,
        ansatz_kind=ansatz_kind,
        include_reference=include_reference,
        reference=reference,
    )
    return Statevector.from_instruction(circuit)


def exact_energy_from_parameters(
    spec: HamiltonianSpec,
    depth: int,
    parameters: np.ndarray | list[float],
    *,
    ansatz_kind: str = "hardware_efficient",
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> float:
    ground_data = build_chain_hamiltonian(spec)
    statevector = statevector_from_parameters(
        spec,
        depth,
        parameters,
        ansatz_kind=ansatz_kind,
        include_reference=include_reference,
        reference=reference,
    )
    energy = statevector.expectation_value(ground_data)
    return float(np.real_if_close(energy))


def exact_energy_from_statevector(statevector: Statevector, hamiltonian: SparsePauliOp) -> float:
    energy = statevector.expectation_value(hamiltonian)
    return float(np.real_if_close(energy))


def state_overlap(statevector: Statevector, reference_state: np.ndarray) -> float:
    overlap = np.vdot(reference_state, statevector.data)
    return float(abs(overlap) ** 2)


def z_expectation_from_counts_on_qubit(counts: dict[str, int], qubit: int) -> float:
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0.0
    zero_counts = 0
    for bitstring, count in counts.items():
        if int(bitstring[::-1][qubit]) == 0:
            zero_counts += count
    prob_zero = zero_counts / total_counts
    return float(prob_zero - (1.0 - prob_zero))


def z_stderr_from_expectation(expectation_value: float, shots: int) -> float:
    if shots <= 1:
        return float("nan")
    variance = max(0.0, 1.0 - float(expectation_value) ** 2)
    return float(np.sqrt(variance / shots))


def add_measurement_basis_for_pauli(circuit: QuantumCircuit, pauli_label: str) -> None:
    for qubit, pauli in enumerate(pauli_label[::-1]):
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
        elif pauli in ("Z", "I"):
            continue
        else:
            raise ValueError(f"Unexpected Pauli label {pauli}")


def pauli_string_expectation_from_counts(pauli_label: str, counts: dict[str, int]) -> float:
    active_qubits = [index for index, pauli in enumerate(pauli_label[::-1]) if pauli != "I"]
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0.0
    expectation_value = 0.0
    for bitstring, count in counts.items():
        parity = 0
        for qubit in active_qubits:
            parity ^= int(bitstring[::-1][qubit])
        expectation_value += (1.0 if parity == 0 else -1.0) * (count / total_counts)
    return float(expectation_value)


def embed_local_pauli_label(local_label: str, window: list[int], num_qubits: int) -> str:
    if len(local_label) != len(window):
        raise ValueError("Local label length must match the target window")
    per_qubit = ["I"] * num_qubits
    local_per_qubit = list(local_label[::-1])
    for offset, qubit in enumerate(window):
        per_qubit[qubit] = local_per_qubit[offset]
    return "".join(per_qubit[::-1])


def globalize_local_operator(local_operator: SparsePauliOp, window: list[int], num_qubits: int) -> SparsePauliOp:
    labels: list[str] = []
    coeffs: list[complex] = []
    for local_label, coeff in local_operator.to_list():
        if abs(coeff) < 1e-12:
            continue
        labels.append(embed_local_pauli_label(local_label, window, num_qubits))
        coeffs.append(coeff)
    if not labels:
        return SparsePauliOp(["I" * num_qubits], coeffs=np.asarray([0.0], dtype=complex)).simplify(atol=1e-12)
    return SparsePauliOp(labels, coeffs=np.asarray(coeffs, dtype=complex)).simplify(atol=1e-12)


def qwc_basis_label(group_operator: SparsePauliOp, num_qubits: int) -> str:
    per_qubit_basis = ["I"] * num_qubits
    for pauli_label, _ in group_operator.to_list():
        for qubit, pauli in enumerate(pauli_label[::-1]):
            if pauli == "I":
                continue
            current = per_qubit_basis[qubit]
            if current in ("I", pauli):
                per_qubit_basis[qubit] = pauli
                continue
            raise ValueError(f"Group is not qubit-wise commuting on qubit {qubit}: {current} vs {pauli}")
    return "".join(per_qubit_basis[::-1])


def fr_measurement_circuit(base_circuit: QuantumCircuit, term_kind: str, window: list[int]) -> tuple[QuantumCircuit, int]:
    measurement_circuit = base_circuit.copy()
    if term_kind == "BF4":
        b_gate = b_move_gate_3q()
        f_gate = f_gate_optimal()
        measurement_circuit.append(b_gate, window[0:3])
        measurement_circuit.append(f_gate, window[1:4])
        z_qubit = window[2]
    elif term_kind == "F3":
        f_gate = f_gate_optimal()
        measurement_circuit.append(f_gate, window)
        z_qubit = window[1]
    else:
        raise ValueError(f"Unknown term kind {term_kind}")

    measurement_circuit.measure_all()
    return measurement_circuit, int(z_qubit)


def build_ps_circuits_for_local_op(
    state_circuit: QuantumCircuit,
    local_operator: SparsePauliOp,
    window: list[int],
    num_qubits: int,
) -> list[tuple[QuantumCircuit, complex, str]]:
    output: list[tuple[QuantumCircuit, complex, str]] = []
    for local_label, coeff in local_operator.to_list():
        if abs(coeff) < 1e-12:
            continue
        global_label = embed_local_pauli_label(local_label, window, num_qubits)
        measurement_circuit = state_circuit.copy()
        add_measurement_basis_for_pauli(measurement_circuit, global_label)
        measurement_circuit.measure_all()
        output.append((measurement_circuit, coeff, global_label))
    return output


def build_qwc_grouped_ps_circuits_for_local_op(
    state_circuit: QuantumCircuit,
    local_operator: SparsePauliOp,
    window: list[int],
    num_qubits: int,
) -> list[tuple[QuantumCircuit, SparsePauliOp, str]]:
    global_operator = globalize_local_operator(local_operator, window, num_qubits)
    grouped_operator_list = global_operator.group_commuting(qubit_wise=True)
    output: list[tuple[QuantumCircuit, SparsePauliOp, str]] = []
    for group_operator in grouped_operator_list:
        simplified_group = group_operator.simplify(atol=1e-12)
        if len(simplified_group) == 0:
            continue
        basis_label = qwc_basis_label(simplified_group, num_qubits)
        measurement_circuit = state_circuit.copy()
        add_measurement_basis_for_pauli(measurement_circuit, basis_label)
        measurement_circuit.measure_all()
        output.append((measurement_circuit, simplified_group, basis_label))
    return output


def _pad_bitstring(bitstring: str, width: int | None) -> str:
    if width is None:
        return str(bitstring)
    clean = str(bitstring).replace(" ", "")
    return clean.zfill(width)


def _quasi_to_counts(quasi_distribution: dict[Any, float], shots: int, width: int | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, probability in quasi_distribution.items():
        if probability is None:
            continue
        if isinstance(key, str):
            bitstring = _pad_bitstring(key, width)
        elif isinstance(key, numbers.Integral):
            bitstring = format(int(key), f"0{width}b") if width is not None else format(int(key), "b")
        else:
            bitstring = _pad_bitstring(str(key), width)
        counts[bitstring] = counts.get(bitstring, 0) + int(round(float(probability) * shots))
    return counts


def extract_counts_list_from_sampler_result(result: Any, shots: int, width: int | None = None) -> list[dict[str, int]]:
    if hasattr(result, "quasi_dists"):
        return [_quasi_to_counts(quasi, shots, width=width) for quasi in list(result.quasi_dists)]

    extracted: list[dict[str, int]] = []
    item_count = len(result)
    for index in range(item_count):
        item = result[index]
        if hasattr(item, "data") and hasattr(item.data, "meas") and hasattr(item.data.meas, "get_counts"):
            raw_counts = dict(item.data.meas.get_counts())
            if width is not None:
                raw_counts = {_pad_bitstring(key, width): int(value) for key, value in raw_counts.items()}
            extracted.append(raw_counts)
            continue
        if hasattr(item, "quasi_dists"):
            quasi_distributions = list(item.quasi_dists)
            if len(quasi_distributions) == 1:
                extracted.append(_quasi_to_counts(quasi_distributions[0], shots, width=width))
                continue
        raise RuntimeError("Unknown Sampler result structure")
    return extracted


def run_backend_circuits(backend: Any, circuits: list[QuantumCircuit], shots: int, optimization_level: int = 1) -> list[dict[str, int]]:
    if not circuits:
        return []
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=optimization_level)
    result = backend.run(transpiled_circuits, shots=shots).result()
    return [dict(result.get_counts(index)) for index in range(len(transpiled_circuits))]


def run_sampler_circuits(
    sampler: Any,
    circuits: list[QuantumCircuit],
    backend: Any,
    shots: int,
    optimization_level: int = 1,
) -> tuple[list[dict[str, int]], dict[str, Any]]:
    if not circuits:
        return [], {"job_id": None, "n_circuits": 0, "shots": int(shots), "wall_s": 0.0}
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=optimization_level)
    start = time.monotonic()
    job = sampler.run(transpiled_circuits, shots=shots)
    result = job.result()
    metadata = {
        "job_id": str(job.job_id()) if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", None)),
        "n_circuits": len(transpiled_circuits),
        "shots": int(shots),
        "wall_s": float(time.monotonic() - start),
    }
    width = transpiled_circuits[0].num_qubits
    return extract_counts_list_from_sampler_result(result, shots=shots, width=width), metadata


def submit_sampler_circuits(
    sampler: Any,
    circuits: list[QuantumCircuit],
    backend: Any,
    shots: int,
    optimization_level: int = 1,
) -> dict[str, Any]:
    if not circuits:
        return {
            "job_id": None,
            "n_circuits": 0,
            "shots": int(shots),
            "wall_s": 0.0,
            "submitted_only": True,
        }
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=optimization_level)
    start = time.monotonic()
    job = sampler.run(transpiled_circuits, shots=shots)
    return {
        "job_id": str(job.job_id()) if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", None)),
        "n_circuits": len(transpiled_circuits),
        "shots": int(shots),
        "wall_s": float(time.monotonic() - start),
        "submitted_only": True,
    }


def _estimate_fr_from_counts(counts: dict[str, int], coefficient: float, z_qubit: int, shots: int) -> tuple[float, float]:
    expectation_value = z_expectation_from_counts_on_qubit(counts, z_qubit)
    return coefficient * expectation_value, abs(coefficient) * z_stderr_from_expectation(expectation_value, shots)


def _estimate_ps_from_counts(
    counts_list: list[dict[str, int]],
    decomposition: list[tuple[QuantumCircuit, complex, str]],
    shots: int,
) -> tuple[float, float]:
    estimate = 0.0
    variance = 0.0
    for counts, (_, coeff, pauli_label) in zip(counts_list, decomposition):
        expectation_value = pauli_string_expectation_from_counts(pauli_label, counts)
        coeff_real = float(np.real_if_close(coeff))
        estimate += coeff_real * expectation_value
        variance += (coeff_real * z_stderr_from_expectation(expectation_value, shots)) ** 2
    return float(estimate), float(np.sqrt(max(variance, 0.0)))


def grouped_pauli_stats_from_counts(counts: dict[str, int], group_operator: SparsePauliOp) -> tuple[float, float, int]:
    total_counts = sum(counts.values())
    if total_counts == 0:
        return 0.0, 0.0, 0

    prepared_terms: list[tuple[float, list[int]]] = []
    for pauli_label, coeff in group_operator.to_list():
        coeff_real = float(np.real_if_close(coeff))
        active_qubits = [index for index, pauli in enumerate(pauli_label[::-1]) if pauli != "I"]
        prepared_terms.append((coeff_real, active_qubits))

    mean = 0.0
    second_moment = 0.0
    for bitstring, count in counts.items():
        reversed_bitstring = bitstring[::-1]
        observable_value = 0.0
        for coeff_real, active_qubits in prepared_terms:
            parity = 0
            for qubit in active_qubits:
                parity ^= int(reversed_bitstring[qubit])
            observable_value += coeff_real * (1.0 if parity == 0 else -1.0)
        probability = count / total_counts
        mean += observable_value * probability
        second_moment += (observable_value ** 2) * probability

    variance = max(0.0, second_moment - mean ** 2)
    return float(mean), float(variance), int(total_counts)


def _estimate_grouped_ps_from_counts(
    counts_list: list[dict[str, int]],
    grouped_decomposition: list[tuple[QuantumCircuit, SparsePauliOp, str]],
) -> tuple[float, float]:
    estimate = 0.0
    variance = 0.0
    for counts, (_, group_operator, _) in zip(counts_list, grouped_decomposition):
        group_estimate, group_variance, total_counts = grouped_pauli_stats_from_counts(counts, group_operator)
        estimate += group_estimate
        if total_counts > 0:
            variance += group_variance / total_counts
    return float(estimate), float(np.sqrt(max(variance, 0.0)))


def estimate_energy_fr_backend(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term: int,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    circuits: list[QuantumCircuit] = []
    descriptors: list[tuple[float, int]] = []
    for term in hamiltonian_terms(spec):
        measurement_circuit, z_qubit = fr_measurement_circuit(state_circuit, term["kind"], term["window"])
        circuits.append(measurement_circuit)
        coefficient = spec.j_f3 if term["kind"] == "F3" else spec.j_bf4
        descriptors.append((coefficient, z_qubit))
    counts_list = run_backend_circuits(backend, circuits, shots=shots_per_term, optimization_level=optimization_level)
    estimate = 0.0
    variance = 0.0
    for counts, (coefficient, z_qubit) in zip(counts_list, descriptors):
        term_estimate, term_stderr = _estimate_fr_from_counts(counts, coefficient, z_qubit, shots_per_term)
        estimate += term_estimate
        variance += term_stderr ** 2
    return {
        "method": "FR",
        "estimate": float(estimate),
        "stderr": float(np.sqrt(max(variance, 0.0))),
        "shots_per_term": int(shots_per_term),
        "num_circuits": len(circuits),
    }


def estimate_energy_ps_backend(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term_total: int,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    total_estimate = 0.0
    total_variance = 0.0
    total_circuits = 0
    total_groups = 0
    total_pauli_terms = 0
    for term in hamiltonian_terms(spec):
        grouped_decomposition = build_qwc_grouped_ps_circuits_for_local_op(
            state_circuit,
            term["operator"],
            term["window"],
            spec.num_qubits,
        )
        total_circuits += len(grouped_decomposition)
        total_groups += len(grouped_decomposition)
        total_pauli_terms += sum(len(group_operator) for _, group_operator, _ in grouped_decomposition)
        shots = max(1, int(shots_per_term_total) // max(1, len(grouped_decomposition)))
        counts_list = run_backend_circuits(
            backend,
            [circuit for circuit, _, _ in grouped_decomposition],
            shots=shots,
            optimization_level=optimization_level,
        )
        estimate, stderr = _estimate_grouped_ps_from_counts(counts_list, grouped_decomposition)
        total_estimate += estimate
        total_variance += stderr ** 2
    return {
        "method": "PS",
        "grouping": "qwc_equal_shots",
        "estimate": float(total_estimate),
        "stderr": float(np.sqrt(max(total_variance, 0.0))),
        "shots_per_term_total": int(shots_per_term_total),
        "num_circuits": int(total_circuits),
        "num_groups": int(total_groups),
        "num_pauli_terms": int(total_pauli_terms),
    }


def estimate_energy_fr_sampler(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term: int,
    sampler: Any,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    circuits: list[QuantumCircuit] = []
    descriptors: list[tuple[float, int]] = []
    for term in hamiltonian_terms(spec):
        measurement_circuit, z_qubit = fr_measurement_circuit(state_circuit, term["kind"], term["window"])
        circuits.append(measurement_circuit)
        coefficient = spec.j_f3 if term["kind"] == "F3" else spec.j_bf4
        descriptors.append((coefficient, z_qubit))
    counts_list, metadata = run_sampler_circuits(
        sampler,
        circuits,
        backend=backend,
        shots=shots_per_term,
        optimization_level=optimization_level,
    )
    estimate = 0.0
    variance = 0.0
    for counts, (coefficient, z_qubit) in zip(counts_list, descriptors):
        term_estimate, term_stderr = _estimate_fr_from_counts(counts, coefficient, z_qubit, shots_per_term)
        estimate += term_estimate
        variance += term_stderr ** 2
    metadata.update(
        {
            "method": "FR",
            "estimate": float(estimate),
            "stderr": float(np.sqrt(max(variance, 0.0))),
            "shots_per_term": int(shots_per_term),
        }
    )
    return metadata


def submit_energy_fr_sampler(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term: int,
    sampler: Any,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    circuits: list[QuantumCircuit] = []
    for term in hamiltonian_terms(spec):
        measurement_circuit, _ = fr_measurement_circuit(state_circuit, term["kind"], term["window"])
        circuits.append(measurement_circuit)
    metadata = submit_sampler_circuits(
        sampler,
        circuits,
        backend=backend,
        shots=shots_per_term,
        optimization_level=optimization_level,
    )
    metadata.update(
        {
            "method": "FR",
            "shots_per_term": int(shots_per_term),
        }
    )
    return metadata


def estimate_energy_ps_sampler(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term_total: int,
    sampler: Any,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    total_estimate = 0.0
    total_variance = 0.0
    total_circuits = 0
    total_groups = 0
    total_pauli_terms = 0
    job_batches: list[dict[str, Any]] = []
    for term in hamiltonian_terms(spec):
        grouped_decomposition = build_qwc_grouped_ps_circuits_for_local_op(
            state_circuit,
            term["operator"],
            term["window"],
            spec.num_qubits,
        )
        shots = max(1, int(shots_per_term_total) // max(1, len(grouped_decomposition)))
        counts_list, metadata = run_sampler_circuits(
            sampler,
            [circuit for circuit, _, _ in grouped_decomposition],
            backend=backend,
            shots=shots,
            optimization_level=optimization_level,
        )
        estimate, stderr = _estimate_grouped_ps_from_counts(counts_list, grouped_decomposition)
        total_estimate += estimate
        total_variance += stderr ** 2
        total_circuits += len(grouped_decomposition)
        total_groups += len(grouped_decomposition)
        total_pauli_terms += sum(len(group_operator) for _, group_operator, _ in grouped_decomposition)
        job_batches.append(metadata)
    return {
        "method": "PS",
        "grouping": "qwc_equal_shots",
        "estimate": float(total_estimate),
        "stderr": float(np.sqrt(max(total_variance, 0.0))),
        "shots_per_term_total": int(shots_per_term_total),
        "num_circuits": int(total_circuits),
        "num_groups": int(total_groups),
        "num_pauli_terms": int(total_pauli_terms),
        "job_batches": job_batches,
    }


def submit_energy_ps_sampler(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    shots_per_term_total: int,
    sampler: Any,
    backend: Any,
    optimization_level: int = 1,
) -> dict[str, Any]:
    job_batches: list[dict[str, Any]] = []
    total_circuits = 0
    total_groups = 0
    total_pauli_terms = 0
    for term in hamiltonian_terms(spec):
        grouped_decomposition = build_qwc_grouped_ps_circuits_for_local_op(
            state_circuit,
            term["operator"],
            term["window"],
            spec.num_qubits,
        )
        shots = max(1, int(shots_per_term_total) // max(1, len(grouped_decomposition)))
        metadata = submit_sampler_circuits(
            sampler,
            [circuit for circuit, _, _ in grouped_decomposition],
            backend=backend,
            shots=shots,
            optimization_level=optimization_level,
        )
        total_circuits += len(grouped_decomposition)
        total_groups += len(grouped_decomposition)
        total_pauli_terms += sum(len(group_operator) for _, group_operator, _ in grouped_decomposition)
        job_batches.append(metadata)
    return {
        "method": "PS",
        "grouping": "qwc_equal_shots",
        "shots_per_term_total": int(shots_per_term_total),
        "num_circuits": int(total_circuits),
        "num_groups": int(total_groups),
        "num_pauli_terms": int(total_pauli_terms),
        "submitted_only": True,
        "job_batches": job_batches,
    }


def optimize_vqe_exact(
    spec: HamiltonianSpec,
    depth: int,
    initial_parameters: np.ndarray,
    *,
    maxiter: int = 200,
    optimizer_method: str = "COBYLA",
    ansatz_kind: str = "hardware_efficient",
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> dict[str, Any]:
    history: list[OptimizationRecord] = []
    hamiltonian = build_chain_hamiltonian(spec)

    def objective(flat_parameters: np.ndarray) -> float:
        statevector = statevector_from_parameters(
            spec,
            depth,
            flat_parameters,
            ansatz_kind=ansatz_kind,
            include_reference=include_reference,
            reference=reference,
        )
        energy = exact_energy_from_statevector(statevector, hamiltonian)
        history.append(OptimizationRecord(iteration=len(history), energy=energy, method="EXACT"))
        return energy

    result = minimize(
        objective,
        np.asarray(initial_parameters, dtype=float),
        method=optimizer_method,
        options={"maxiter": int(maxiter)},
    )
    return {
        "result": result,
        "history": [asdict(item) for item in history],
    }


def optimize_vqe_measurement(
    spec: HamiltonianSpec,
    depth: int,
    initial_parameters: np.ndarray,
    *,
    method: str,
    maxiter: int,
    optimizer_method: str,
    budget: EstimatorBudget,
    estimator: Any,
    ansatz_kind: str = "hardware_efficient",
    include_reference: bool = True,
    reference: str = "fusion_valid",
) -> dict[str, Any]:
    history: list[OptimizationRecord] = []
    normalized_method = method.upper()
    normalized_optimizer = str(optimizer_method).strip().upper()

    def objective(flat_parameters: np.ndarray) -> float:
        state_circuit = build_ansatz(
            spec.num_qubits,
            depth,
            flat_parameters,
            ansatz_kind=ansatz_kind,
            include_reference=include_reference,
            reference=reference,
        )
        if normalized_method == "FR":
            outcome = estimator(state_circuit, spec, budget.fr_shots_per_term)
        elif normalized_method == "PS":
            outcome = estimator(state_circuit, spec, budget.ps_shots_per_term_total)
        else:
            raise ValueError(f"Unknown optimization method {method}")
        energy = float(outcome["estimate"])
        history.append(OptimizationRecord(iteration=len(history), energy=energy, method=normalized_method))
        return energy

    if normalized_optimizer == "SPSA":
        result = _spsa_minimize(objective, np.asarray(initial_parameters, dtype=float), maxiter=int(maxiter))
    else:
        result = minimize(
            objective,
            np.asarray(initial_parameters, dtype=float),
            method=optimizer_method,
            options={"maxiter": int(maxiter)},
        )
    return {
        "result": result,
        "history": [asdict(item) for item in history],
    }


def _spsa_minimize(
    objective: Any,
    initial_parameters: np.ndarray,
    *,
    maxiter: int,
    a: float = 0.08,
    c: float = 0.12,
    alpha: float = 0.602,
    gamma: float = 0.101,
    stability_constant: float | None = None,
    seed: int = 1234,
) -> OptimizeResult:
    rng = np.random.default_rng(seed)
    theta = np.asarray(initial_parameters, dtype=float).copy()
    best_x = theta.copy()
    best_fun = float(objective(theta))
    nfev = 1
    stability = float(stability_constant if stability_constant is not None else max(1, maxiter // 5))

    for iteration in range(maxiter):
        ak = float(a / ((iteration + 1 + stability) ** alpha))
        ck = float(c / ((iteration + 1) ** gamma))
        delta = rng.choice([-1.0, 1.0], size=theta.shape)
        y_plus = float(objective(theta + ck * delta))
        y_minus = float(objective(theta - ck * delta))
        nfev += 2
        gradient = ((y_plus - y_minus) / (2.0 * ck)) * delta
        theta = theta - ak * gradient
        value = float(objective(theta))
        nfev += 1
        if value < best_fun:
            best_fun = value
            best_x = theta.copy()

    return OptimizeResult(
        x=best_x,
        fun=float(best_fun),
        success=True,
        status=0,
        message="SPSA completed",
        nfev=int(nfev),
        nit=int(maxiter),
    )


def initial_parameter_vector(
    num_qubits: int,
    depth: int,
    seed: int = 1234,
    scale: float = 0.1,
    *,
    ansatz_kind: str = "hardware_efficient",
    scheme: str = "normal",
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = num_ansatz_parameters_for_kind(num_qubits, depth, ansatz_kind)
    normalized_scheme = str(scheme).strip().lower().replace("-", "_")
    if normalized_scheme == "zeros":
        return np.zeros(size, dtype=float)
    if normalized_scheme == "uniform":
        return rng.uniform(low=-scale, high=scale, size=size)
    if normalized_scheme == "normal":
        return scale * rng.standard_normal(size)
    raise ValueError(f"Unknown initialization scheme {scheme}")


def serializable_result_payload(payload: dict[str, Any]) -> dict[str, Any]:
    serializable = dict(payload)
    if "result" in serializable:
        result = serializable["result"]
        serializable["result"] = {
            "x": np.asarray(result.x, dtype=float).tolist(),
            "fun": float(result.fun),
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", -1)),
            "nit": int(getattr(result, "nit", -1)),
        }
    return serializable


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)