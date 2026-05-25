from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
DEPENDENCIES_DIR = PROJECT_ROOT / "dependencies"
VQE_DIR = DEPENDENCIES_DIR / "VQE_FR_PS"

if str(VQE_DIR) not in sys.path:
    sys.path.insert(0, str(VQE_DIR))

from anyon_vqe import (  # noqa: E402
    HamiltonianSpec,
    add_measurement_basis_for_pauli,
    b_move_gate_3q,
    build_ansatz,
    build_chain_hamiltonian,
    embed_local_operator,
    f_gate_optimal,
    fr_measurement_circuit,
    hamiltonian_terms,
    qwc_basis_label,
)


DEFAULT_LOCKED_STATE_FILENAMES: dict[int, str] = {
    5: "noiseless_hardware_efficient_nnn_5q_d3_powell.json",
    7: "noiseless_hardware_efficient_nnn_7q_d3_powell.json",
    9: "noiseless_hardware_efficient_nnn_9q_d3_powell.json",
    11: "noiseless_hardware_efficient_nnn_11q_d4_powell_tight.json",
}

LOCKED_STATE_DIR = VQE_DIR / "locked_state_payloads"


_TRANSPILE_CACHE: dict[tuple[int, str, int], QuantumCircuit] = {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def portable_path_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def default_locked_state_map() -> dict[int, Path]:
    mapping = {size: LOCKED_STATE_DIR / filename for size, filename in DEFAULT_LOCKED_STATE_FILENAMES.items()}
    missing = [path for path in mapping.values() if not path.exists()]
    if missing:
        missing_text = ", ".join(str(path.name) for path in missing)
        raise FileNotFoundError(f"Missing bundled locked-state payloads: {missing_text}")
    return mapping


def equal_partition(total: int, bins: int) -> list[int]:
    if bins <= 0:
        raise ValueError("bins must be positive")
    if total < 0:
        raise ValueError("total must be non-negative")
    base = int(total) // int(bins)
    remainder = int(total) % int(bins)
    return [base + (1 if index < remainder else 0) for index in range(bins)]


def circuit_resources(circuit: QuantumCircuit) -> dict[str, Any]:
    ops = circuit.count_ops()
    non_unitary = {"barrier", "delay", "measure", "reset"}
    twoq_gate_names = {"cx", "cz", "ecr", "iswap", "rzz", "swap"}
    twoq = 0
    singleq = 0
    for name, count in ops.items():
        if name in twoq_gate_names:
            twoq += int(count)
        elif name not in non_unitary:
            singleq += int(count)
    return {
        "depth": int(circuit.depth() or 0),
        "ops": {str(name): int(count) for name, count in ops.items()},
        "singleq_count_rough": int(singleq),
        "twoq_count_rough": int(twoq),
    }


def create_fusion_valid_reference(num_qubits: int) -> QuantumCircuit:
    circuit = QuantumCircuit(num_qubits, name="init")
    for qubit in range(num_qubits):
        circuit.x(qubit)
    return circuit


def rz_for_exp_minus_i_t_z(circuit: QuantumCircuit, qubit: int, time_value: float) -> None:
    circuit.rz(2.0 * float(time_value), qubit)


def apply_f_conjugated_z_evolution(circuit: QuantumCircuit, qubits_3: list[int], time_value: float) -> None:
    gate = f_gate_optimal()
    circuit.append(gate, qubits_3)
    rz_for_exp_minus_i_t_z(circuit, qubits_3[1], time_value)
    circuit.append(gate.inverse(), qubits_3)


def apply_bf_conjugated_z_evolution(circuit: QuantumCircuit, qubits_4: list[int], time_value: float) -> None:
    b_gate = b_move_gate_3q()
    f_gate = f_gate_optimal()
    circuit.append(b_gate, qubits_4[0:3])
    circuit.append(f_gate, qubits_4[1:4])
    rz_for_exp_minus_i_t_z(circuit, qubits_4[2], time_value)
    circuit.append(f_gate.inverse(), qubits_4[1:4])
    circuit.append(b_gate.inverse(), qubits_4[0:3])


def build_trotter_circuit(num_qubits: int, num_steps: int, dt: float, j1: float, j2: float) -> QuantumCircuit:
    circuit = create_fusion_valid_reference(num_qubits)
    for _ in range(int(num_steps)):
        for start in range(0, num_qubits - 2):
            apply_f_conjugated_z_evolution(circuit, [start, start + 1, start + 2], time_value=dt * j1)
        for start in range(0, num_qubits - 3):
            apply_bf_conjugated_z_evolution(circuit, [start, start + 1, start + 2, start + 3], time_value=dt * j2)
    return circuit


def load_locked_state_payload(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def locked_state_to_circuit(path: Path) -> tuple[dict[str, Any], HamiltonianSpec, QuantumCircuit]:
    payload = load_locked_state_payload(path)
    spec_payload = payload["spec"]
    parameters = payload["exact_vqe"]["result"]["x"]
    spec = HamiltonianSpec(
        num_qubits=int(spec_payload["num_qubits"]),
        j_f3=float(spec_payload["j_f3"]),
        j_bf4=float(spec_payload["j_bf4"]),
    )
    circuit = build_ansatz(
        spec.num_qubits,
        int(spec_payload["depth"]),
        parameters,
        ansatz_kind=str(spec_payload["ansatz"]),
        reference=str(spec_payload.get("reference", "fusion_valid")),
    )
    return payload, spec, circuit


def _real_coeff_map(operator: Any) -> dict[str, float]:
    output: dict[str, float] = {}
    for label, coeff in operator.to_list():
        coeff_real = float(np.real_if_close(coeff))
        if abs(coeff_real) < 1e-12:
            continue
        output[str(label)] = coeff_real
    return output


def local_term_descriptors(spec: HamiltonianSpec) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for index, term in enumerate(hamiltonian_terms(spec)):
        global_operator = embed_local_operator(term["operator"], term["window"], spec.num_qubits)
        output.append(
            {
                "index": int(index),
                "kind": str(term["kind"]),
                "window": list(term["window"]),
                "operator": global_operator,
                "coeff_by_label": _real_coeff_map(global_operator),
            }
        )
    return output


def global_qwc_group_descriptors(spec: HamiltonianSpec, local_terms: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    if local_terms is None:
        local_terms = local_term_descriptors(spec)
    hamiltonian = build_chain_hamiltonian(spec)
    output: list[dict[str, Any]] = []
    for index, raw_group in enumerate(hamiltonian.group_commuting(qubit_wise=True)):
        group_operator = raw_group.simplify(atol=1e-12)
        coeff_by_label = _real_coeff_map(group_operator)
        if not coeff_by_label:
            continue
        term_coeffs: dict[int, dict[str, float]] = {}
        for term in local_terms:
            overlap = {label: coeff for label, coeff in term["coeff_by_label"].items() if label in coeff_by_label}
            if overlap:
                term_coeffs[int(term["index"])] = overlap
        output.append(
            {
                "index": int(index),
                "basis_label": qwc_basis_label(group_operator, spec.num_qubits),
                "operator": group_operator,
                "coeff_by_label": coeff_by_label,
                "term_coeffs": term_coeffs,
            }
        )
    return output


def fr_basis_circuit(base_circuit: QuantumCircuit, term_kind: str, window: list[int]) -> tuple[QuantumCircuit, int]:
    circuit = base_circuit.copy()
    if term_kind == "BF4":
        circuit.append(b_move_gate_3q(), window[0:3])
        circuit.append(f_gate_optimal(), window[1:4])
        z_qubit = int(window[2])
    elif term_kind == "F3":
        circuit.append(f_gate_optimal(), window)
        z_qubit = int(window[1])
    else:
        raise ValueError(f"Unknown term kind {term_kind}")
    return circuit, z_qubit


def basis_probabilities(base_circuit: QuantumCircuit, basis_label: str) -> dict[str, float]:
    circuit = base_circuit.copy()
    add_measurement_basis_for_pauli(circuit, basis_label)
    probabilities = Statevector.from_instruction(circuit).probabilities()
    width = circuit.num_qubits
    output: dict[str, float] = {}
    for index, probability in enumerate(probabilities):
        if float(probability) < 1e-15:
            continue
        output[format(index, f"0{width}b")] = float(probability)
    return output


def parity_value(pauli_label: str, bitstring: str) -> float:
    parity = 0
    for qubit, pauli in enumerate(pauli_label[::-1]):
        if pauli == "I":
            continue
        parity ^= int(bitstring[::-1][qubit])
    return 1.0 if parity == 0 else -1.0


def _group_term_value_vector(bitstring: str, term_coeffs: dict[int, dict[str, float]], num_terms: int) -> np.ndarray:
    values = np.zeros(num_terms, dtype=float)
    for term_index, coeff_by_label in term_coeffs.items():
        total = 0.0
        for label, coeff in coeff_by_label.items():
            total += float(coeff) * parity_value(label, bitstring)
        values[int(term_index)] = total
    return values


def exact_group_term_moments(
    probabilities: dict[str, float],
    term_coeffs: dict[int, dict[str, float]],
    num_terms: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_vector = np.zeros(num_terms, dtype=float)
    second_moment = np.zeros((num_terms, num_terms), dtype=float)
    for bitstring, probability in probabilities.items():
        values = _group_term_value_vector(bitstring, term_coeffs, num_terms)
        mean_vector += float(probability) * values
        second_moment += float(probability) * np.outer(values, values)
    covariance = second_moment - np.outer(mean_vector, mean_vector)
    covariance = np.maximum(covariance, covariance.T)
    return mean_vector, covariance


def sample_group_term_statistics_from_counts(
    counts: dict[str, int],
    term_coeffs: dict[int, dict[str, float]],
    num_terms: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    total_counts = sum(int(value) for value in counts.values())
    if total_counts <= 0:
        zeros = np.zeros(num_terms, dtype=float)
        matrix = np.zeros((num_terms, num_terms), dtype=float)
        return zeros, matrix, matrix
    mean_vector = np.zeros(num_terms, dtype=float)
    second_moment = np.zeros((num_terms, num_terms), dtype=float)
    for bitstring, count in counts.items():
        values = _group_term_value_vector(bitstring, term_coeffs, num_terms)
        weight = float(count) / float(total_counts)
        mean_vector += weight * values
        second_moment += weight * np.outer(values, values)
    population_covariance = second_moment - np.outer(mean_vector, mean_vector)
    if total_counts > 1:
        sample_covariance = population_covariance * (float(total_counts) / float(total_counts - 1))
    else:
        sample_covariance = np.zeros_like(population_covariance)
    covariance_of_mean = sample_covariance / float(total_counts)
    return mean_vector, sample_covariance, covariance_of_mean


def _run_counts_batch(
    backend: Any,
    circuits: list[QuantumCircuit],
    shots: int,
    *,
    optimization_level: int,
    seed: int | None = None,
) -> list[dict[str, int]]:
    if not circuits:
        return []
    backend_name = getattr(backend, "name", None)
    if callable(backend_name):
        backend_name = backend_name()
    backend_key = str(backend_name or backend.__class__.__name__)
    transpiled: list[QuantumCircuit] = []
    for circuit in circuits:
        cache_key = (id(circuit), backend_key, int(optimization_level))
        cached = _TRANSPILE_CACHE.get(cache_key)
        if cached is None:
            cached = transpile(circuit, backend=backend, optimization_level=optimization_level)
            _TRANSPILE_CACHE[cache_key] = cached
        transpiled.append(cached)
    run_kwargs: dict[str, Any] = {"shots": int(shots)}
    if seed is not None:
        run_kwargs["seed_simulator"] = int(seed)
    result = backend.run(transpiled, **run_kwargs).result()
    return [dict(result.get_counts(index)) for index in range(len(transpiled))]


def prepare_fr_measurement_plan(state_circuit: QuantumCircuit, spec: HamiltonianSpec) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    measurements: list[tuple[dict[str, Any], int, float, QuantumCircuit]] = []
    for term in local_terms:
        circuit, z_qubit = fr_measurement_circuit(state_circuit, term["kind"], term["window"])
        coefficient = float(spec.j_f3 if term["kind"] == "F3" else spec.j_bf4)
        measurements.append((term, int(z_qubit), coefficient, circuit))
    return {
        "local_terms": local_terms,
        "measurements": measurements,
    }


def prepare_ps_qwc_measurement_plan(state_circuit: QuantumCircuit, spec: HamiltonianSpec) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    groups = global_qwc_group_descriptors(spec, local_terms=local_terms)
    measurements: list[tuple[int, dict[str, Any], QuantumCircuit]] = []
    for index, group in enumerate(groups):
        circuit = state_circuit.copy()
        add_measurement_basis_for_pauli(circuit, group["basis_label"])
        circuit.measure_all()
        measurements.append((index, group, circuit))
    return {
        "local_terms": local_terms,
        "groups": groups,
        "measurements": measurements,
    }


def exact_hamiltonian_energy(state_circuit: QuantumCircuit, spec: HamiltonianSpec) -> float:
    hamiltonian = build_chain_hamiltonian(spec)
    return float(np.real(Statevector.from_instruction(state_circuit).expectation_value(hamiltonian)))


def exact_fr_theory(state_circuit: QuantumCircuit, spec: HamiltonianSpec, total_shots: int) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    shot_allocations = equal_partition(total_shots, len(local_terms))
    term_means = np.zeros(len(local_terms), dtype=float)
    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    term_rows: list[dict[str, Any]] = []
    for term, shots_allocated in zip(local_terms, shot_allocations):
        basis_circuit, z_qubit = fr_basis_circuit(state_circuit, term["kind"], term["window"])
        probabilities = Statevector.from_instruction(basis_circuit).probabilities()
        z_mean = 0.0
        for index, probability in enumerate(probabilities):
            bitstring = format(index, f"0{spec.num_qubits}b")
            sign = 1.0 if bitstring[::-1][z_qubit] == "0" else -1.0
            z_mean += sign * float(probability)
        coefficient = float(spec.j_f3 if term["kind"] == "F3" else spec.j_bf4)
        mean_value = coefficient * z_mean
        single_shot_variance = max(0.0, coefficient * coefficient - mean_value * mean_value)
        if shots_allocated > 0:
            covariance[int(term["index"]), int(term["index"])] = single_shot_variance / float(shots_allocated)
        term_means[int(term["index"])] = mean_value
        term_rows.append(
            {
                "term_index": int(term["index"]),
                "kind": term["kind"],
                "window": list(term["window"]),
                "shots_allocated": int(shots_allocated),
                "exact_mean": float(mean_value),
                "single_shot_variance": float(single_shot_variance),
            }
        )
    return {
        "method": "FR",
        "budget_total": int(total_shots),
        "num_observables": len(local_terms),
        "observable_shots": [int(value) for value in shot_allocations],
        "local_terms": term_rows,
        "local_term_means": [float(value) for value in term_means],
        "covariance_local_terms": covariance.tolist(),
        "energy_mean": float(term_means.sum()),
        "energy_variance": float(covariance.sum()),
        "energy_stderr": float(math.sqrt(max(covariance.sum(), 0.0))),
    }


def exact_ps_qwc_theory(state_circuit: QuantumCircuit, spec: HamiltonianSpec, total_shots: int) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    groups = global_qwc_group_descriptors(spec, local_terms=local_terms)
    shot_allocations = equal_partition(total_shots, len(groups))
    term_means = np.zeros(len(local_terms), dtype=float)
    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    group_rows: list[dict[str, Any]] = []
    for group, shots_allocated in zip(groups, shot_allocations):
        probabilities = basis_probabilities(state_circuit, group["basis_label"])
        group_mean, group_covariance = exact_group_term_moments(
            probabilities,
            group["term_coeffs"],
            len(local_terms),
        )
        term_means += group_mean
        if shots_allocated > 0:
            covariance += group_covariance / float(shots_allocated)
        group_rows.append(
            {
                "group_index": int(group["index"]),
                "basis_label": group["basis_label"],
                "shots_allocated": int(shots_allocated),
                "num_pauli_terms": len(group["coeff_by_label"]),
                "num_local_terms": len(group["term_coeffs"]),
                "single_shot_energy_variance": float(group_covariance.sum()),
            }
        )
    return {
        "method": "PS_QWC",
        "budget_total": int(total_shots),
        "num_groups": len(groups),
        "group_shots": [int(value) for value in shot_allocations],
        "groups": group_rows,
        "local_term_means": [float(value) for value in term_means],
        "covariance_local_terms": covariance.tolist(),
        "energy_mean": float(term_means.sum()),
        "energy_variance": float(covariance.sum()),
        "energy_stderr": float(math.sqrt(max(covariance.sum(), 0.0))),
    }


def exact_strategy_analysis(state_circuit: QuantumCircuit, spec: HamiltonianSpec, total_shots: int) -> dict[str, Any]:
    exact_energy = exact_hamiltonian_energy(state_circuit, spec)
    fr_payload = exact_fr_theory(state_circuit, spec, total_shots=total_shots)
    ps_payload = exact_ps_qwc_theory(state_circuit, spec, total_shots=total_shots)
    fr_variance = float(fr_payload["energy_variance"])
    ps_variance = float(ps_payload["energy_variance"])
    return {
        "exact_energy": float(exact_energy),
        "fr": fr_payload,
        "ps_qwc": ps_payload,
        "winner_by_sampling_variance": "FR" if fr_variance < ps_variance else "PS_QWC",
        "variance_gap": float(fr_variance - ps_variance),
    }


def sample_fr_hamiltonian_estimate(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    total_shots: int,
    *,
    backend: Any,
    optimization_level: int = 1,
    seed: int | None = None,
    prepared_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prepared_plan is None:
        prepared_plan = prepare_fr_measurement_plan(state_circuit, spec)
    local_terms = list(prepared_plan["local_terms"])
    measurements = list(prepared_plan["measurements"])
    shot_allocations = equal_partition(total_shots, len(local_terms))
    term_rows: list[dict[str, Any] | None] = [None] * len(local_terms)
    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    batch_map: dict[int, list[tuple[dict[str, Any], int, float, QuantumCircuit]]] = {}
    for (term, z_qubit, coefficient, circuit), shots_allocated in zip(measurements, shot_allocations):
        batch_map.setdefault(int(shots_allocated), []).append((term, int(z_qubit), coefficient, circuit))

    for batch_index, (shots_allocated, batch) in enumerate(sorted(batch_map.items())):
        if shots_allocated <= 0:
            continue
        counts_list = _run_counts_batch(
            backend,
            [item[3] for item in batch],
            shots_allocated,
            optimization_level=optimization_level,
            seed=None if seed is None else int(seed) + batch_index,
        )
        for counts, (term, z_qubit, coefficient, _) in zip(counts_list, batch):
            shots_used = max(1, sum(int(value) for value in counts.values()))
            z_mean = 0.0
            for bitstring, count in counts.items():
                sign = 1.0 if bitstring[::-1][z_qubit] == "0" else -1.0
                z_mean += sign * (float(count) / float(shots_used))
            estimate = coefficient * z_mean
            if shots_used > 1:
                sample_variance = max(0.0, coefficient * coefficient - estimate * estimate) * (float(shots_used) / float(shots_used - 1))
                variance_of_mean = sample_variance / float(shots_used)
            else:
                variance_of_mean = 0.0
            covariance[int(term["index"]), int(term["index"])] = float(variance_of_mean)
            term_rows[int(term["index"])] = {
                "term_index": int(term["index"]),
                "kind": term["kind"],
                "window": list(term["window"]),
                "shots_allocated": int(shots_allocated),
                "shots_used": int(shots_used),
                "estimate": float(estimate),
                "variance_of_mean": float(variance_of_mean),
                "stderr": float(math.sqrt(max(variance_of_mean, 0.0))),
            }

    term_payloads = [row for row in term_rows if row is not None]
    energy_estimate = float(sum(float(row["estimate"]) for row in term_payloads))
    energy_variance = float(covariance.sum())
    return {
        "method": "FR",
        "budget_total": int(total_shots),
        "num_observables": len(local_terms),
        "observable_shots": [int(value) for value in shot_allocations],
        "local_terms": term_payloads,
        "covariance_local_terms": covariance.tolist(),
        "energy_estimate": float(energy_estimate),
        "sampling_variance": float(energy_variance),
        "energy_stderr": float(math.sqrt(max(energy_variance, 0.0))),
    }


def sample_ps_qwc_hamiltonian_estimate(
    state_circuit: QuantumCircuit,
    spec: HamiltonianSpec,
    total_shots: int,
    *,
    backend: Any,
    optimization_level: int = 1,
    seed: int | None = None,
    prepared_plan: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if prepared_plan is None:
        prepared_plan = prepare_ps_qwc_measurement_plan(state_circuit, spec)
    local_terms = list(prepared_plan["local_terms"])
    groups = list(prepared_plan["groups"])
    measurements = list(prepared_plan["measurements"])
    shot_allocations = equal_partition(total_shots, len(groups))
    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    term_estimates = np.zeros(len(local_terms), dtype=float)
    group_rows: list[dict[str, Any] | None] = [None] * len(groups)
    batch_map: dict[int, list[tuple[int, dict[str, Any], QuantumCircuit]]] = {}
    for (_index, group, circuit), shots_allocated in zip(measurements, shot_allocations):
        index = int(group["index"])
        batch_map.setdefault(int(shots_allocated), []).append((index, group, circuit))

    for batch_index, (shots_allocated, batch) in enumerate(sorted(batch_map.items())):
        if shots_allocated <= 0:
            continue
        counts_list = _run_counts_batch(
            backend,
            [item[2] for item in batch],
            shots_allocated,
            optimization_level=optimization_level,
            seed=None if seed is None else int(seed) + 1000 + batch_index,
        )
        for counts, (group_index, group, _) in zip(counts_list, batch):
            shots_used = max(1, sum(int(value) for value in counts.values()))
            mean_vector, _, covariance_of_mean = sample_group_term_statistics_from_counts(
                counts,
                group["term_coeffs"],
                len(local_terms),
            )
            term_estimates += mean_vector
            covariance += covariance_of_mean
            group_rows[int(group_index)] = {
                "group_index": int(group["index"]),
                "basis_label": group["basis_label"],
                "shots_allocated": int(shots_allocated),
                "shots_used": int(shots_used),
                "num_pauli_terms": len(group["coeff_by_label"]),
                "num_local_terms": len(group["term_coeffs"]),
                "group_energy_estimate": float(mean_vector.sum()),
                "group_energy_variance": float(covariance_of_mean.sum()),
            }

    energy_estimate = float(term_estimates.sum())
    energy_variance = float(covariance.sum())
    local_term_rows = []
    for term, estimate in zip(local_terms, term_estimates):
        local_term_rows.append(
            {
                "term_index": int(term["index"]),
                "kind": term["kind"],
                "window": list(term["window"]),
                "estimate": float(estimate),
            }
        )
    return {
        "method": "PS_QWC",
        "budget_total": int(total_shots),
        "num_groups": len(groups),
        "group_shots": [int(value) for value in shot_allocations],
        "groups": [row for row in group_rows if row is not None],
        "local_terms": local_term_rows,
        "covariance_local_terms": covariance.tolist(),
        "energy_estimate": float(energy_estimate),
        "sampling_variance": float(energy_variance),
        "energy_stderr": float(math.sqrt(max(energy_variance, 0.0))),
    }


def measurement_plan_manifest(spec: HamiltonianSpec, total_shots: int) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    groups = global_qwc_group_descriptors(spec, local_terms=local_terms)
    fr_shots = equal_partition(total_shots, len(local_terms))
    ps_shots = equal_partition(total_shots, len(groups))
    return {
        "budget_total": int(total_shots),
        "num_local_terms": len(local_terms),
        "num_qwc_groups": len(groups),
        "fr": {
            "observable_shots": [int(value) for value in fr_shots],
            "observables": [
                {
                    "term_index": int(term["index"]),
                    "kind": term["kind"],
                    "window": list(term["window"]),
                    "shots_allocated": int(shots),
                }
                for term, shots in zip(local_terms, fr_shots)
            ],
        },
        "ps_qwc": {
            "group_shots": [int(value) for value in ps_shots],
            "groups": [
                {
                    "group_index": int(group["index"]),
                    "basis_label": group["basis_label"],
                    "shots_allocated": int(shots),
                    "num_pauli_terms": len(group["coeff_by_label"]),
                    "num_local_terms": len(group["term_coeffs"]),
                }
                for group, shots in zip(groups, ps_shots)
            ],
        },
    }