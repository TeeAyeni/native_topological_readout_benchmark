from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_ibm_runtime import QiskitRuntimeService


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
WORKFLOWS_DIR = ROOT / "workflows"

if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))

from workflow_support import (  # noqa: E402
    HamiltonianSpec,
    build_trotter_circuit,
    circuit_resources,
    exact_hamiltonian_energy,
    global_qwc_group_descriptors,
    local_term_descriptors,
    locked_state_to_circuit,
    measurement_plan_manifest,
    portable_path_string,
    prepare_fr_measurement_plan,
    prepare_ps_qwc_measurement_plan,
    sample_group_term_statistics_from_counts,
)


DEFAULT_FLOQUET_TAG = "qwc_hamiltonian_hardware_floquet"
DEFAULT_LOCKED_VQE_TAG = "qwc_hamiltonian_hardware_locked_state_vqe"
MODE_ALIASES = {
    "digital": "floquet",
    "floquet": "floquet",
    "vqe": "locked-vqe",
    "locked-vqe": "locked-vqe",
    "locked-state-vqe": "locked-vqe",
}
RESOURCE_TRANSPILE_SEED = 1234


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect returned IBM job results and reduce them into canonical JSONL benchmark datasets.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--mode",
        choices=["auto", "floquet", "locked-state-vqe", "locked-vqe", "digital", "vqe"],
        default="auto",
    )
    parser.add_argument("--ibm-name", default="")
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--dataset-tag", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def infer_mode(payload: dict[str, Any], requested_mode: str) -> str:
    if requested_mode in MODE_ALIASES:
        return MODE_ALIASES[requested_mode]
    entries = payload.get("entries", [])
    if not entries:
        raise ValueError("Manifest has no entries")
    first = entries[0]
    if "num_steps" in first:
        return "floquet"
    if "locked_state_file" in first:
        return "locked-vqe"
    workflow = str(payload.get("workflow", ""))
    if "floquet" in workflow or "digital" in workflow:
        return "floquet"
    if "locked_state_vqe" in workflow or "vqe" in workflow:
        return "locked-vqe"
    raise ValueError("Could not infer manifest mode")


def default_output_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(f"{manifest_path.stem}_reduced.jsonl")


def service_from_name(ibm_name: str) -> QiskitRuntimeService:
    resolved_name = str(ibm_name).strip() or os.environ.get("IBM_NAME", "").strip()
    if resolved_name:
        return QiskitRuntimeService(name=resolved_name)
    return QiskitRuntimeService()


def counts_fetcher(service: QiskitRuntimeService, retries: int):
    cache: dict[str, list[dict[str, int]]] = {}

    def fetch(job_id: str) -> list[dict[str, int]]:
        if job_id in cache:
            return cache[job_id]
        last_exception: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                job = service.job(job_id)
                result = job.result()
                counts_list = [pub.data.meas.get_counts() for pub in result]
                cache[job_id] = counts_list
                return counts_list
            except Exception as exc:
                last_exception = exc
                if attempt < retries:
                    time.sleep(float(attempt) * 2.0)
        if last_exception is None:
            raise RuntimeError(f"Failed fetching job {job_id}")
        raise RuntimeError(f"Failed fetching job {job_id}: {last_exception}")

    fetch.cache = cache  # type: ignore[attr-defined]
    return fetch


def backend_transpile_config(backend: Any) -> tuple[list[str], list[list[int]]]:
    configuration = backend.configuration()
    basis_gates = list(dict.fromkeys(list(getattr(configuration, "basis_gates", [])) + ["measure", "reset", "delay"]))
    coupling_map = [list(edge) for edge in getattr(configuration, "coupling_map", [])]
    return basis_gates, coupling_map


def weighted_mean(values: list[float], weights: list[int]) -> float:
    total_weight = float(sum(int(weight) for weight in weights))
    if total_weight <= 0.0:
        return float(np.mean(values)) if values else 0.0
    return float(sum(float(value) * float(weight) for value, weight in zip(values, weights)) / total_weight)


def fallback_shot_allocations(total_shots: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base = int(total_shots) // int(count)
    remainder = int(total_shots) % int(count)
    return [base + (1 if index < remainder else 0) for index in range(int(count))]


def summarize_resource_distribution(
    *,
    state_depth: float,
    state_singleq: float,
    state_twoq: float,
    depth_overheads: list[float],
    singleq_overheads: list[float],
    twoq_overheads: list[float],
    shot_allocations: list[int],
) -> dict[str, float]:
    depths = [state_depth + overhead for overhead in depth_overheads]
    singleqs = [state_singleq + overhead for overhead in singleq_overheads]
    twoqs = [state_twoq + overhead for overhead in twoq_overheads]
    return {
        "state_depth": state_depth,
        "state_singleq_count": state_singleq,
        "state_twoq_count": state_twoq,
        "depth_min": float(min(depths)) if depths else 0.0,
        "depth_max": float(max(depths)) if depths else 0.0,
        "depth_mean": float(np.mean(depths)) if depths else 0.0,
        "depth_shot_weighted_mean": weighted_mean(depths, shot_allocations),
        "depth_overhead_mean": float(np.mean(depth_overheads)) if depth_overheads else 0.0,
        "depth_overhead_shot_weighted_mean": weighted_mean(depth_overheads, shot_allocations),
        "singleq_min": float(min(singleqs)) if singleqs else 0.0,
        "singleq_max": float(max(singleqs)) if singleqs else 0.0,
        "singleq_mean": float(np.mean(singleqs)) if singleqs else 0.0,
        "singleq_shot_weighted_mean": weighted_mean(singleqs, shot_allocations),
        "singleq_overhead_mean": float(np.mean(singleq_overheads)) if singleq_overheads else 0.0,
        "singleq_overhead_shot_weighted_mean": weighted_mean(singleq_overheads, shot_allocations),
        "twoq_min": float(min(twoqs)) if twoqs else 0.0,
        "twoq_max": float(max(twoqs)) if twoqs else 0.0,
        "twoq_mean": float(np.mean(twoqs)) if twoqs else 0.0,
        "twoq_shot_weighted_mean": weighted_mean(twoqs, shot_allocations),
        "twoq_overhead_mean": float(np.mean(twoq_overheads)) if twoq_overheads else 0.0,
        "twoq_overhead_shot_weighted_mean": weighted_mean(twoq_overheads, shot_allocations),
    }


def build_digital_layer_template(
    row: dict[str, Any],
    *,
    basis_gates: list[str],
    coupling_map: list[list[int]],
    template_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, Any]:
    cache_key = (
        int(row["num_qubits"]),
        float(row["j1"]),
        float(row["j2"]),
        float(row["dt"]),
        str(row["method"]),
    )
    cached = template_cache.get(cache_key)
    if cached is not None:
        return cached

    spec = HamiltonianSpec(
        num_qubits=int(row["num_qubits"]),
        j_f3=float(row["j1"]),
        j_bf4=float(row["j2"]),
    )
    state_circuit = build_trotter_circuit(
        spec.num_qubits,
        1,
        float(row["dt"]),
        float(row["j1"]),
        float(row["j2"]),
    )
    transpiled_state = transpile(
        state_circuit,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        optimization_level=1,
        seed_transpiler=RESOURCE_TRANSPILE_SEED,
    )
    state_resources = circuit_resources(transpiled_state)

    if str(row["method"]) == "FR":
        plan = prepare_fr_measurement_plan(state_circuit, spec)
        measurement_circuits = [item[3] for item in plan["measurements"]]
    else:
        plan = prepare_ps_qwc_measurement_plan(state_circuit, spec)
        measurement_circuits = [item[2] for item in plan["measurements"]]

    transpiled_measurements = transpile(
        measurement_circuits,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        optimization_level=1,
        seed_transpiler=RESOURCE_TRANSPILE_SEED,
    )
    if isinstance(transpiled_measurements, QuantumCircuit):
        transpiled_measurements = [transpiled_measurements]
    measurement_resources = [circuit_resources(circuit) for circuit in transpiled_measurements]

    cached = {
        "num_measurement_circuits": float(len(measurement_resources)),
        "state_depth_per_trotter_layer": float(state_resources["depth"]),
        "state_singleq_per_trotter_layer": float(state_resources["singleq_count_rough"]),
        "state_twoq_per_trotter_layer": float(state_resources["twoq_count_rough"]),
        "depth_overheads": [float(resource["depth"]) - float(state_resources["depth"]) for resource in measurement_resources],
        "singleq_overheads": [
            float(resource["singleq_count_rough"]) - float(state_resources["singleq_count_rough"])
            for resource in measurement_resources
        ],
        "twoq_overheads": [
            float(resource["twoq_count_rough"]) - float(state_resources["twoq_count_rough"])
            for resource in measurement_resources
        ],
    }
    template_cache[cache_key] = cached
    return cached


def estimate_digital_row_resource_summary(
    row: dict[str, Any],
    *,
    basis_gates: list[str],
    coupling_map: list[list[int]],
    template_cache: dict[tuple[Any, ...], dict[str, Any]],
) -> dict[str, float]:
    template = build_digital_layer_template(
        row,
        basis_gates=basis_gates,
        coupling_map=coupling_map,
        template_cache=template_cache,
    )
    if str(row["method"]) == "FR":
        shot_allocations = [int(value) for value in row.get("observable_shots", [])]
        if not shot_allocations:
            shot_allocations = fallback_shot_allocations(int(row["budget_total"]), int(row.get("num_observables", 0)))
    else:
        shot_allocations = [int(value) for value in row.get("group_shots", [])]
        if not shot_allocations:
            shot_allocations = fallback_shot_allocations(int(row["budget_total"]), int(row.get("num_groups", 0)))

    num_steps = float(int(row["num_steps"]))
    state_depth = num_steps * float(template["state_depth_per_trotter_layer"])
    state_singleq = num_steps * float(template["state_singleq_per_trotter_layer"])
    state_twoq = num_steps * float(template["state_twoq_per_trotter_layer"])

    summary = {
        "num_trotter_steps": num_steps,
        "num_measurement_circuits": float(template["num_measurement_circuits"]),
        "state_depth_per_trotter_layer": float(template["state_depth_per_trotter_layer"]),
        "state_singleq_per_trotter_layer": float(template["state_singleq_per_trotter_layer"]),
        "state_twoq_per_trotter_layer": float(template["state_twoq_per_trotter_layer"]),
    }
    summary.update(
        summarize_resource_distribution(
            state_depth=state_depth,
            state_singleq=state_singleq,
            state_twoq=state_twoq,
            depth_overheads=list(template["depth_overheads"]),
            singleq_overheads=list(template["singleq_overheads"]),
            twoq_overheads=list(template["twoq_overheads"]),
            shot_allocations=shot_allocations,
        )
    )
    return summary


def normalize_job_batches(method_section: dict[str, Any], shot_allocations: list[int]) -> list[dict[str, Any]]:
    if isinstance(method_section.get("job_batches"), list):
        return [dict(batch) for batch in method_section["job_batches"]]

    job_ids = [str(job_id) for job_id in method_section.get("job_ids", []) if str(job_id).strip()]
    if not job_ids:
        raise ValueError("Manifest method section has neither job_batches nor job_ids")

    unique_shots = sorted(set(int(value) for value in shot_allocations))
    if len(unique_shots) != len(job_ids):
        raise ValueError("Could not align job_ids with unique shot bins")

    return [
        {
            "job_id": job_id,
            "n_circuits": None,
            "shots": int(shots),
            "submitted_only": True,
        }
        for job_id, shots in zip(job_ids, unique_shots)
    ]


def entry_measurement_plan(spec: HamiltonianSpec, entry: dict[str, Any], budget_total: int) -> dict[str, Any]:
    for key in ("fr", "ps_qwc"):
        section = entry.get(key, {})
        if isinstance(section, dict) and isinstance(section.get("measurement_plan"), dict):
            return dict(section["measurement_plan"])
    if isinstance(entry.get("measurement_plan"), dict):
        return dict(entry["measurement_plan"])
    return measurement_plan_manifest(spec, int(budget_total))


def reduce_fr(spec: HamiltonianSpec, plan: dict[str, Any], job_batches: list[dict[str, Any]], fetch_counts) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    coeffs = [float(spec.j_f3 if term["kind"] == "F3" else spec.j_bf4) for term in local_terms]
    z_qubits = [int(term["window"][2] if term["kind"] == "BF4" else term["window"][1]) for term in local_terms]

    allocations = [int(value) for value in plan["fr"]["observable_shots"]]
    indices_by_shots: dict[int, list[int]] = defaultdict(list)
    for term_index, shots in enumerate(allocations):
        indices_by_shots[int(shots)].append(int(term_index))

    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    rows: list[dict[str, Any] | None] = [None] * len(local_terms)

    sorted_batches = sorted(job_batches, key=lambda batch: int(batch["shots"]))
    sorted_groups = sorted(indices_by_shots.items())
    for batch, (_, term_indices) in zip(sorted_batches, sorted_groups):
        counts_list = fetch_counts(str(batch["job_id"]))
        for counts, term_index in zip(counts_list, term_indices):
            shots_used = max(1, sum(int(value) for value in counts.values()))
            z_qubit = int(z_qubits[term_index])
            coeff = float(coeffs[term_index])
            z_mean = 0.0
            for bitstring, count in counts.items():
                sign = 1.0 if bitstring[::-1][z_qubit] == "0" else -1.0
                z_mean += sign * (float(count) / float(shots_used))
            estimate = coeff * z_mean
            if shots_used > 1:
                sample_var = max(0.0, coeff * coeff - estimate * estimate) * (float(shots_used) / float(shots_used - 1))
                variance_of_mean = sample_var / float(shots_used)
            else:
                variance_of_mean = 0.0
            covariance[term_index, term_index] = float(variance_of_mean)
            term = local_terms[term_index]
            rows[term_index] = {
                "term_index": int(term["index"]),
                "kind": term["kind"],
                "window": list(term["window"]),
                "shots_allocated": int(batch["shots"]),
                "shots_used": int(shots_used),
                "estimate": float(estimate),
                "variance_of_mean": float(variance_of_mean),
                "stderr": float(math.sqrt(max(variance_of_mean, 0.0))),
            }

    term_rows = [row for row in rows if row is not None]
    energy = float(sum(float(row["estimate"]) for row in term_rows))
    variance = float(covariance.sum())
    return {
        "method": "FR",
        "budget_total": int(plan["budget_total"]),
        "num_observables": len(local_terms),
        "observable_shots": allocations,
        "local_terms": term_rows,
        "covariance_local_terms": covariance.tolist(),
        "energy_estimate": energy,
        "sampling_variance": variance,
        "energy_stderr": float(math.sqrt(max(variance, 0.0))),
    }


def reduce_ps(spec: HamiltonianSpec, plan: dict[str, Any], job_batches: list[dict[str, Any]], fetch_counts) -> dict[str, Any]:
    local_terms = local_term_descriptors(spec)
    groups = global_qwc_group_descriptors(spec, local_terms=local_terms)

    allocations = [int(value) for value in plan["ps_qwc"]["group_shots"]]
    indices_by_shots: dict[int, list[int]] = defaultdict(list)
    for group_index, shots in enumerate(allocations):
        indices_by_shots[int(shots)].append(int(group_index))

    covariance = np.zeros((len(local_terms), len(local_terms)), dtype=float)
    term_estimates = np.zeros(len(local_terms), dtype=float)
    group_rows: list[dict[str, Any] | None] = [None] * len(groups)

    sorted_batches = sorted(job_batches, key=lambda batch: int(batch["shots"]))
    sorted_groups = sorted(indices_by_shots.items())
    for batch, (_, group_indices) in zip(sorted_batches, sorted_groups):
        counts_list = fetch_counts(str(batch["job_id"]))
        for counts, group_index in zip(counts_list, group_indices):
            group = groups[group_index]
            shots_used = max(1, sum(int(value) for value in counts.values()))
            mean_vector, _, covariance_of_mean = sample_group_term_statistics_from_counts(
                counts,
                group["term_coeffs"],
                len(local_terms),
            )
            term_estimates += mean_vector
            covariance += covariance_of_mean
            group_rows[group_index] = {
                "group_index": int(group["index"]),
                "basis_label": group["basis_label"],
                "shots_allocated": int(batch["shots"]),
                "shots_used": int(shots_used),
                "num_pauli_terms": len(group["coeff_by_label"]),
                "num_local_terms": len(group["term_coeffs"]),
                "group_energy_estimate": float(mean_vector.sum()),
                "group_energy_variance": float(covariance_of_mean.sum()),
            }

    local_rows = []
    for term, estimate in zip(local_terms, term_estimates):
        local_rows.append(
            {
                "term_index": int(term["index"]),
                "kind": term["kind"],
                "window": list(term["window"]),
                "estimate": float(estimate),
            }
        )

    energy = float(term_estimates.sum())
    variance = float(covariance.sum())
    return {
        "method": "PS_QWC",
        "budget_total": int(plan["budget_total"]),
        "num_groups": len(groups),
        "group_shots": allocations,
        "groups": [row for row in group_rows if row is not None],
        "local_terms": local_rows,
        "covariance_local_terms": covariance.tolist(),
        "energy_estimate": energy,
        "sampling_variance": variance,
        "energy_stderr": float(math.sqrt(max(variance, 0.0))),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def resolve_locked_state_file(raw_path: str) -> Path:
    candidate = Path(raw_path).expanduser()
    if candidate.exists():
        return candidate.resolve()

    local_name = candidate.name
    search_dirs = [ROOT / "dependencies" / "VQE_FR_PS" / "locked_state_payloads"]

    for search_dir in search_dirs:
        resolved = search_dir / local_name
        if resolved.exists():
            return resolved.resolve()

    raise FileNotFoundError(f"Could not resolve locked-state payload: {raw_path}")


def reduce_digital_manifest(
    payload: dict[str, Any],
    *,
    basis_gates: list[str] | None,
    coupling_map: list[list[int]] | None,
    fetch_counts,
    dataset_tag: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    template_cache: dict[tuple[Any, ...], dict[str, Any]] = {}
    manifest_num_qubits = int(payload.get("num_qubits", 7))
    manifest_dt = float(payload.get("dt", 0.2))
    manifest_j1 = float(payload.get("j1", 1.0))
    manifest_j2 = float(payload.get("j2", 0.5))

    for entry in payload.get("entries", []):
        num_qubits = int(entry.get("num_qubits", manifest_num_qubits))
        num_steps = int(entry["num_steps"])
        budget_total = int(entry["budget_total"])
        dt = float(entry.get("dt", manifest_dt))
        j1 = float(entry.get("j1", manifest_j1))
        j2 = float(entry.get("j2", manifest_j2))

        spec = HamiltonianSpec(num_qubits=num_qubits, j_f3=j1, j_bf4=j2)
        state = build_trotter_circuit(num_qubits, num_steps, dt, j1, j2)
        exact_energy = float(exact_hamiltonian_energy(state, spec))
        plan = entry_measurement_plan(spec, entry, budget_total)

        fr = reduce_fr(
            spec,
            plan,
            normalize_job_batches(dict(entry["fr"]), list(plan["fr"]["observable_shots"])),
            fetch_counts,
        )
        ps = reduce_ps(
            spec,
            plan,
            normalize_job_batches(dict(entry["ps_qwc"]), list(plan["ps_qwc"]["group_shots"])),
            fetch_counts,
        )
        common = {
            "dataset_tag": dataset_tag,
            "replicate_id": int(entry.get("replicate_id", 1)),
            "num_qubits": int(num_qubits),
            "num_steps": int(num_steps),
            "dt": float(dt),
            "j1": float(j1),
            "j2": float(j2),
            "budget_total": int(budget_total),
            "exact_energy": float(exact_energy),
        }
        fr_row = {**common, **fr, "squared_error": float((fr["energy_estimate"] - exact_energy) ** 2)}
        ps_row = {**common, **ps, "squared_error": float((ps["energy_estimate"] - exact_energy) ** 2)}
        if basis_gates is not None and coupling_map is not None:
            fr_row["resource_summary"] = estimate_digital_row_resource_summary(
                fr_row,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                template_cache=template_cache,
            )
            ps_row["resource_summary"] = estimate_digital_row_resource_summary(
                ps_row,
                basis_gates=basis_gates,
                coupling_map=coupling_map,
                template_cache=template_cache,
            )
        rows.append(fr_row)
        rows.append(ps_row)
    return rows


def reduce_vqe_manifest(
    payload: dict[str, Any],
    *,
    fetch_counts,
    dataset_tag: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for entry in payload.get("entries", []):
        budget_total = int(entry["budget_total"])
        locked_state_file = resolve_locked_state_file(str(entry["locked_state_file"]))
        locked_payload, spec, state = locked_state_to_circuit(locked_state_file)
        exact_energy = float(exact_hamiltonian_energy(state, spec))
        plan = entry_measurement_plan(spec, entry, budget_total)

        fr = reduce_fr(
            spec,
            plan,
            normalize_job_batches(dict(entry["fr"]), list(plan["fr"]["observable_shots"])),
            fetch_counts,
        )
        ps = reduce_ps(
            spec,
            plan,
            normalize_job_batches(dict(entry["ps_qwc"]), list(plan["ps_qwc"]["group_shots"])),
            fetch_counts,
        )
        common = {
            "dataset_tag": dataset_tag,
            "replicate_id": int(entry.get("replicate_id", 1)),
            "num_qubits": int(entry["num_qubits"]),
            "budget_total": int(budget_total),
            "locked_state_file": portable_path_string(locked_state_file),
            "ansatz": str(entry.get("ansatz", locked_payload["spec"]["ansatz"])),
            "depth": int(entry.get("depth", locked_payload["spec"]["depth"])),
            "reference": str(entry.get("reference", locked_payload["spec"].get("reference", "fusion_valid"))),
            "j_f3": float(spec.j_f3),
            "j_bf4": float(spec.j_bf4),
            "exact_locked_energy": float(locked_payload["exact_vqe"]["result"]["fun"]),
            "exact_ground_energy": float(locked_payload["exact_ground_energy"]),
            "exact_energy": float(exact_energy),
            "exact_hamiltonian_energy": float(exact_energy),
        }
        rows.append({**common, **fr, "squared_error": float((fr["energy_estimate"] - exact_energy) ** 2)})
        rows.append({**common, **ps, "squared_error": float((ps["energy_estimate"] - exact_energy) ** 2)})
    return rows


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    mode = infer_mode(payload, args.mode)
    output_path = args.output.resolve() if args.output is not None else default_output_path(manifest_path).resolve()
    dataset_tag = str(args.dataset_tag).strip() or (DEFAULT_FLOQUET_TAG if mode == "floquet" else DEFAULT_LOCKED_VQE_TAG)

    preview = {
        "manifest": str(manifest_path),
        "mode": mode,
        "entries": len(payload.get("entries", [])),
        "output": str(output_path),
        "dataset_tag": dataset_tag,
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0

    service = service_from_name(args.ibm_name)
    fetch_counts = counts_fetcher(service, int(args.retries))

    if mode == "floquet":
        basis_gates = None
        coupling_map = None
        backend_name = str(payload.get("backend", "")).strip()
        if backend_name:
            backend = service.backend(backend_name)
            basis_gates, coupling_map = backend_transpile_config(backend)
        rows = reduce_digital_manifest(
            payload,
            basis_gates=basis_gates,
            coupling_map=coupling_map,
            fetch_counts=fetch_counts,
            dataset_tag=dataset_tag,
        )
    else:
        rows = reduce_vqe_manifest(payload, fetch_counts=fetch_counts, dataset_tag=dataset_tag)

    write_jsonl(output_path, rows)
    summary = {
        **preview,
        "rows": len(rows),
        "jobs_cached": len(fetch_counts.cache),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())