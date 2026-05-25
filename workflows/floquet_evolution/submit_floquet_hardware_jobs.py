from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from qiskit import transpile
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from qiskit_ibm_runtime import SamplerV2 as Sampler
except Exception:
    from qiskit_ibm_runtime import Sampler


THIS_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = THIS_DIR.parent
ROOT = WORKFLOWS_DIR.parent
OUTPUTS_DIR = ROOT / "outputs"

if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))

from workflow_support import (  # noqa: E402
    HamiltonianSpec,
    add_measurement_basis_for_pauli,
    build_trotter_circuit,
    fr_measurement_circuit,
    global_qwc_group_descriptors,
    local_term_descriptors,
    measurement_plan_manifest,
    portable_path_string,
    write_json,
)


def parse_int_list(spec: str) -> list[int]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []
    return [int(chunk.strip()) for chunk in cleaned.split(",") if chunk.strip()]


def service_from_env() -> QiskitRuntimeService:
    kwargs: dict[str, Any] = {}
    if os.environ.get("IBM_NAME"):
        kwargs["name"] = os.environ["IBM_NAME"]
    if os.environ.get("IBM_CHANNEL"):
        kwargs["channel"] = os.environ["IBM_CHANNEL"]
    if os.environ.get("IBM_INSTANCE"):
        kwargs["instance"] = os.environ["IBM_INSTANCE"]
    return QiskitRuntimeService(**kwargs)


def submit_batch(sampler: Any, backend: Any, circuits: list[Any], shots: int, optimization_level: int) -> dict[str, Any]:
    if not circuits:
        return {"job_id": None, "n_circuits": 0, "shots": int(shots), "submitted_only": True}
    transpiled_circuits = transpile(circuits, backend=backend, optimization_level=optimization_level)
    job = sampler.run(transpiled_circuits, shots=shots)
    return {
        "job_id": str(job.job_id()) if callable(getattr(job, "job_id", None)) else str(getattr(job, "job_id", None)),
        "n_circuits": len(transpiled_circuits),
        "shots": int(shots),
        "submitted_only": True,
    }


def submit_fr(state_circuit: Any, spec: Any, total_shots: int, sampler: Any, backend: Any, optimization_level: int) -> dict[str, Any]:
    plan = measurement_plan_manifest(spec, total_shots)
    local_terms = local_term_descriptors(spec)
    batches: dict[int, list[Any]] = {}
    for term, shots in zip(local_terms, plan["fr"]["observable_shots"]):
        circuit, _ = fr_measurement_circuit(state_circuit, term["kind"], term["window"])
        circuit.measure_all()
        batches.setdefault(int(shots), []).append(circuit)
    job_batches = []
    for shots, circuits in sorted(batches.items()):
        job_batches.append(submit_batch(sampler, backend, circuits, shots, optimization_level))
    return {
        "method": "FR",
        "submitted_only": True,
        "budget_total": int(total_shots),
        "measurement_plan": plan,
        "job_batches": job_batches,
    }


def submit_ps_qwc(state_circuit: Any, spec: Any, total_shots: int, sampler: Any, backend: Any, optimization_level: int) -> dict[str, Any]:
    plan = measurement_plan_manifest(spec, total_shots)
    local_terms = local_term_descriptors(spec)
    groups = global_qwc_group_descriptors(spec, local_terms)
    batches: dict[int, list[Any]] = {}
    for group, shots in zip(groups, plan["ps_qwc"]["group_shots"]):
        circuit = state_circuit.copy()
        add_measurement_basis_for_pauli(circuit, group["basis_label"])
        circuit.measure_all()
        batches.setdefault(int(shots), []).append(circuit)
    job_batches = []
    for shots, circuits in sorted(batches.items()):
        job_batches.append(submit_batch(sampler, backend, circuits, shots, optimization_level))
    return {
        "method": "PS_QWC",
        "submitted_only": True,
        "budget_total": int(total_shots),
        "measurement_plan": plan,
        "job_batches": job_batches,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Submit Floquet-evolution FR-vs-PS_QWC ladder jobs to IBM hardware.")
    parser.add_argument("--backend", default="ibm_pittsburgh")
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--steps", default="1,2,3,4,5,6")
    parser.add_argument("--budgets", default="2000,4000,8000,16000")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / f"floquet_hardware_submit_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = parse_int_list(args.steps)
    budgets = parse_int_list(args.budgets)
    preview = {
        "backend": args.backend,
        "num_qubits": int(args.num_qubits),
        "steps": steps,
        "budgets": budgets,
        "dt": float(args.dt),
        "j1": float(args.j1),
        "j2": float(args.j2),
        "ibm_name": os.environ.get("IBM_NAME"),
        "output": portable_path_string(args.output),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0

    service = service_from_env()
    backend = service.backend(args.backend)
    sampler = Sampler(mode=backend)
    spec = HamiltonianSpec(num_qubits=int(args.num_qubits), j_f3=float(args.j1), j_bf4=float(args.j2))

    payload: dict[str, Any] = {
        "workflow": "floquet_hardware_submit",
        "backend": args.backend,
        "num_qubits": int(args.num_qubits),
        "dt": float(args.dt),
        "j1": float(args.j1),
        "j2": float(args.j2),
        "ibm_name": os.environ.get("IBM_NAME"),
        "entries": [],
    }

    for step in steps:
        state_circuit = build_trotter_circuit(int(args.num_qubits), int(step), float(args.dt), float(args.j1), float(args.j2))
        for budget in budgets:
            fr = submit_fr(state_circuit, spec, int(budget), sampler, backend, int(args.optimization_level))
            ps = submit_ps_qwc(state_circuit, spec, int(budget), sampler, backend, int(args.optimization_level))
            entry = {
                "num_steps": int(step),
                "budget_total": int(budget),
                "fr": fr,
                "ps_qwc": ps,
            }
            payload["entries"].append(entry)
            print(
                json.dumps(
                    {
                        "step": int(step),
                        "budget": int(budget),
                        "fr_job_ids": [batch["job_id"] for batch in fr["job_batches"]],
                        "ps_job_ids": [batch["job_id"] for batch in ps["job_batches"]],
                    }
                )
            )

    output_path = args.output.resolve()
    write_json(output_path, payload)
    print(f"Wrote Floquet hardware submit manifest to {portable_path_string(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())