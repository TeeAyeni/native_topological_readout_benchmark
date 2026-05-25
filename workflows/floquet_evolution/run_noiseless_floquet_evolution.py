from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qiskit_aer import AerSimulator

THIS_DIR = Path(__file__).resolve().parent
WORKFLOWS_DIR = THIS_DIR.parent
ROOT = WORKFLOWS_DIR.parent
OUTPUTS_DIR = ROOT / "outputs"

if str(WORKFLOWS_DIR) not in sys.path:
    sys.path.insert(0, str(WORKFLOWS_DIR))

from workflow_support import (  # noqa: E402
    HamiltonianSpec,
    append_jsonl,
    build_trotter_circuit,
    circuit_resources,
    exact_hamiltonian_energy,
    portable_path_string,
    prepare_fr_measurement_plan,
    prepare_ps_qwc_measurement_plan,
    sample_fr_hamiltonian_estimate,
    sample_ps_qwc_hamiltonian_estimate,
)


def parse_steps(spec: str) -> list[int]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []
    if "-" in cleaned and cleaned.count("-") == 1 and all(chunk.strip().isdigit() for chunk in cleaned.split("-")):
        start, stop = [int(chunk.strip()) for chunk in cleaned.split("-")]
        return list(range(start, stop + 1))
    return [int(chunk.strip()) for chunk in cleaned.split(",") if chunk.strip()]


def parse_budgets(spec: str) -> list[int]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []
    return [int(chunk.strip()) for chunk in cleaned.split(",") if chunk.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a covariance-aware noiseless FR-vs-PS_QWC Floquet-evolution ladder.")
    parser.add_argument("--num-qubits", type=int, default=7)
    parser.add_argument("--steps", default="1-6")
    parser.add_argument("--budgets", default="2000,4000,8000,16000")
    parser.add_argument("--dt", type=float, default=0.2)
    parser.add_argument("--j1", type=float, default=1.0)
    parser.add_argument("--j2", type=float, default=0.5)
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=12345)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "floquet_noiseless_hamiltonian.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    steps = parse_steps(args.steps)
    budgets = parse_budgets(args.budgets)
    preview = {
        "workflow": "floquet_noiseless_hamiltonian",
        "num_qubits": int(args.num_qubits),
        "steps": steps,
        "budgets": budgets,
        "replicates": int(args.replicates),
        "output": portable_path_string(args.output),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0

    backend = AerSimulator(method="automatic")
    spec = HamiltonianSpec(num_qubits=args.num_qubits, j_f3=args.j1, j_bf4=args.j2)
    output_path = args.output.resolve()

    for replicate_index in range(1, int(args.replicates) + 1):
        for step in steps:
            state_circuit = build_trotter_circuit(args.num_qubits, step, args.dt, args.j1, args.j2)
            exact_energy = exact_hamiltonian_energy(state_circuit, spec)
            state_resources = circuit_resources(state_circuit)
            fr_plan = prepare_fr_measurement_plan(state_circuit, spec)
            ps_plan = prepare_ps_qwc_measurement_plan(state_circuit, spec)
            for budget in budgets:
                fr_seed = int(args.seed_base) + 100000 * replicate_index + 1000 * step + budget
                ps_seed = fr_seed + 500000
                fr = sample_fr_hamiltonian_estimate(
                    state_circuit,
                    spec,
                    total_shots=budget,
                    backend=backend,
                    optimization_level=args.optimization_level,
                    seed=fr_seed,
                    prepared_plan=fr_plan,
                )
                ps = sample_ps_qwc_hamiltonian_estimate(
                    state_circuit,
                    spec,
                    total_shots=budget,
                    backend=backend,
                    optimization_level=args.optimization_level,
                    seed=ps_seed,
                    prepared_plan=ps_plan,
                )
                base_row = {
                    "dataset_tag": "qwc_hamiltonian_floquet_noiseless",
                    "replicate_id": int(replicate_index),
                    "num_qubits": int(args.num_qubits),
                    "num_steps": int(step),
                    "dt": float(args.dt),
                    "j1": float(args.j1),
                    "j2": float(args.j2),
                    "budget_total": int(budget),
                    "exact_energy": float(exact_energy),
                    "state_resources": state_resources,
                    "optimization_level": int(args.optimization_level),
                }
                append_jsonl(
                    output_path,
                    {
                        **base_row,
                        **fr,
                        "squared_error": float((float(fr["energy_estimate"]) - float(exact_energy)) ** 2),
                    },
                )
                append_jsonl(
                    output_path,
                    {
                        **base_row,
                        **ps,
                        "squared_error": float((float(ps["energy_estimate"]) - float(exact_energy)) ** 2),
                    },
                )
                print(
                    json.dumps(
                        {
                            "replicate": int(replicate_index),
                            "step": int(step),
                            "budget": int(budget),
                            "fr": round(float(fr["energy_estimate"]), 8),
                            "ps_qwc": round(float(ps["energy_estimate"]), 8),
                        }
                    )
                )

    print(f"Appended noiseless Floquet-evolution rows to {portable_path_string(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())