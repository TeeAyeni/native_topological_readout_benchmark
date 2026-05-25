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
    append_jsonl,
    circuit_resources,
    default_locked_state_map,
    exact_hamiltonian_energy,
    locked_state_to_circuit,
    portable_path_string,
    sample_fr_hamiltonian_estimate,
    sample_ps_qwc_hamiltonian_estimate,
)


def parse_int_list(spec: str) -> list[int]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []
    return [int(chunk.strip()) for chunk in cleaned.split(",") if chunk.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a noiseless FR-vs-PS_QWC locked-state VQE ladder.")
    parser.add_argument("--sizes", default="5,7,9,11")
    parser.add_argument("--budgets", default="2000,4000,8000,16000")
    parser.add_argument("--replicates", type=int, default=5)
    parser.add_argument("--seed-base", type=int, default=24680)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUTS_DIR / "locked_state_vqe_noiseless_hamiltonian.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sizes = parse_int_list(args.sizes)
    budgets = parse_int_list(args.budgets)
    locked_map = default_locked_state_map()
    preview = {
        "workflow": "locked_state_vqe_noiseless_hamiltonian",
        "sizes": sizes,
        "budgets": budgets,
        "replicates": int(args.replicates),
        "locked_states": {size: portable_path_string(locked_map[size]) for size in sizes},
        "output": portable_path_string(args.output),
    }
    if args.dry_run:
        print(json.dumps(preview, indent=2))
        return 0

    backend = AerSimulator(method="automatic")
    output_path = args.output.resolve()
    for size in sizes:
        locked_path = locked_map[size]
        payload, spec, circuit = locked_state_to_circuit(locked_path)
        exact_energy = exact_hamiltonian_energy(circuit, spec)
        for replicate_index in range(1, int(args.replicates) + 1):
            for budget in budgets:
                fr_seed = int(args.seed_base) + 100000 * size + 1000 * replicate_index + budget
                ps_seed = fr_seed + 700000
                fr = sample_fr_hamiltonian_estimate(
                    circuit,
                    spec,
                    total_shots=budget,
                    backend=backend,
                    optimization_level=args.optimization_level,
                    seed=fr_seed,
                )
                ps = sample_ps_qwc_hamiltonian_estimate(
                    circuit,
                    spec,
                    total_shots=budget,
                    backend=backend,
                    optimization_level=args.optimization_level,
                    seed=ps_seed,
                )
                base_row = {
                    "dataset_tag": "qwc_hamiltonian_locked_state_vqe_noiseless",
                    "replicate_id": int(replicate_index),
                    "num_qubits": int(size),
                    "budget_total": int(budget),
                    "locked_state_file": portable_path_string(locked_path),
                    "exact_locked_energy": float(payload["exact_vqe"]["result"]["fun"]),
                    "exact_ground_energy": float(payload["exact_ground_energy"]),
                    "exact_energy": float(exact_energy),
                    "exact_hamiltonian_energy": float(exact_energy),
                    "state_resources": circuit_resources(circuit),
                    "optimization_level": int(args.optimization_level),
                    "ansatz": str(payload["spec"]["ansatz"]),
                    "depth": int(payload["spec"]["depth"]),
                    "reference": str(payload["spec"].get("reference", "fusion_valid")),
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
                            "size": int(size),
                            "replicate": int(replicate_index),
                            "budget": int(budget),
                            "fr": round(float(fr["energy_estimate"]), 8),
                            "ps_qwc": round(float(ps["energy_estimate"]), 8),
                        }
                    )
                )

    print(f"Appended locked-state VQE noiseless rows to {portable_path_string(output_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())