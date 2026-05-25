from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TEMP_ROOT = ROOT / "outputs" / "_publish_preflight"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a minimal preflight check before publishing the self-contained github_public package.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--keep-temp", action="store_true", help="Keep temporary outputs for inspection.")
    return parser.parse_args()


def relative_to_root(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def run_step(python_path: Path, label: str, *args: str) -> None:
    command = [str(python_path.resolve()), "run_workflow.py", *args]
    print(f"[{label}] {' '.join(args)}")
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    args = parse_args()

    if TEMP_ROOT.exists():
        shutil.rmtree(TEMP_ROOT)
    TEMP_ROOT.mkdir(parents=True, exist_ok=True)

    floquet_jsonl = TEMP_ROOT / "floquet_noiseless_hamiltonian.jsonl"
    locked_vqe_jsonl = TEMP_ROOT / "locked_state_vqe_noiseless_hamiltonian.jsonl"
    floquet_summary = TEMP_ROOT / "floquet_noiseless_hamiltonian_summary.json"
    locked_vqe_summary = TEMP_ROOT / "locked_state_vqe_noiseless_hamiltonian_summary.json"
    example_figures = TEMP_ROOT / "example_figures"

    try:
        run_step(
            args.python,
            "floquet-noiseless",
            "floquet-noiseless",
            "--num-qubits",
            "5",
            "--steps",
            "1",
            "--budgets",
            "100",
            "--replicates",
            "1",
            "--output",
            relative_to_root(floquet_jsonl),
        )
        run_step(
            args.python,
            "locked-state-vqe-noiseless",
            "locked-state-vqe-noiseless",
            "--sizes",
            "5",
            "--budgets",
            "100",
            "--replicates",
            "1",
            "--output",
            relative_to_root(locked_vqe_jsonl),
        )
        run_step(
            args.python,
            "floquet-hardware-submit",
            "floquet-hardware-submit",
            "--backend",
            "ibm_pittsburgh",
            "--num-qubits",
            "5",
            "--steps",
            "1",
            "--budgets",
            "100",
            "--output",
            relative_to_root(TEMP_ROOT / "floquet_hardware_manifest.json"),
            "--dry-run",
        )
        run_step(
            args.python,
            "locked-state-vqe-hardware-submit",
            "locked-state-vqe-hardware-submit",
            "--backend",
            "ibm_pittsburgh",
            "--sizes",
            "5",
            "--budgets",
            "100",
            "--output",
            relative_to_root(TEMP_ROOT / "locked_state_vqe_hardware_manifest.json"),
            "--dry-run",
        )
        run_step(
            args.python,
            "summarize-floquet",
            "summarize-dataset",
            "--dataset",
            relative_to_root(floquet_jsonl),
            "--group-fields",
            "dataset_tag,num_qubits,num_steps,budget_total",
            "--out-json",
            relative_to_root(floquet_summary),
        )
        run_step(
            args.python,
            "summarize-locked-state-vqe",
            "summarize-dataset",
            "--dataset",
            relative_to_root(locked_vqe_jsonl),
            "--group-fields",
            "dataset_tag,num_qubits,budget_total",
            "--out-json",
            relative_to_root(locked_vqe_summary),
        )
        run_step(
            args.python,
            "plot-example-figures",
            "generate-manuscript-figures",
            "--date-tag",
            "preflight",
            "--floquet-noiseless-summary",
            "examples/summaries/floquet_noiseless_hamiltonian_summary.json",
            "--floquet-hardware-summary",
            "examples/summaries/floquet_hardware_with_snapshot_fill_20260515_summary.json",
            "--locked-state-vqe-noiseless-summary",
            "examples/summaries/locked_state_vqe_noiseless_hamiltonian_summary.json",
            "--locked-state-vqe-hardware-summary",
            "examples/summaries/locked_state_vqe_hardware_with_snapshot_fill_20260515_summary.json",
            "--output-dir",
            relative_to_root(example_figures),
        )
        if args.keep_temp:
            print(f"Publish preflight passed. Temporary outputs kept in {relative_to_root(TEMP_ROOT)}")
        else:
            print("Publish preflight passed.")
        return 0
    finally:
        if not args.keep_temp and TEMP_ROOT.exists():
            shutil.rmtree(TEMP_ROOT)


if __name__ == "__main__":
    raise SystemExit(main())