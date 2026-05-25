from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent

COMMANDS = {
    "floquet-noiseless": ROOT / "workflows" / "floquet_evolution" / "run_noiseless_floquet_evolution.py",
    "floquet-hardware-submit": ROOT / "workflows" / "floquet_evolution" / "submit_floquet_hardware_jobs.py",
    "locked-state-vqe-noiseless": ROOT / "workflows" / "locked_state_vqe" / "run_noiseless_locked_state_vqe.py",
    "locked-state-vqe-hardware-submit": ROOT / "workflows" / "locked_state_vqe" / "submit_locked_state_vqe_hardware_jobs.py",
    "collect-ibm-job-results": ROOT / "analysis" / "collect_ibm_job_results.py",
    "summarize-dataset": ROOT / "analysis" / "summarize_benchmark_dataset.py",
    "generate-manuscript-figures": ROOT / "analysis" / "generate_manuscript_figures.py",
}

COMMAND_ALIASES = {
    "floquet-noiseless": ["digital-noiseless"],
    "floquet-hardware-submit": ["digital-hardware-submit"],
    "locked-state-vqe-noiseless": ["locked-vqe-noiseless", "vqe-noiseless"],
    "locked-state-vqe-hardware-submit": ["locked-vqe-hardware-submit", "vqe-hardware-submit"],
    "collect-ibm-job-results": ["reduce-hardware-results", "hardware-reduce"],
    "summarize-dataset": ["analyze-dataset", "analyze"],
    "generate-manuscript-figures": ["generate-figures", "plot"],
}


def display_token(token: str, index: int) -> str:
    path = Path(token)
    if path.is_absolute():
        if path.is_relative_to(ROOT):
            return path.relative_to(ROOT).as_posix()
        if index == 0:
            return path.name
    return token


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Convenience wrapper for the self-contained public benchmark workflows.")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command_name in COMMANDS:
        subparsers.add_parser(command_name, aliases=COMMAND_ALIASES.get(command_name, []), add_help=False)
    return parser.parse_known_args()


def main() -> int:
    args, passthrough = parse_args()
    command = [str(args.python.resolve()), str(COMMANDS[args.command])] + passthrough
    display_command = [display_token(token, index) for index, token in enumerate(command)]
    print(f"$ {shlex.join(display_command)}")
    completed = subprocess.run(command, cwd=ROOT)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())