from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent


def portable_path_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def parse_fields(spec: str) -> list[str]:
    cleaned = str(spec).strip()
    if not cleaned:
        return []
    return [chunk.strip() for chunk in cleaned.split(",") if chunk.strip()]


def key_for_row(row: dict[str, Any], fields: list[str]) -> tuple[Any, ...]:
    return tuple(row.get(field) for field in fields)


def summarize_method(rows: list[dict[str, Any]]) -> dict[str, Any]:
    estimates = [float(row["energy_estimate"]) for row in rows]
    exact_values = [float(row["exact_energy"]) if "exact_energy" in row else float(row["exact_hamiltonian_energy"]) for row in rows]
    if max(exact_values) - min(exact_values) > 1e-9:
        raise ValueError("Grouped rows do not share the same exact energy")
    exact_energy = float(mean(exact_values))
    squared_errors = [float(row.get("squared_error", (float(row["energy_estimate"]) - exact_energy) ** 2)) for row in rows]
    sampling_variances = [float(row.get("sampling_variance", 0.0)) for row in rows]
    mean_estimate = float(mean(estimates))
    bias_empirical = float(mean_estimate - exact_energy)
    mse_empirical = float(mean(squared_errors))
    var_empirical = float(pstdev(estimates) ** 2) if len(estimates) >= 2 else 0.0
    mse_empirical_from_decomposition = float(bias_empirical * bias_empirical + var_empirical)
    model_based_mse = float(bias_empirical * bias_empirical + mean(sampling_variances))
    summary = {
        "n": len(rows),
        "exact_energy": exact_energy,
        "estimate_mean": mean_estimate,
        "estimate_std": float(pstdev(estimates)) if len(estimates) >= 2 else 0.0,
        "bias_empirical": bias_empirical,
        "var_empirical": var_empirical,
        "mse_empirical": mse_empirical,
        "mse_empirical_from_decomposition": mse_empirical_from_decomposition,
        "mse_decomposition_gap": float(mse_empirical - mse_empirical_from_decomposition),
        "mean_sampling_variance": float(mean(sampling_variances)) if sampling_variances else 0.0,
        "mean_sampling_stderr": float(mean(v ** 0.5 for v in sampling_variances)) if sampling_variances else 0.0,
        "mse_model_bias_plus_mean_sampling_var": model_based_mse,
    }
    resource_summaries = [row.get("resource_summary") for row in rows if isinstance(row.get("resource_summary"), dict)]
    if resource_summaries:
        numeric_keys = sorted(
            {
                key
                for resource_summary in resource_summaries
                for key, value in resource_summary.items()
                if isinstance(value, (int, float))
            }
        )
        if numeric_keys:
            summary["resource_summary"] = {
                key: float(mean(float(resource_summary[key]) for resource_summary in resource_summaries if key in resource_summary))
                for key in numeric_keys
            }
    return summary

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark JSONL datasets into comparison-cell statistics and covariance-aware error metrics.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--group-fields",
        default="dataset_tag,num_qubits,num_steps,budget_total",
        help="Comma-separated fields used to define a comparison cell.",
    )
    parser.add_argument("--out-json", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset_path = args.dataset.resolve()
    rows = load_jsonl(dataset_path)
    group_fields = parse_fields(args.group_fields)
    grouped: dict[tuple[Any, ...], dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[key_for_row(row, group_fields)][str(row["method"])].append(row)

    payload: dict[str, Any] = {
        "dataset": portable_path_string(dataset_path),
        "group_fields": group_fields,
        "groups": [],
    }
    for key in sorted(grouped.keys(), key=lambda item: tuple("" if value is None else str(value) for value in item)):
        methods = grouped[key]
        cell = {
            "group": {field: value for field, value in zip(group_fields, key)},
            "methods": {},
        }
        for method_name, method_rows in sorted(methods.items()):
            cell["methods"][method_name] = summarize_method(method_rows)
        if "FR" in cell["methods"] and "PS_QWC" in cell["methods"]:
            fr = cell["methods"]["FR"]
            ps = cell["methods"]["PS_QWC"]
            cell["comparison"] = {
                "winner_by_empirical_mse": "FR" if float(fr["mse_empirical"]) < float(ps["mse_empirical"]) else "PS_QWC",
                "winner_by_model_based_mse": "FR" if float(fr["mse_model_bias_plus_mean_sampling_var"]) < float(ps["mse_model_bias_plus_mean_sampling_var"]) else "PS_QWC",
                "delta_empirical_mse": float(fr["mse_empirical"]) - float(ps["mse_empirical"]),
                "delta_model_based_mse": float(fr["mse_model_bias_plus_mean_sampling_var"]) - float(ps["mse_model_bias_plus_mean_sampling_var"]),
            }
        payload["groups"].append(cell)

    if args.out_json is not None:
        output_path = args.out_json.resolve()
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote summary to {portable_path_string(output_path)}")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())