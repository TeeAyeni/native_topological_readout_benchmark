from __future__ import annotations

import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
OUT_DIR = PROJECT_ROOT / "outputs"
DEFAULT_DATE_TAG = "20260515"
FIGURE_FILE_SUFFIXES = (".png", ".pdf", ".svg")

METHODS = ("FR", "PS_QWC")
EPSILON = 1e-18
FR_COLOR = "#2f5aa8"
PS_COLOR = "#c4681e"
RATIO_CMAP = "coolwarm"
RESOURCE_STEP_MARKERS = ("o", "s", "^", "D", "P", "X", "v", "<", ">")

METRIC_SPECS = {
    "empirical": {
        "field": "mse_empirical",
        "title": "Empirical MSE",
        "filename_stub": "empirical_mse_ratio",
    },
    "model": {
        "field": "mse_model_bias_plus_mean_sampling_var",
        "title": "Model-based MSE",
        "filename_stub": "model_mse_ratio",
    },
    "sampling_variance": {
        "field": "mean_sampling_variance",
        "title": "Mean sampling variance",
        "filename_stub": "sampling_variance_ratio",
    },
}

DATASET_LABELS = {
    "digital_noiseless": "Floquet noiseless",
    "digital_hardware_fill": "Floquet hardware",
    "vqe_noiseless": "Locked-state VQE noiseless",
    "vqe_hardware_fill": "Locked-state VQE hardware",
}


def portable_path_string(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return path.name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate manuscript-ready figures and summary reports from benchmark summary JSON files."
    )
    parser.add_argument("--date-tag", default=DEFAULT_DATE_TAG)
    parser.add_argument(
        "--floquet-noiseless-summary",
        "--digital-noiseless-summary",
        dest="digital_noiseless_summary",
        type=Path,
        default=OUT_DIR / "floquet_noiseless_hamiltonian_summary.json",
    )
    parser.add_argument(
        "--floquet-hardware-summary",
        "--digital-hardware-summary",
        dest="digital_hardware_summary",
        type=Path,
        default=OUT_DIR / f"floquet_hardware{DEFAULT_DATE_TAG}_summary.json",
    )
    parser.add_argument(
        "--locked-state-vqe-noiseless-summary",
        "--locked-vqe-noiseless-summary",
        "--vqe-noiseless-summary",
        dest="vqe_noiseless_summary",
        type=Path,
        default=OUT_DIR / "locked_state_vqe_noiseless_hamiltonian_summary.json",
    )
    parser.add_argument(
        "--locked-state-vqe-hardware-summary",
        "--locked-vqe-hardware-summary",
        "--vqe-hardware-summary",
        dest="vqe_hardware_summary",
        type=Path,
        default=OUT_DIR / f"locked_state_vqe_hardware_{DEFAULT_DATE_TAG}_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for figures and report. Defaults to outputs/manuscript_figures_<date-tag>.",
    )
    parser.add_argument(
        "--copy-dir",
        type=Path,
        default=None,
        help="Optional safe copy target for generated assets. Existing manuscript figures are never overwritten unless this points to a new directory.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def save_figure_bundle(
    fig: plt.Figure,
    output_path: Path,
    *,
    dpi: int = 220,
    bbox_inches: str | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs: dict[str, Any] = {"dpi": dpi}
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    for suffix in FIGURE_FILE_SUFFIXES:
        fig.savefig(output_path.with_suffix(suffix), **save_kwargs)


def safe_log10_ratio(fr_value: float, ps_value: float) -> float:
    numerator = max(float(fr_value), EPSILON)
    denominator = max(float(ps_value), EPSILON)
    return math.log10(numerator / denominator)


def method_winner(fr_value: float, ps_value: float) -> str:
    if math.isclose(float(fr_value), float(ps_value), rel_tol=1e-12, abs_tol=1e-15):
        return "tie"
    return "FR" if float(fr_value) < float(ps_value) else "PS_QWC"


def metric_values(group: dict[str, Any], field_name: str) -> tuple[float, float]:
    methods = group["methods"]
    return float(methods["FR"][field_name]), float(methods["PS_QWC"][field_name])


def sorted_unique_ints(summary_payload: dict[str, Any], field_name: str) -> list[int]:
    return sorted({int(group["group"][field_name]) for group in summary_payload.get("groups", [])})


def annotate_cell(axis: plt.Axes, x: int, y: int, value: float, winner: str, vmax: float) -> None:
    label = "=" if winner == "tie" else ("F" if winner == "FR" else "P")
    color = "white" if abs(value) >= 0.45 * vmax else "black"
    axis.text(
        x,
        y,
        f"{label}\n{value:+.2f}",
        ha="center",
        va="center",
        fontsize=7,
        color=color,
    )


def collect_dataset_metric_stats(summary_payload: dict[str, Any], field_name: str) -> dict[str, Any]:
    ratios: list[float] = []
    winners: Counter[str] = Counter()
    cells: list[dict[str, Any]] = []
    for group in summary_payload.get("groups", []):
        fr_value, ps_value = metric_values(group, field_name)
        ratio = safe_log10_ratio(fr_value, ps_value)
        winner = method_winner(fr_value, ps_value)
        winners[winner] += 1
        cell_record = {
            "group": dict(group["group"]),
            "fr": fr_value,
            "ps_qwc": ps_value,
            "log10_fr_over_ps": ratio,
            "winner": winner,
        }
        ratios.append(ratio)
        cells.append(cell_record)
    ranked = sorted(cells, key=lambda item: item["log10_fr_over_ps"])
    return {
        "winner_counts": {name: int(count) for name, count in winners.items()},
        "median_log10_fr_over_ps": float(np.median(ratios)) if ratios else None,
        "mean_log10_fr_over_ps": float(np.mean(ratios)) if ratios else None,
        "most_fr_favored_cell": ranked[0] if ranked else None,
        "most_ps_favored_cell": ranked[-1] if ranked else None,
    }


def plot_digital_metric_heatmaps(
    summary_payload: dict[str, Any],
    *,
    dataset_label: str,
    metric_key: str,
    output_path: Path,
) -> dict[str, Any]:
    metric_spec = METRIC_SPECS[metric_key]
    field_name = metric_spec["field"]
    qubits = sorted_unique_ints(summary_payload, "num_qubits")
    steps = sorted_unique_ints(summary_payload, "num_steps")
    budgets = sorted_unique_ints(summary_payload, "budget_total")

    matrices: dict[int, np.ndarray] = {
        num_qubits: np.full((len(steps), len(budgets)), np.nan, dtype=float) for num_qubits in qubits
    }
    winners: dict[int, np.ndarray] = {
        num_qubits: np.full((len(steps), len(budgets)), "", dtype=object) for num_qubits in qubits
    }
    step_index = {value: index for index, value in enumerate(steps)}
    budget_index = {value: index for index, value in enumerate(budgets)}

    for group in summary_payload.get("groups", []):
        num_qubits = int(group["group"]["num_qubits"])
        num_steps = int(group["group"]["num_steps"])
        budget_total = int(group["group"]["budget_total"])
        fr_value, ps_value = metric_values(group, field_name)
        ratio = safe_log10_ratio(fr_value, ps_value)
        winner = method_winner(fr_value, ps_value)
        matrices[num_qubits][step_index[num_steps], budget_index[budget_total]] = ratio
        winners[num_qubits][step_index[num_steps], budget_index[budget_total]] = winner

    finite_values = np.concatenate([matrix[np.isfinite(matrix)] for matrix in matrices.values()]) if matrices else np.array([])
    vmax = max(float(np.max(np.abs(finite_values))), 0.05) if finite_values.size else 1.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14.5, 10.0),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(2, 2)
    image = None
    win_counts: dict[str, int] = {"FR": 0, "PS_QWC": 0, "tie": 0}

    for axis, num_qubits in zip(axes_array.flat, qubits):
        matrix = matrices[num_qubits]
        image = axis.imshow(matrix, aspect="auto", cmap=RATIO_CMAP, vmin=-vmax, vmax=vmax)
        axis.set_title(f"{num_qubits} qubits")
        axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
        axis.set_yticks(range(len(steps)), [str(value) for value in steps])
        for row_index in range(matrix.shape[0]):
            for col_index in range(matrix.shape[1]):
                value = float(matrix[row_index, col_index])
                winner = str(winners[num_qubits][row_index, col_index])
                if not math.isfinite(value):
                    continue
                win_counts[winner] = win_counts.get(winner, 0) + 1
                annotate_cell(axis, col_index, row_index, value, winner, vmax)
    for axis in axes_array.flat[len(qubits) :]:
        axis.axis("off")
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array.flat[: len(qubits)], shrink=0.92)
        colorbar.set_label("log10(FR / PS-QWC), lower is better", fontsize=12)
    fig.suptitle(f"{dataset_label}: {metric_spec['title']} ratio by qubits, steps, and budget", fontsize=18)
    fig.supxlabel("Total shot budget", fontsize=15)
    fig.supylabel("Trotter step count", fontsize=15)
    save_figure_bundle(fig, output_path)
    plt.close(fig)
    return win_counts


def build_vqe_matrix(summary_payload: dict[str, Any], field_name: str) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    qubits = sorted_unique_ints(summary_payload, "num_qubits")
    budgets = sorted_unique_ints(summary_payload, "budget_total")
    matrix = np.full((len(qubits), len(budgets)), np.nan, dtype=float)
    winners = np.full((len(qubits), len(budgets)), "", dtype=object)
    qubit_index = {value: index for index, value in enumerate(qubits)}
    budget_index = {value: index for index, value in enumerate(budgets)}
    for group in summary_payload.get("groups", []):
        num_qubits = int(group["group"]["num_qubits"])
        budget_total = int(group["group"]["budget_total"])
        fr_value, ps_value = metric_values(group, field_name)
        matrix[qubit_index[num_qubits], budget_index[budget_total]] = safe_log10_ratio(fr_value, ps_value)
        winners[qubit_index[num_qubits], budget_index[budget_total]] = method_winner(fr_value, ps_value)
    return matrix, winners, qubits, budgets


def plot_vqe_overview(
    noiseless_summary: dict[str, Any],
    hardware_summary: dict[str, Any],
    *,
    metric_columns: tuple[str, str] = ("empirical", "model"),
    title: str = "VQE covariance-aware FR vs PS-QWC overview",
    output_path: Path,
) -> dict[str, dict[str, int]]:
    dataset_rows = [
        ("Noiseless", noiseless_summary),
        ("Hardware", hardware_summary),
    ]
    prepared: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, list[int], list[int]]] = {}
    all_values: list[np.ndarray] = []
    for row_index, (_label, summary_payload) in enumerate(dataset_rows):
        for col_index, metric_key in enumerate(metric_columns):
            field_name = METRIC_SPECS[metric_key]["field"]
            matrix_bundle = build_vqe_matrix(summary_payload, field_name)
            prepared[(row_index, col_index)] = matrix_bundle
            matrix = matrix_bundle[0]
            finite = matrix[np.isfinite(matrix)]
            if finite.size:
                all_values.append(finite)
    finite_values = np.concatenate(all_values) if all_values else np.array([])
    vmax = max(float(np.max(np.abs(finite_values))), 0.05) if finite_values.size else 1.0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(13.0, 8.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(2, 2)
    image = None
    counts_by_metric: dict[str, dict[str, int]] = {
        metric_key: {"FR": 0, "PS_QWC": 0, "tie": 0} for metric_key in metric_columns
    }

    for row_index, (dataset_label, _summary_payload) in enumerate(dataset_rows):
        for col_index, metric_key in enumerate(metric_columns):
            axis = axes_array[row_index, col_index]
            matrix, winners, qubits, budgets = prepared[(row_index, col_index)]
            image = axis.imshow(matrix, aspect="auto", cmap=RATIO_CMAP, vmin=-vmax, vmax=vmax)
            axis.set_title(f"{dataset_label}: {METRIC_SPECS[metric_key]['title']}")
            axis.set_xticks(range(len(budgets)), [str(value) for value in budgets])
            axis.set_yticks(range(len(qubits)), [str(value) for value in qubits])
            for row_cell in range(matrix.shape[0]):
                for col_cell in range(matrix.shape[1]):
                    value = float(matrix[row_cell, col_cell])
                    winner = str(winners[row_cell, col_cell])
                    if not math.isfinite(value):
                        continue
                    counts_by_metric[metric_key][winner] = counts_by_metric[metric_key].get(winner, 0) + 1
                    annotate_cell(axis, col_cell, row_cell, value, winner, vmax)
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes_array.ravel().tolist(), shrink=0.92)
        colorbar.set_label("log10(FR / PS-QWC), lower is better", fontsize=12)
    fig.suptitle(title, fontsize=18)
    fig.supxlabel("Total shot budget", fontsize=15)
    fig.supylabel("Qubit count", fontsize=15)
    save_figure_bundle(fig, output_path)
    plt.close(fig)
    return counts_by_metric


def group_field_value(group_payload: dict[str, Any], field_name: str) -> Any:
    value = group_payload["group"][field_name]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    return value


def groups_by_fields(summary_payload: dict[str, Any], field_names: tuple[str, ...]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for group in summary_payload.get("groups", []):
        key = tuple(group_field_value(group, field_name) for field_name in field_names)
        grouped.setdefault(key, []).append(group)
    for entries in grouped.values():
        entries.sort(key=lambda item: int(group_field_value(item, "budget_total")))
    return grouped


def extract_method_series(groups: list[dict[str, Any]], field_name: str) -> tuple[list[int], list[float], list[float]]:
    budgets = [int(group_field_value(group, "budget_total")) for group in groups]
    fr_values = [float(group["methods"]["FR"][field_name]) for group in groups]
    ps_values = [float(group["methods"]["PS_QWC"][field_name]) for group in groups]
    return budgets, fr_values, ps_values


def fit_one_over_n_line(budgets: list[int], values: list[float]) -> dict[str, Any]:
    x = np.array([1.0 / float(budget) for budget in budgets], dtype=float)
    y = np.array([float(value) for value in values], dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    design = np.column_stack((np.ones_like(x), x))
    coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coeffs[0])
    slope = float(coeffs[1])
    fit_y = intercept + slope * x
    ss_res = float(np.sum((y - fit_y) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot <= EPSILON else float(1.0 - ss_res / ss_tot)
    return {
        "budgets": [int(value) for value in np.array(budgets, dtype=int)[order]],
        "x": x.tolist(),
        "y": y.tolist(),
        "fit_y": fit_y.tolist(),
        "intercept": intercept,
        "slope": slope,
        "r2": r2,
    }


def aggregate_method_resource(groups: list[dict[str, Any]], method_name: str) -> dict[str, float]:
    resource_entries = [
        group["methods"][method_name].get("resource_summary", {})
        for group in groups
        if isinstance(group["methods"][method_name].get("resource_summary", {}), dict)
        and group["methods"][method_name].get("resource_summary", {})
    ]
    if not resource_entries:
        return {}
    numeric_keys = sorted(
        {
            key
            for entry in resource_entries
            for key, value in entry.items()
            if isinstance(value, (int, float))
        }
    )
    return {
        key: float(np.mean([float(entry[key]) for entry in resource_entries if key in entry]))
        for key in numeric_keys
    }


def build_digital_scaling_records(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped = groups_by_fields(summary_payload, ("num_qubits", "num_steps"))
    for (num_qubits, num_steps), groups in grouped.items():
        if len(groups) < 2:
            continue
        budgets, fr_empirical, ps_empirical = extract_method_series(groups, METRIC_SPECS["empirical"]["field"])
        _, fr_sampling, ps_sampling = extract_method_series(groups, METRIC_SPECS["sampling_variance"]["field"])
        empirical_log10_ratios = [safe_log10_ratio(fr_value, ps_value) for fr_value, ps_value in zip(fr_empirical, ps_empirical)]
        sampling_log10_ratios = [safe_log10_ratio(fr_value, ps_value) for fr_value, ps_value in zip(fr_sampling, ps_sampling)]
        fr_resource_summary = aggregate_method_resource(groups, "FR")
        ps_resource_summary = aggregate_method_resource(groups, "PS_QWC")
        resource_keys = sorted(
            {
                key
                for entry in (fr_resource_summary, ps_resource_summary)
                for key, value in entry.items()
                if isinstance(value, (int, float))
            }
        )
        resource_delta = {
            key: float(fr_resource_summary.get(key, 0.0) - ps_resource_summary.get(key, 0.0)) for key in resource_keys
        }
        records.append(
            {
                "num_qubits": int(num_qubits),
                "num_steps": int(num_steps),
                "budgets": budgets,
                "one_over_budgets": [1.0 / float(budget) for budget in budgets],
                "fr_empirical": fr_empirical,
                "ps_empirical": ps_empirical,
                "fr_sampling": fr_sampling,
                "ps_sampling": ps_sampling,
                "empirical_log10_ratios": empirical_log10_ratios,
                "sampling_log10_ratios": sampling_log10_ratios,
                "mean_empirical_log10_ratio": float(np.mean(empirical_log10_ratios)),
                "mean_sampling_log10_ratio": float(np.mean(sampling_log10_ratios)),
                "high_budget_total": int(budgets[-1]),
                "high_budget_empirical_delta": float(fr_empirical[-1] - ps_empirical[-1]),
                "high_budget_sampling_delta": float(fr_sampling[-1] - ps_sampling[-1]),
                "fr_empirical_fit": fit_one_over_n_line(budgets, fr_empirical),
                "ps_empirical_fit": fit_one_over_n_line(budgets, ps_empirical),
                "fr_sampling_fit": fit_one_over_n_line(budgets, fr_sampling),
                "ps_sampling_fit": fit_one_over_n_line(budgets, ps_sampling),
                "fr_resource_summary": fr_resource_summary,
                "ps_resource_summary": ps_resource_summary,
                "resource_delta": resource_delta,
            }
        )
    records.sort(key=lambda record: (record["num_qubits"], record["num_steps"]))
    for record in records:
        record["delta_empirical_intercept"] = float(
            record["fr_empirical_fit"]["intercept"] - record["ps_empirical_fit"]["intercept"]
        )
        record["delta_empirical_slope"] = float(record["fr_empirical_fit"]["slope"] - record["ps_empirical_fit"]["slope"])
        record["delta_sampling_slope"] = float(record["fr_sampling_fit"]["slope"] - record["ps_sampling_fit"]["slope"])
    return records


def build_vqe_scaling_records(summary_payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    grouped = groups_by_fields(summary_payload, ("num_qubits",))
    for (num_qubits,), groups in grouped.items():
        if len(groups) < 2:
            continue
        budgets, fr_empirical, ps_empirical = extract_method_series(groups, METRIC_SPECS["empirical"]["field"])
        _, fr_sampling, ps_sampling = extract_method_series(groups, METRIC_SPECS["sampling_variance"]["field"])
        empirical_log10_ratios = [safe_log10_ratio(fr_value, ps_value) for fr_value, ps_value in zip(fr_empirical, ps_empirical)]
        sampling_log10_ratios = [safe_log10_ratio(fr_value, ps_value) for fr_value, ps_value in zip(fr_sampling, ps_sampling)]
        records.append(
            {
                "num_qubits": int(num_qubits),
                "budgets": budgets,
                "one_over_budgets": [1.0 / float(budget) for budget in budgets],
                "fr_empirical": fr_empirical,
                "ps_empirical": ps_empirical,
                "fr_sampling": fr_sampling,
                "ps_sampling": ps_sampling,
                "empirical_log10_ratios": empirical_log10_ratios,
                "sampling_log10_ratios": sampling_log10_ratios,
                "mean_empirical_log10_ratio": float(np.mean(empirical_log10_ratios)),
                "mean_sampling_log10_ratio": float(np.mean(sampling_log10_ratios)),
                "high_budget_total": int(budgets[-1]),
                "high_budget_empirical_delta": float(fr_empirical[-1] - ps_empirical[-1]),
                "high_budget_sampling_delta": float(fr_sampling[-1] - ps_sampling[-1]),
                "fr_empirical_fit": fit_one_over_n_line(budgets, fr_empirical),
                "ps_empirical_fit": fit_one_over_n_line(budgets, ps_empirical),
                "fr_sampling_fit": fit_one_over_n_line(budgets, fr_sampling),
                "ps_sampling_fit": fit_one_over_n_line(budgets, ps_sampling),
            }
        )
    records.sort(key=lambda record: record["num_qubits"])
    return records


def select_digital_representative_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    always_fr = [record for record in records if all(value < 0.0 for value in record["empirical_log10_ratios"])]
    mostly_fr = [
        record
        for record in records
        if record["mean_empirical_log10_ratio"] < 0.0 and record["high_budget_empirical_delta"] < 0.0
    ]
    pool = always_fr or mostly_fr or records
    return min(pool, key=lambda record: (record["mean_empirical_log10_ratio"], record["high_budget_empirical_delta"]))


def select_vqe_representative_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    strongest_contrast = [
        record
        for record in records
        if all(value > 0.0 for value in record["empirical_log10_ratios"])
        and all(value < 0.0 for value in record["sampling_log10_ratios"])
    ]
    fallback = [
        record
        for record in records
        if record["high_budget_empirical_delta"] > 0.0 and record["high_budget_sampling_delta"] < 0.0
    ]
    pool = strongest_contrast or fallback or records
    return max(
        pool,
        key=lambda record: (
            record["mean_empirical_log10_ratio"] - record["mean_sampling_log10_ratio"],
            record["high_budget_empirical_delta"] - record["high_budget_sampling_delta"],
        ),
    )


def _plot_scaling_panel(axis: plt.Axes, record: dict[str, Any], *, title: str) -> None:
    order = np.argsort(np.array(record["one_over_budgets"], dtype=float))
    x = np.array(record["one_over_budgets"], dtype=float)[order]
    fr_empirical = np.array(record["fr_empirical"], dtype=float)[order]
    ps_empirical = np.array(record["ps_empirical"], dtype=float)[order]
    fr_sampling = np.array(record["fr_sampling"], dtype=float)[order]
    ps_sampling = np.array(record["ps_sampling"], dtype=float)[order]

    axis.plot(x, fr_empirical, marker="o", linewidth=2.0, color=FR_COLOR, label="FR empirical MSE")
    axis.plot(x, ps_empirical, marker="s", linewidth=2.0, color=PS_COLOR, label="PS-QWC empirical MSE")
    axis.plot(x, fr_sampling, marker="o", linestyle="--", linewidth=1.5, color=FR_COLOR, alpha=0.8, label="FR sampling variance")
    axis.plot(x, ps_sampling, marker="s", linestyle="--", linewidth=1.5, color=PS_COLOR, alpha=0.8, label="PS-QWC sampling variance")
    axis.plot(
        record["fr_empirical_fit"]["x"],
        record["fr_empirical_fit"]["fit_y"],
        linestyle=":",
        linewidth=1.8,
        color=FR_COLOR,
        alpha=0.9,
    )
    axis.plot(
        record["ps_empirical_fit"]["x"],
        record["ps_empirical_fit"]["fit_y"],
        linestyle=":",
        linewidth=1.8,
        color=PS_COLOR,
        alpha=0.9,
    )
    axis.set_title(title)
    axis.set_xlabel("1 / N")
    axis.set_ylabel("Estimator MSE / variance")
    axis.grid(True, alpha=0.25)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fit_text = (
        f"FR fit: a={record['fr_empirical_fit']['intercept']:.2e}, b={record['fr_empirical_fit']['slope']:.2e}\n"
        f"PS fit: a={record['ps_empirical_fit']['intercept']:.2e}, b={record['ps_empirical_fit']['slope']:.2e}"
    )
    axis.text(
        0.97,
        0.96,
        fit_text,
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
    )


def summarize_representative_record(record: dict[str, Any], dataset_name: str) -> dict[str, Any]:
    summary = {
        "dataset": dataset_name,
        "num_qubits": int(record["num_qubits"]),
        "budgets": [int(value) for value in record["budgets"]],
        "high_budget_total": int(record["high_budget_total"]),
        "high_budget_empirical_delta": float(record["high_budget_empirical_delta"]),
        "high_budget_sampling_delta": float(record["high_budget_sampling_delta"]),
        "mean_empirical_log10_ratio": float(record["mean_empirical_log10_ratio"]),
        "mean_sampling_log10_ratio": float(record["mean_sampling_log10_ratio"]),
        "fr_empirical_fit": dict(record["fr_empirical_fit"]),
        "ps_empirical_fit": dict(record["ps_empirical_fit"]),
    }
    if "num_steps" in record:
        summary["num_steps"] = int(record["num_steps"])
    if "resource_delta" in record:
        summary["resource_delta"] = dict(record["resource_delta"])
    return summary


def serialize_scaling_record(record: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "num_qubits": int(record["num_qubits"]),
        "budgets": [int(value) for value in record["budgets"]],
        "one_over_budgets": [float(value) for value in record["one_over_budgets"]],
        "fr_empirical": [float(value) for value in record["fr_empirical"]],
        "ps_empirical": [float(value) for value in record["ps_empirical"]],
        "fr_sampling": [float(value) for value in record["fr_sampling"]],
        "ps_sampling": [float(value) for value in record["ps_sampling"]],
        "fr_empirical_fit": dict(record["fr_empirical_fit"]),
        "ps_empirical_fit": dict(record["ps_empirical_fit"]),
        "fr_sampling_fit": dict(record["fr_sampling_fit"]),
        "ps_sampling_fit": dict(record["ps_sampling_fit"]),
        "delta_alpha": float(record["fr_empirical_fit"]["intercept"] - record["ps_empirical_fit"]["intercept"]),
        "delta_beta": float(record["fr_empirical_fit"]["slope"] - record["ps_empirical_fit"]["slope"]),
    }
    if "num_steps" in record:
        payload["num_steps"] = int(record["num_steps"])
    if "resource_delta" in record:
        payload["resource_delta"] = dict(record["resource_delta"])
    return payload


def serialize_scaling_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [serialize_scaling_record(record) for record in records]


def plot_representative_budget_scaling(
    digital_records: list[dict[str, Any]],
    vqe_records: list[dict[str, Any]],
    *,
    output_path: Path,
) -> dict[str, Any]:
    digital_record = select_digital_representative_record(digital_records)
    vqe_record = select_vqe_representative_record(vqe_records)
    if digital_record is None or vqe_record is None:
        raise ValueError("Representative budget-scaling figure requires both digital and VQE scaling records")

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(14.5, 6.2), constrained_layout=False)
    axes_array = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.84, bottom=0.23, wspace=0.10)

    _plot_scaling_panel(
        axes_array[0],
        digital_record,
        title=f"Digital hardware: n={digital_record['num_qubits']}, steps={digital_record['num_steps']}",
    )
    _plot_scaling_panel(
        axes_array[1],
        vqe_record,
        title=f"Locked-state VQE hardware: n={vqe_record['num_qubits']}",
    )
    handles, labels = axes_array[0].get_legend_handles_labels()
    fig.suptitle("Representative fixed-budget scaling and fitted 1/N laws", fontsize=17, y=0.93)
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.05),
        borderaxespad=0.0,
    )
    save_figure_bundle(fig, output_path)
    plt.close(fig)

    return {
        "digital": summarize_representative_record(digital_record, "digital_hardware_fill"),
        "vqe": summarize_representative_record(vqe_record, "vqe_hardware_fill"),
    }


def linear_relation_summary(xs: list[float], ys: list[float]) -> dict[str, float] | None:
    if len(xs) < 2 or len(ys) < 2:
        return None
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    if np.std(x) <= EPSILON:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    corr = None if np.std(y) <= EPSILON else float(np.corrcoef(x, y)[0, 1])
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": corr,
    }


def fitted_crossover_summary(records: list[dict[str, Any]], *, budget_window: tuple[int, int]) -> dict[str, Any]:
    lower_budget, upper_budget = budget_window
    valid_cells: list[dict[str, Any]] = []
    for record in records:
        delta_alpha = float(record["fr_empirical_fit"]["intercept"] - record["ps_empirical_fit"]["intercept"])
        delta_beta = float(record["fr_empirical_fit"]["slope"] - record["ps_empirical_fit"]["slope"])
        if not (delta_alpha > 0.0 and delta_beta < 0.0):
            continue
        crossover_budget = abs(delta_beta) / delta_alpha
        cell_summary = {
            "num_qubits": int(record["num_qubits"]),
            "delta_alpha": delta_alpha,
            "delta_beta": delta_beta,
            "fitted_crossover_budget": float(crossover_budget),
        }
        if "num_steps" in record:
            cell_summary["num_steps"] = int(record["num_steps"])
        valid_cells.append(cell_summary)

    crossover_values = sorted(cell["fitted_crossover_budget"] for cell in valid_cells)
    return {
        "total_cells": int(len(records)),
        "valid_cells": int(len(valid_cells)),
        "within_budget_window": int(sum(lower_budget <= value <= upper_budget for value in crossover_values)),
        "below_budget_window": int(sum(value < lower_budget for value in crossover_values)),
        "above_budget_window": int(sum(value > upper_budget for value in crossover_values)),
        "budget_window": {"min": int(lower_budget), "max": int(upper_budget)},
        "median_fitted_crossover_budget": float(np.median(crossover_values)) if crossover_values else None,
        "min_fitted_crossover_budget": float(min(crossover_values)) if crossover_values else None,
        "max_fitted_crossover_budget": float(max(crossover_values)) if crossover_values else None,
        "cells": valid_cells,
    }


def _resource_style_maps(records: list[dict[str, Any]]) -> tuple[dict[int, Any], dict[int, str]]:
    qubits = sorted({int(record["num_qubits"]) for record in records})
    steps = sorted({int(record["num_steps"]) for record in records})
    cmap = matplotlib.colormaps.get_cmap("viridis")
    denominator = max(len(qubits) - 1, 1)
    qubit_colors = {
        qubit: cmap(index / denominator) if denominator else cmap(0.5) for index, qubit in enumerate(qubits)
    }
    step_markers = {
        step: RESOURCE_STEP_MARKERS[index % len(RESOURCE_STEP_MARKERS)] for index, step in enumerate(steps)
    }
    return qubit_colors, step_markers


def _plot_resource_scatter(
    axis: plt.Axes,
    xs: list[float],
    ys: list[float],
    records: list[dict[str, Any]],
    qubit_colors: dict[int, Any],
    step_markers: dict[int, str],
    *,
    title: str,
    ylabel: str,
) -> dict[str, float] | None:
    summary = linear_relation_summary(xs, ys)
    for x_value, y_value, record in zip(xs, ys, records):
        axis.scatter(
            [x_value],
            [y_value],
            s=74,
            color=qubit_colors[int(record["num_qubits"])],
            marker=step_markers[int(record["num_steps"])],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
        )
    if summary is not None:
        line_x = np.linspace(min(xs), max(xs), 200)
        line_y = summary["intercept"] + summary["slope"] * line_x
        axis.plot(line_x, line_y, linestyle="--", linewidth=1.5, color="0.3")
        corr_value = summary["correlation"]
        corr_text = "nan" if corr_value is None else f"{corr_value:+.2f}"
        axis.text(
            0.03,
            0.97,
            f"corr = {corr_text}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "0.7", "alpha": 0.9},
        )
    axis.set_title(title)
    axis.set_xlabel("FR - PS-QWC mean 2Q overhead")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    return summary


def plot_digital_resource_diagnostics(records: list[dict[str, Any]], *, output_path: Path) -> dict[str, Any]:
    usable_records = [
        record
        for record in records
        if math.isfinite(float(record.get("resource_delta", {}).get("twoq_overhead_mean", float("nan"))))
        and math.isfinite(float(record.get("delta_empirical_intercept", float("nan"))))
        and math.isfinite(float(record.get("delta_empirical_slope", float("nan"))))
    ]
    if not usable_records:
        raise ValueError("Digital resource diagnostics require resource summaries in the hardware summary JSON")

    twoq_overheads = [float(record["resource_delta"]["twoq_overhead_mean"]) for record in usable_records]
    delta_intercepts = [float(record["delta_empirical_intercept"]) for record in usable_records]
    delta_slopes = [float(record["delta_empirical_slope"]) for record in usable_records]
    qubit_colors, step_markers = _resource_style_maps(usable_records)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16.4, 5.4), constrained_layout=False)
    axes_array = np.atleast_1d(axes)
    intercept_summary = _plot_resource_scatter(
        axes_array[0],
        twoq_overheads,
        delta_intercepts,
        usable_records,
        qubit_colors,
        step_markers,
        title="Hardware overhead vs fitted intercept proxy",
        ylabel="FR - PS fitted intercept",
    )
    slope_summary = _plot_resource_scatter(
        axes_array[1],
        twoq_overheads,
        delta_slopes,
        usable_records,
        qubit_colors,
        step_markers,
        title="Hardware overhead vs fitted 1/N coefficient",
        ylabel="FR - PS fitted slope",
    )
    qubit_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="None",
            markersize=8,
            markerfacecolor=qubit_colors[qubit],
            markeredgecolor="white",
            label=f"n={qubit}",
        )
        for qubit in sorted(qubit_colors)
    ]
    step_handles = [
        Line2D(
            [0],
            [0],
            marker=step_markers[step],
            linestyle="None",
            markersize=8,
            markerfacecolor="0.35",
            markeredgecolor="white",
            label=f"s={step}",
        )
        for step in sorted(step_markers)
    ]
    fig.subplots_adjust(left=0.07, right=0.80, top=0.84, bottom=0.13, wspace=0.17)
    qubit_legend = fig.legend(
        qubit_handles,
        [handle.get_label() for handle in qubit_handles],
        title="Qubit count",
        loc="upper left",
        bbox_to_anchor=(0.81, 0.84),
        frameon=False,
    )
    fig.add_artist(qubit_legend)
    fig.legend(
        step_handles,
        [handle.get_label() for handle in step_handles],
        title="Trotter steps",
        loc="upper left",
        bbox_to_anchor=(0.81, 0.46),
        frameon=False,
        ncol=2,
        columnspacing=1.2,
        handletextpad=0.4,
    )
    fig.suptitle("Digital hardware resource-vs-error diagnostics from 1/N fits", fontsize=17, y=0.97)
    save_figure_bundle(fig, output_path, bbox_inches="tight")
    plt.close(fig)

    depth_overheads = [float(record["resource_delta"].get("depth_overhead_mean", float("nan"))) for record in usable_records]
    singleq_overheads = [float(record["resource_delta"].get("singleq_overhead_mean", float("nan"))) for record in usable_records]

    return {
        "record_count": len(usable_records),
        "records": [
            {
                "num_qubits": int(record["num_qubits"]),
                "num_steps": int(record["num_steps"]),
                "delta_empirical_intercept": float(record["delta_empirical_intercept"]),
                "delta_empirical_slope": float(record["delta_empirical_slope"]),
                "delta_sampling_slope": float(record["delta_sampling_slope"]),
                "resource_delta": dict(record["resource_delta"]),
            }
            for record in usable_records
        ],
        "correlations": {
            "twoq_overhead_vs_intercept": intercept_summary,
            "twoq_overhead_vs_slope": slope_summary,
            "depth_overhead_vs_intercept": linear_relation_summary(depth_overheads, delta_intercepts),
            "depth_overhead_vs_slope": linear_relation_summary(depth_overheads, delta_slopes),
            "singleq_overhead_vs_intercept": linear_relation_summary(singleq_overheads, delta_intercepts),
            "singleq_overhead_vs_slope": linear_relation_summary(singleq_overheads, delta_slopes),
        },
    }


def compute_boundary_index(values: list[float]) -> int | None:
    for index, value in enumerate(values):
        if value < 0.0:
            return index
    return None


def _boundary_matrix_from_digital_summary(
    summary_payload: dict[str, Any],
    field_name: str,
) -> tuple[np.ndarray, list[int], list[int], list[int]]:
    qubits = sorted_unique_ints(summary_payload, "num_qubits")
    steps = sorted_unique_ints(summary_payload, "num_steps")
    budgets = sorted_unique_ints(summary_payload, "budget_total")
    qubit_index = {value: index for index, value in enumerate(qubits)}
    step_index = {value: index for index, value in enumerate(steps)}
    budget_index = {value: index for index, value in enumerate(budgets)}

    ratio_cube = np.full((len(qubits), len(steps), len(budgets)), np.nan, dtype=float)
    for group in summary_payload.get("groups", []):
        num_qubits = int(group["group"]["num_qubits"])
        num_steps = int(group["group"]["num_steps"])
        budget_total = int(group["group"]["budget_total"])
        fr_value, ps_value = metric_values(group, field_name)
        ratio_cube[
            qubit_index[num_qubits],
            step_index[num_steps],
            budget_index[budget_total],
        ] = safe_log10_ratio(fr_value, ps_value)

    boundary_matrix = np.full((len(qubits), len(steps)), np.nan, dtype=float)
    for qubit_pos in range(len(qubits)):
        for step_pos in range(len(steps)):
            row_values = ratio_cube[qubit_pos, step_pos, :]
            if np.any(~np.isfinite(row_values)):
                continue
            boundary_index = compute_boundary_index(row_values.tolist())
            if boundary_index is None:
                boundary_matrix[qubit_pos, step_pos] = -1.0
            else:
                boundary_matrix[qubit_pos, step_pos] = float(boundary_index)
    return boundary_matrix, qubits, steps, budgets


def _boundary_matrix_from_vqe_summary(
    summary_payload: dict[str, Any],
    field_name: str,
) -> tuple[np.ndarray, list[int], list[int]]:
    qubits = sorted_unique_ints(summary_payload, "num_qubits")
    budgets = sorted_unique_ints(summary_payload, "budget_total")
    qubit_index = {value: index for index, value in enumerate(qubits)}
    budget_index = {value: index for index, value in enumerate(budgets)}

    ratio_matrix = np.full((len(qubits), len(budgets)), np.nan, dtype=float)
    for group in summary_payload.get("groups", []):
        num_qubits = int(group["group"]["num_qubits"])
        budget_total = int(group["group"]["budget_total"])
        fr_value, ps_value = metric_values(group, field_name)
        ratio_matrix[qubit_index[num_qubits], budget_index[budget_total]] = safe_log10_ratio(fr_value, ps_value)

    boundary_matrix = np.full((len(qubits), 1), np.nan, dtype=float)
    for qubit_pos in range(len(qubits)):
        row_values = ratio_matrix[qubit_pos, :]
        if np.any(~np.isfinite(row_values)):
            continue
        boundary_index = compute_boundary_index(row_values.tolist())
        if boundary_index is None:
            boundary_matrix[qubit_pos, 0] = -1.0
        else:
            boundary_matrix[qubit_pos, 0] = float(boundary_index)
    return boundary_matrix, qubits, budgets


def _plot_discrete_boundary_panel(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    row_labels: list[int],
    col_labels: list[str],
    title: str,
    max_index: int,
) -> None:
    cmap = matplotlib.colormaps.get_cmap("tab10")
    for row_index in range(matrix.shape[0]):
        for col_index in range(matrix.shape[1]):
            value = matrix[row_index, col_index]
            if np.isnan(value):
                face = (1.0, 1.0, 1.0, 1.0)
                text = ""
            elif value < 0:
                face = (0.7, 0.7, 0.7, 1.0)
                text = "None"
            else:
                face = cmap(int(value) % max(max_index + 1, 1))
                text = str(int(value))
            rect = plt.Rectangle(
                (col_index - 0.5, row_index - 0.5),
                1,
                1,
                facecolor=face,
                edgecolor="black",
                linewidth=0.6,
            )
            axis.add_patch(rect)
            if text:
                axis.text(col_index, row_index, text, ha="center", va="center", fontsize=8)

    axis.set_xlim(-0.5, matrix.shape[1] - 0.5)
    axis.set_ylim(matrix.shape[0] - 0.5, -0.5)
    axis.set_title(title)
    axis.set_xticks(range(len(col_labels)), col_labels)
    axis.set_yticks(range(len(row_labels)), [str(value) for value in row_labels])


def plot_empirical_boundary_overview(
    digital_noiseless_summary: dict[str, Any],
    digital_hardware_summary: dict[str, Any],
    vqe_noiseless_summary: dict[str, Any],
    vqe_hardware_summary: dict[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    field_name = METRIC_SPECS["empirical"]["field"]

    digital_noiseless_matrix, digital_qubits, digital_steps, digital_budgets = _boundary_matrix_from_digital_summary(
        digital_noiseless_summary,
        field_name,
    )
    digital_hardware_matrix, _, _, _ = _boundary_matrix_from_digital_summary(
        digital_hardware_summary,
        field_name,
    )
    vqe_noiseless_matrix, vqe_qubits, vqe_budgets = _boundary_matrix_from_vqe_summary(
        vqe_noiseless_summary,
        field_name,
    )
    vqe_hardware_matrix, _, _ = _boundary_matrix_from_vqe_summary(
        vqe_hardware_summary,
        field_name,
    )

    matrices = [digital_noiseless_matrix, digital_hardware_matrix, vqe_noiseless_matrix, vqe_hardware_matrix]
    finite_nonnegative = [matrix[np.isfinite(matrix) & (matrix >= 0)] for matrix in matrices]
    flat_nonnegative = np.concatenate([values for values in finite_nonnegative if values.size]) if any(values.size for values in finite_nonnegative) else np.array([])
    max_index = int(np.max(flat_nonnegative)) if flat_nonnegative.size else 0

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(12.5, 9.5),
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes).reshape(2, 2)

    _plot_discrete_boundary_panel(
        axes_array[0, 0],
        digital_noiseless_matrix,
        row_labels=digital_qubits,
        col_labels=[str(step) for step in digital_steps],
        title="Digital noiseless",
        max_index=max_index,
    )
    _plot_discrete_boundary_panel(
        axes_array[0, 1],
        digital_hardware_matrix,
        row_labels=digital_qubits,
        col_labels=[str(step) for step in digital_steps],
        title="Digital hardware",
        max_index=max_index,
    )
    _plot_discrete_boundary_panel(
        axes_array[1, 0],
        vqe_noiseless_matrix,
        row_labels=vqe_qubits,
        col_labels=["boundary"],
        title="VQE locked noiseless",
        max_index=max_index,
    )
    _plot_discrete_boundary_panel(
        axes_array[1, 1],
        vqe_hardware_matrix,
        row_labels=vqe_qubits,
        col_labels=["boundary"],
        title="VQE locked hardware",
        max_index=max_index,
    )

    axes_array[0, 0].set_ylabel("Qubit count", fontsize=12)
    axes_array[1, 0].set_ylabel("Qubit count", fontsize=12)
    axes_array[0, 0].set_xlabel("Trotter steps", fontsize=12)
    axes_array[0, 1].set_xlabel("Trotter steps", fontsize=12)
    axes_array[1, 0].set_xlabel("Boundary index", fontsize=12)
    axes_array[1, 1].set_xlabel("Boundary index", fontsize=12)

    fig.suptitle(
        "Empirical-MSE boundary index: first budget rung where FR beats PS-QWC",
        fontsize=17,
    )
    save_figure_bundle(fig, output_path)
    plt.close(fig)

    return {
        "budgets": {
            "digital": digital_budgets,
            "vqe": vqe_budgets,
        },
        "max_boundary_index": max_index,
        "none_counts": {
            "digital_noiseless": int(np.sum(digital_noiseless_matrix < 0)),
            "digital_hardware": int(np.sum(digital_hardware_matrix < 0)),
            "vqe_noiseless": int(np.sum(vqe_noiseless_matrix < 0)),
            "vqe_hardware": int(np.sum(vqe_hardware_matrix < 0)),
        },
    }


def plot_win_counts(report_payload: dict[str, Any], output_path: Path) -> None:
    dataset_order = [
        "digital_noiseless",
        "digital_hardware_fill",
        "vqe_noiseless",
        "vqe_hardware_fill",
    ]
    metric_order = ["empirical", "model", "sampling_variance"]
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(16.0, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)
    x_positions = np.arange(len(dataset_order))
    width = 0.34
    for axis, metric_key in zip(axes_array, metric_order):
        fr_counts = [
            int(report_payload["datasets"][dataset_name]["metrics"][metric_key]["winner_counts"].get("FR", 0))
            for dataset_name in dataset_order
        ]
        ps_counts = [
            int(report_payload["datasets"][dataset_name]["metrics"][metric_key]["winner_counts"].get("PS_QWC", 0))
            for dataset_name in dataset_order
        ]
        fr_bars = axis.bar(x_positions - width / 2.0, fr_counts, width=width, color=FR_COLOR, label="FR")
        ps_bars = axis.bar(x_positions + width / 2.0, ps_counts, width=width, color=PS_COLOR, label="PS-QWC")
        axis.set_title(METRIC_SPECS[metric_key]["title"])
        axis.set_xticks(x_positions, [DATASET_LABELS[name] for name in dataset_order], rotation=18, ha="right")
        axis.grid(True, axis="y", alpha=0.25)
        for bars in (fr_bars, ps_bars):
            for bar in bars:
                height = float(bar.get_height())
                axis.text(bar.get_x() + bar.get_width() / 2.0, height + 0.25, str(int(height)), ha="center", va="bottom", fontsize=9)
    axes_array[0].set_ylabel("Winning cells", fontsize=12)
    axes_array[0].legend(loc="upper left", frameon=False)
    fig.suptitle("FR vs PS-QWC win counts across covariance-aware workflow summaries", fontsize=17)
    save_figure_bundle(fig, output_path)
    plt.close(fig)


def report_to_markdown(report_payload: dict[str, Any]) -> str:
    lines = [
        "# Covariance-Aware QWC Manuscript Summary",
        "",
        f"Date tag: {report_payload['date_tag']}",
        "",
    ]
    for dataset_name, dataset_payload in report_payload["datasets"].items():
        lines.append(f"## {DATASET_LABELS[dataset_name]}")
        lines.append(f"- Cells: {dataset_payload['cells']}")
        for metric_key in ("empirical", "model", "sampling_variance"):
            metric_payload = dataset_payload["metrics"][metric_key]
            winner_counts = metric_payload["winner_counts"]
            lines.append(
                "- "
                f"{METRIC_SPECS[metric_key]['title']}: FR {winner_counts.get('FR', 0)}, "
                f"PS-QWC {winner_counts.get('PS_QWC', 0)}, tie {winner_counts.get('tie', 0)}, "
                f"median log10(FR/PS) {metric_payload['median_log10_fr_over_ps']:+.3f}"
            )
        lines.append("")
    if "representative_budget_scaling" in report_payload:
        lines.append("## Representative Budget Scaling")
        digital_payload = report_payload["representative_budget_scaling"]["digital"]
        vqe_payload = report_payload["representative_budget_scaling"]["vqe"]
        lines.append(
            "- "
            f"Digital hardware representative cell: n={digital_payload['num_qubits']}, steps={digital_payload['num_steps']}, "
            f"highest-budget empirical delta {digital_payload['high_budget_empirical_delta']:+.3e}"
        )
        lines.append(
            "- "
            f"VQE hardware representative cell: n={vqe_payload['num_qubits']}, highest-budget empirical delta "
            f"{vqe_payload['high_budget_empirical_delta']:+.3e}, highest-budget sampling delta {vqe_payload['high_budget_sampling_delta']:+.3e}"
        )
        lines.append("")
    if "digital_resource_analysis" in report_payload:
        lines.append("## Digital Resource Diagnostics")
        correlation_payload = report_payload["digital_resource_analysis"]["correlations"]
        twoq_intercept = correlation_payload.get("twoq_overhead_vs_intercept") or {}
        twoq_slope = correlation_payload.get("twoq_overhead_vs_slope") or {}
        lines.append(
            "- "
            f"2Q overhead vs fitted intercept: corr {twoq_intercept.get('correlation', 'nan')}"
        )
        lines.append(
            "- "
            f"2Q overhead vs fitted slope: corr {twoq_slope.get('correlation', 'nan')}"
        )
        lines.append("")
    if "fitted_crossover_summary" in report_payload:
        lines.append("## Fitted Crossover Summary")
        for dataset_name, summary in report_payload["fitted_crossover_summary"].items():
            lines.append(
                "- "
                f"{DATASET_LABELS[dataset_name]}: valid {summary['valid_cells']}/{summary['total_cells']}, "
                f"below window {summary['below_budget_window']}, within window {summary['within_budget_window']}, "
                f"above window {summary['above_budget_window']}, median fitted N_c {summary['median_fitted_crossover_budget']}"
            )
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def resolve_summary_paths(args: argparse.Namespace) -> argparse.Namespace:
    default_digital_hardware = OUT_DIR / f"floquet_hardware_{DEFAULT_DATE_TAG}_summary.json"
    default_vqe_hardware = OUT_DIR / f"locked_state_vqe_hardware_{DEFAULT_DATE_TAG}_summary.json"
    if args.digital_hardware_summary == default_digital_hardware:
        args.digital_hardware_summary = OUT_DIR / f"floquet_hardware_{args.date_tag}_summary.json"
    if args.vqe_hardware_summary == default_vqe_hardware:
        args.vqe_hardware_summary = OUT_DIR / f"locked_state_vqe_hardware_{args.date_tag}_summary.json"
    return args


def copy_outputs(paths: dict[str, Path], destination_dir: Path | None) -> dict[str, str]:
    copied: dict[str, str] = {}
    if destination_dir is None:
        return copied
    destination_dir.mkdir(parents=True, exist_ok=True)
    for key, path in paths.items():
        candidates = [path]
        if path.suffix.lower() == ".png":
            candidates.extend(path.with_suffix(suffix) for suffix in FIGURE_FILE_SUFFIXES[1:])
        for candidate in candidates:
            if candidate.suffix.lower() not in FIGURE_FILE_SUFFIXES or not candidate.exists():
                continue
            destination = destination_dir / candidate.name
            shutil.copy2(candidate, destination)
            copied_key = key if candidate == path else f"{key}_{candidate.suffix.lstrip('.') }"
            copied[copied_key] = portable_path_string(destination)
    return copied


def main() -> int:
    args = resolve_summary_paths(parse_args())
    output_dir = args.output_dir or (OUT_DIR / f"manuscript_figures_{args.date_tag}")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {
        "digital_noiseless": load_json(args.digital_noiseless_summary),
        "digital_hardware_fill": load_json(args.digital_hardware_summary),
        "vqe_noiseless": load_json(args.vqe_noiseless_summary),
        "vqe_hardware_fill": load_json(args.vqe_hardware_summary),
    }

    generated_paths: dict[str, Path] = {}

    for dataset_name in ("digital_noiseless", "digital_hardware_fill"):
        dataset_label = DATASET_LABELS[dataset_name]
        for metric_key in ("empirical", "model", "sampling_variance"):
            output_path = output_dir / f"{dataset_name}_{METRIC_SPECS[metric_key]['filename_stub']}.png"
            plot_digital_metric_heatmaps(
                summaries[dataset_name],
                dataset_label=dataset_label,
                metric_key=metric_key,
                output_path=output_path,
            )
            generated_paths[f"{dataset_name}_{metric_key}_figure"] = output_path

    vqe_overview_path = output_dir / "vqe_empirical_model_ratio_overview.png"
    plot_vqe_overview(
        summaries["vqe_noiseless"],
        summaries["vqe_hardware_fill"],
        metric_columns=("empirical", "model"),
        title="VQE covariance-aware FR vs PS-QWC overview",
        output_path=vqe_overview_path,
    )
    generated_paths["vqe_overview_figure"] = vqe_overview_path

    vqe_empirical_sampling_overview_path = output_dir / "vqe_empirical_sampling_ratio_overview.png"
    plot_vqe_overview(
        summaries["vqe_noiseless"],
        summaries["vqe_hardware_fill"],
        metric_columns=("empirical", "sampling_variance"),
        title="VQE realized MSE vs sampling-variance overview",
        output_path=vqe_empirical_sampling_overview_path,
    )
    generated_paths["vqe_empirical_sampling_overview_figure"] = vqe_empirical_sampling_overview_path

    digital_noiseless_scaling_records = build_digital_scaling_records(summaries["digital_noiseless"])
    digital_scaling_records = build_digital_scaling_records(summaries["digital_hardware_fill"])
    vqe_noiseless_scaling_records = build_vqe_scaling_records(summaries["vqe_noiseless"])
    vqe_scaling_records = build_vqe_scaling_records(summaries["vqe_hardware_fill"])

    budget_scaling_path = output_dir / "representative_budget_scaling.png"
    representative_budget_scaling = plot_representative_budget_scaling(
        digital_scaling_records,
        vqe_scaling_records,
        output_path=budget_scaling_path,
    )
    generated_paths["representative_budget_scaling_figure"] = budget_scaling_path

    digital_resource_path = output_dir / "digital_resource_scaling_diagnostics.png"
    digital_resource_analysis = plot_digital_resource_diagnostics(
        digital_scaling_records,
        output_path=digital_resource_path,
    )
    generated_paths["digital_resource_scaling_figure"] = digital_resource_path

    digital_resource_json_path = output_dir / "digital_resource_scaling_analysis.json"
    write_json(digital_resource_json_path, digital_resource_analysis)
    generated_paths["digital_resource_scaling_json"] = digital_resource_json_path

    all_scaling_fits_path = output_dir / "all_budget_scaling_fit_records.json"
    write_json(
        all_scaling_fits_path,
        {
            "digital_noiseless": serialize_scaling_records(digital_noiseless_scaling_records),
            "digital_hardware_fill": serialize_scaling_records(digital_scaling_records),
            "vqe_noiseless": serialize_scaling_records(vqe_noiseless_scaling_records),
            "vqe_hardware_fill": serialize_scaling_records(vqe_scaling_records),
        },
    )
    generated_paths["all_scaling_fit_records_json"] = all_scaling_fits_path

    boundary_overview_path = output_dir / "empirical_boundary_index_overview.png"
    boundary_summary = plot_empirical_boundary_overview(
        summaries["digital_noiseless"],
        summaries["digital_hardware_fill"],
        summaries["vqe_noiseless"],
        summaries["vqe_hardware_fill"],
        output_path=boundary_overview_path,
    )
    generated_paths["boundary_overview_figure"] = boundary_overview_path

    report_payload: dict[str, Any] = {
        "date_tag": str(args.date_tag),
        "source_summaries": {
            "digital_noiseless": portable_path_string(args.digital_noiseless_summary),
            "digital_hardware_fill": portable_path_string(args.digital_hardware_summary),
            "vqe_noiseless": portable_path_string(args.vqe_noiseless_summary),
            "vqe_hardware_fill": portable_path_string(args.vqe_hardware_summary),
        },
        "datasets": {},
        "boundary_summary": boundary_summary,
        "representative_budget_scaling": representative_budget_scaling,
        "digital_resource_analysis": digital_resource_analysis,
        "fitted_crossover_summary": {},
        "main_text_figure_manifest": {},
        "outputs": {},
    }

    for dataset_name, summary_payload in summaries.items():
        report_payload["datasets"][dataset_name] = {
            "cells": int(len(summary_payload.get("groups", []))),
            "metrics": {
                metric_key: collect_dataset_metric_stats(summary_payload, METRIC_SPECS[metric_key]["field"])
                for metric_key in ("empirical", "model", "sampling_variance")
            },
        }

    report_payload["fitted_crossover_summary"] = {
        "digital_noiseless": fitted_crossover_summary(digital_noiseless_scaling_records, budget_window=(2000, 16000)),
        "digital_hardware_fill": fitted_crossover_summary(digital_scaling_records, budget_window=(2000, 16000)),
        "vqe_noiseless": fitted_crossover_summary(vqe_noiseless_scaling_records, budget_window=(2000, 16000)),
        "vqe_hardware_fill": fitted_crossover_summary(vqe_scaling_records, budget_window=(2000, 16000)),
    }

    win_count_path = output_dir / "workflow_method_win_counts.png"
    plot_win_counts(report_payload, win_count_path)
    generated_paths["win_count_figure"] = win_count_path

    report_path = output_dir / "manuscript_figure_report.json"
    markdown_path = output_dir / "manuscript_figure_summary.md"
    generated_paths["report_json"] = report_path
    generated_paths["summary_markdown"] = markdown_path

    report_payload["main_text_figure_manifest"] = {
        "figure_2_digital_components": {
            "digital_noiseless_empirical": portable_path_string(generated_paths["digital_noiseless_empirical_figure"]),
            "digital_noiseless_sampling_variance": portable_path_string(generated_paths["digital_noiseless_sampling_variance_figure"]),
            "digital_hardware_empirical": portable_path_string(generated_paths["digital_hardware_fill_empirical_figure"]),
            "digital_hardware_sampling_variance": portable_path_string(generated_paths["digital_hardware_fill_sampling_variance_figure"]),
        },
        "figure_3_vqe_overview": portable_path_string(generated_paths["vqe_empirical_sampling_overview_figure"]),
        "figure_4_budget_scaling": portable_path_string(generated_paths["representative_budget_scaling_figure"]),
        "appendix_resource_diagnostic": portable_path_string(generated_paths["digital_resource_scaling_figure"]),
        "appendix_boundary_summary": portable_path_string(generated_paths["boundary_overview_figure"]),
        "appendix_win_counts": portable_path_string(generated_paths["win_count_figure"]),
    }

    copied_paths = copy_outputs(generated_paths, args.copy_dir)
    report_payload["outputs"] = {key: portable_path_string(path) for key, path in generated_paths.items()}
    if copied_paths:
        report_payload["copied_outputs"] = copied_paths

    write_json(report_path, report_payload)
    write_text(markdown_path, report_to_markdown(report_payload))

    print(f"output_dir: {portable_path_string(output_dir)}")
    figure_count = sum(1 for path in generated_paths.values() if path.suffix.lower() in set(FIGURE_FILE_SUFFIXES))
    print(f"figure_outputs: {figure_count}")
    print(f"report: {portable_path_string(report_path)}")
    if copied_paths:
        print(f"copied_outputs: {len(copied_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())