from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TIMING_METRICS = [
    "model_load_seconds",
    "inference_seconds",
    "image_save_seconds",
    "generation_total_seconds",
    "execution_time_seconds",
    "monitoring_duration_seconds",
]

ENERGY_METRICS = [
    "gpu_energy_j",
    "gpu_energy_wh",
    "joules_per_image",
    "wh_per_image",
]

GPU_METRICS = [
    "gpu_power_mean_w",
    "gpu_power_max_w",
    "gpu_utilization_mean_percent",
    "gpu_utilization_max_percent",
    "gpu_temperature_start_c",
    "gpu_temperature_max_c",
    "gpu_temperature_end_c",
    "gpu_power_limit_w",
]

VRAM_METRICS = [
    "torch_peak_allocated_vram_mb",
    "torch_peak_reserved_vram_mb",
    "nvml_peak_used_vram_mb",
    "peak_vram_mb",
]

CPU_METRICS = [
    "process_cpu_mean_percent",
    "process_cpu_max_percent",
    "process_cpu_user_seconds",
    "process_cpu_system_seconds",
]

RAM_METRICS = [
    "process_rss_start_mb",
    "process_rss_peak_mb",
    "process_rss_end_mb",
    "system_ram_used_start_mb",
    "system_ram_used_peak_mb",
    "system_ram_used_end_mb",
]

THROUGHPUT_METRICS = [
    "images_per_hour",
]

GENERATION_METRICS = list(
    dict.fromkeys(
        TIMING_METRICS
        + ENERGY_METRICS
        + GPU_METRICS
        + VRAM_METRICS
        + CPU_METRICS
        + RAM_METRICS
        + THROUGHPUT_METRICS
    )
)

MODEL_LOAD_METRICS = [
    "model_load_seconds",
    "model_load_monitoring_duration_seconds",
    "model_load_gpu_energy_j",
    "model_load_gpu_energy_wh",
    "model_load_gpu_power_mean_w",
    "model_load_gpu_power_max_w",
    "model_load_gpu_utilization_mean_percent",
    "model_load_gpu_utilization_max_percent",
    "model_load_gpu_temperature_start_c",
    "model_load_gpu_temperature_max_c",
    "model_load_gpu_temperature_end_c",
    "model_load_gpu_power_limit_w",
    "model_load_torch_peak_allocated_vram_mb",
    "model_load_torch_peak_reserved_vram_mb",
    "model_load_nvml_peak_used_vram_mb",
    "model_load_process_cpu_mean_percent",
    "model_load_process_cpu_max_percent",
    "model_load_process_cpu_user_seconds",
    "model_load_process_cpu_system_seconds",
    "model_load_process_rss_start_mb",
    "model_load_process_rss_peak_mb",
    "model_load_process_rss_end_mb",
    "model_load_system_ram_used_start_mb",
    "model_load_system_ram_used_peak_mb",
    "model_load_system_ram_used_end_mb",
]

IDENTITY_COLUMNS = [
    "experiment_id",
    "model_name",
    "model_huggingface_id",
    "quantization",
    "quantized_components",
    "requested_device",
    "actual_device",
    "gpu_name",
    "num_inference_steps",
    "guidance_scale",
    "width",
    "height",
    "resolution",
]

PROMPT_GROUP_COLUMNS = [
    "experiment_id",
    "model_name",
    "model_huggingface_id",
    "quantization",
    "prompt_id",
    "task_id",
    "category",
    "task",
    "variant_type",
    "prompt_style",
    "width",
    "height",
    "num_inference_steps",
    "guidance_scale",
]

SEED_GROUP_COLUMNS = [
    "experiment_id",
    "model_name",
    "quantization",
    "seed",
    "width",
    "height",
    "num_inference_steps",
    "guidance_scale",
]

PRIMARY_METRICS = [
    "generation_total_seconds",
    "inference_seconds",
    "image_save_seconds",
    "gpu_energy_j",
    "gpu_energy_wh",
    "gpu_power_mean_w",
    "gpu_power_max_w",
    "gpu_utilization_mean_percent",
    "gpu_utilization_max_percent",
    "gpu_temperature_max_c",
    "torch_peak_allocated_vram_mb",
    "torch_peak_reserved_vram_mb",
    "nvml_peak_used_vram_mb",
    "process_cpu_mean_percent",
    "process_cpu_max_percent",
    "process_rss_peak_mb",
    "system_ram_used_peak_mb",
    "images_per_hour",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyse SynthSE generation and model-loading resource metrics."
        )
    )
    parser.add_argument(
        "--generation-csv",
        nargs="+",
        required=True,
        help="One or more *_generation_metadata.csv files.",
    )
    parser.add_argument(
        "--model-load-csv",
        nargs="*",
        default=[],
        help="Optional one or more *_model_load_resources.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        help=(
            "Analysis output directory. Default: a sibling directory named "
            "<experiment_id>_resource_analysis."
        ),
    )
    parser.add_argument(
        "--warmup-count",
        type=int,
        default=1,
        help=(
            "Number of initial successful generations per experiment to "
            "classify as warm-up. Default: 1."
        ),
    )
    parser.add_argument(
        "--outlier-iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier used to flag outliers. Default: 1.5.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip PNG plot generation.",
    )
    return parser.parse_args()


def sanitize_filename(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "item"


def read_csv_files(
    paths: Iterable[str],
    source_column: str,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []

    for raw_path in paths:
        path = Path(raw_path)

        if not path.exists():
            raise FileNotFoundError(f"CSV file not found: {path}")

        frame = pd.read_csv(path)
        frame[source_column] = str(path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(
        frames,
        ignore_index=True,
        sort=False,
    )


def coerce_numeric_columns(
    dataframe: pd.DataFrame,
    columns: Iterable[str],
) -> pd.DataFrame:
    result = dataframe.copy()

    for column in columns:
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    return result


def normalize_generation_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    result = dataframe.copy()

    result = coerce_numeric_columns(
        result,
        GENERATION_METRICS
        + [
            "seed",
            "width",
            "height",
            "num_inference_steps",
            "guidance_scale",
            "monitoring_sample_count",
        ],
    )

    if "experiment_id" not in result.columns:
        result["experiment_id"] = "unknown_experiment"

    result["experiment_id"] = (
        result["experiment_id"]
        .fillna("unknown_experiment")
        .astype(str)
    )

    if "status" not in result.columns:
        result["status"] = "generated"

    result["status"] = (
        result["status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    if "timestamp" in result.columns:
        result["_timestamp_parsed"] = pd.to_datetime(
            result["timestamp"],
            errors="coerce",
        )
    else:
        result["_timestamp_parsed"] = pd.NaT

    result["_original_row_order"] = np.arange(
        len(result),
        dtype=int,
    )

    result = result.sort_values(
        [
            "experiment_id",
            "_timestamp_parsed",
            "_original_row_order",
        ],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)

    result["generation_order"] = (
        result.groupby(
            "experiment_id",
            dropna=False,
        ).cumcount()
        + 1
    )

    result["successful_generation_order"] = np.nan

    success_mask = result["status"].eq("generated")

    result.loc[
        success_mask,
        "successful_generation_order",
    ] = (
        result.loc[success_mask]
        .groupby(
            "experiment_id",
            dropna=False,
        )
        .cumcount()
        .add(1)
        .astype(float)
    )

    return result


def normalize_model_load_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe.copy()

    result = coerce_numeric_columns(
        dataframe,
        MODEL_LOAD_METRICS,
    )

    if "experiment_id" not in result.columns:
        result["experiment_id"] = "unknown_experiment"

    result["experiment_id"] = (
        result["experiment_id"]
        .fillna("unknown_experiment")
        .astype(str)
    )

    return result


def available_metrics(
    dataframe: pd.DataFrame,
    candidates: Iterable[str],
) -> list[str]:
    return [
        metric
        for metric in candidates
        if (
            metric in dataframe.columns
            and dataframe[metric].notna().any()
        )
    ]


def safe_scalar(
    series: pd.Series,
    operation: str,
) -> float | None:
    values = pd.to_numeric(
        series,
        errors="coerce",
    ).dropna()

    if values.empty:
        return None

    if operation == "mean":
        return float(values.mean())
    if operation == "median":
        return float(values.median())
    if operation == "sum":
        return float(values.sum())
    if operation == "min":
        return float(values.min())
    if operation == "max":
        return float(values.max())
    if operation == "std":
        return float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if operation == "q1":
        return float(values.quantile(0.25))
    if operation == "q3":
        return float(values.quantile(0.75))
    if operation == "p95":
        return float(values.quantile(0.95))

    raise ValueError(f"Unsupported operation: {operation}")


def first_non_null(
    dataframe: pd.DataFrame,
    column: str,
    default: Any = "",
) -> Any:
    if column not in dataframe.columns:
        return default

    values = dataframe[column].dropna()

    if values.empty:
        return default

    return values.iloc[0]


def descriptive_statistics(
    dataframe: pd.DataFrame,
    metrics: list[str],
    group_columns: list[str] | None = None,
) -> pd.DataFrame:
    group_columns = [
        column
        for column in (group_columns or [])
        if column in dataframe.columns
    ]

    rows: list[dict[str, Any]] = []

    if group_columns:
        grouped = dataframe.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
    else:
        grouped = [((), dataframe)]

    for group_key, group in grouped:
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        group_identity = dict(
            zip(
                group_columns,
                group_key,
                strict=False,
            )
        )

        for metric in metrics:
            if metric not in group.columns:
                continue

            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            mean_value = float(values.mean())
            std_value = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )

            cv_percent = (
                std_value / abs(mean_value) * 100.0
                if mean_value != 0
                else np.nan
            )

            rows.append(
                {
                    **group_identity,
                    "metric": metric,
                    "count": int(values.count()),
                    "mean": mean_value,
                    "std": std_value,
                    "cv_percent": cv_percent,
                    "min": float(values.min()),
                    "q1": float(values.quantile(0.25)),
                    "median": float(values.median()),
                    "q3": float(values.quantile(0.75)),
                    "p95": float(values.quantile(0.95)),
                    "max": float(values.max()),
                    "sum": float(values.sum()),
                }
            )

    return pd.DataFrame(rows)


def aggregate_model_loads(
    model_loads: pd.DataFrame,
) -> pd.DataFrame:
    if model_loads.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for experiment_id, group in model_loads.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        row: dict[str, Any] = {
            "experiment_id": str(experiment_id),
            "model_load_count": int(len(group)),
            "model_load_source": "model_load_csv",
        }

        for identity_column in IDENTITY_COLUMNS:
            if identity_column in group.columns:
                row[identity_column] = first_non_null(
                    group,
                    identity_column,
                    "",
                )

        for metric in available_metrics(
            group,
            MODEL_LOAD_METRICS,
        ):
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{metric}_total"] = float(values.sum())
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_max"] = float(values.max())

        rows.append(row)

    return pd.DataFrame(rows)


def infer_model_loads_from_generation(
    generation_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fallback only when no model-load CSV is supplied.

    A repeated single value is treated as one shared model load.
    Multiple distinct values are treated as one load per generation row.
    """
    if (
        generation_data.empty
        or "model_load_seconds" not in generation_data.columns
    ):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for experiment_id, group in generation_data.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        values = pd.to_numeric(
            group["model_load_seconds"],
            errors="coerce",
        ).dropna()

        if values.empty:
            continue

        unique_rounded = values.round(6).nunique()

        if unique_rounded == 1:
            total_seconds = float(values.iloc[0])
            count = 1
            source = "inferred_single_shared_load_from_generation_csv"
        else:
            total_seconds = float(values.sum())
            count = int(values.count())
            source = "inferred_per_row_loads_from_generation_csv"

        row: dict[str, Any] = {
            "experiment_id": str(experiment_id),
            "model_load_count": count,
            "model_load_source": source,
            "model_load_seconds_total": total_seconds,
            "model_load_seconds_mean": float(values.mean()),
            "model_load_seconds_median": float(values.median()),
            "model_load_seconds_min": float(values.min()),
            "model_load_seconds_max": float(values.max()),
        }

        for identity_column in IDENTITY_COLUMNS:
            if identity_column in group.columns:
                row[identity_column] = first_non_null(
                    group,
                    identity_column,
                    "",
                )

        rows.append(row)

    return pd.DataFrame(rows)


def experiment_summary(
    generation_data: pd.DataFrame,
    model_load_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    load_lookup: dict[str, dict[str, Any]] = {}

    if not model_load_summary.empty:
        for _, row in model_load_summary.iterrows():
            load_lookup[str(row["experiment_id"])] = row.to_dict()

    for experiment_id, group in generation_data.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        experiment_id_text = str(experiment_id)
        success = group[group["status"].eq("generated")]
        failed = group[~group["status"].eq("generated")]

        successful_images = int(len(success))
        total_rows = int(len(group))
        failed_images = int(len(failed))

        row: dict[str, Any] = {
            "experiment_id": experiment_id_text,
            "total_generation_records": total_rows,
            "successful_images": successful_images,
            "failed_images": failed_images,
            "success_rate_percent": (
                successful_images / total_rows * 100.0
                if total_rows
                else np.nan
            ),
        }

        for identity_column in IDENTITY_COLUMNS:
            if identity_column in group.columns:
                row[identity_column] = first_non_null(
                    group,
                    identity_column,
                    "",
                )

        for metric in available_metrics(
            success,
            PRIMARY_METRICS,
        ):
            values = pd.to_numeric(
                success[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(values.median())
            row[f"{metric}_std"] = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric}_min"] = float(values.min())
            row[f"{metric}_p95"] = float(values.quantile(0.95))
            row[f"{metric}_max"] = float(values.max())

        generation_duration_total = safe_scalar(
            success.get(
                "generation_total_seconds",
                pd.Series(dtype=float),
            ),
            "sum",
        )

        generation_energy_j_total = safe_scalar(
            success.get(
                "gpu_energy_j",
                pd.Series(dtype=float),
            ),
            "sum",
        )

        generation_energy_wh_total = safe_scalar(
            success.get(
                "gpu_energy_wh",
                pd.Series(dtype=float),
            ),
            "sum",
        )

        row["generation_duration_seconds_total"] = (
            generation_duration_total
        )
        row["generation_gpu_energy_j_total"] = (
            generation_energy_j_total
        )
        row["generation_gpu_energy_wh_total"] = (
            generation_energy_wh_total
        )

        load_row = load_lookup.get(
            experiment_id_text,
            {},
        )

        row.update(
            {
                key: value
                for key, value in load_row.items()
                if key != "experiment_id"
            }
        )

        model_load_seconds_total = load_row.get(
            "model_load_seconds_total"
        )

        model_load_energy_j_total = load_row.get(
            "model_load_gpu_energy_j_total"
        )

        model_load_energy_wh_total = load_row.get(
            "model_load_gpu_energy_wh_total"
        )

        if (
            successful_images > 0
            and model_load_seconds_total is not None
            and pd.notna(model_load_seconds_total)
        ):
            row[
                "model_load_seconds_amortized_per_image"
            ] = (
                float(model_load_seconds_total)
                / successful_images
            )

        if (
            successful_images > 0
            and model_load_energy_j_total is not None
            and pd.notna(model_load_energy_j_total)
        ):
            row[
                "model_load_gpu_energy_j_amortized_per_image"
            ] = (
                float(model_load_energy_j_total)
                / successful_images
            )

        if (
            generation_duration_total is not None
            and model_load_seconds_total is not None
            and pd.notna(model_load_seconds_total)
        ):
            end_to_end_duration = (
                float(generation_duration_total)
                + float(model_load_seconds_total)
            )

            row[
                "end_to_end_duration_seconds"
            ] = end_to_end_duration

            row[
                "end_to_end_images_per_hour"
            ] = (
                successful_images
                / end_to_end_duration
                * 3600.0
                if (
                    successful_images > 0
                    and end_to_end_duration > 0
                )
                else np.nan
            )

        if (
            generation_energy_j_total is not None
            and model_load_energy_j_total is not None
            and pd.notna(model_load_energy_j_total)
        ):
            end_to_end_energy_j = (
                float(generation_energy_j_total)
                + float(model_load_energy_j_total)
            )

            row[
                "end_to_end_gpu_energy_j"
            ] = end_to_end_energy_j

            row[
                "end_to_end_gpu_energy_j_per_image"
            ] = (
                end_to_end_energy_j
                / successful_images
                if successful_images > 0
                else np.nan
            )

        if (
            generation_energy_wh_total is not None
            and model_load_energy_wh_total is not None
            and pd.notna(model_load_energy_wh_total)
        ):
            end_to_end_energy_wh = (
                float(generation_energy_wh_total)
                + float(model_load_energy_wh_total)
            )

            row[
                "end_to_end_gpu_energy_wh"
            ] = end_to_end_energy_wh

            row[
                "end_to_end_gpu_energy_wh_per_image"
            ] = (
                end_to_end_energy_wh
                / successful_images
                if successful_images > 0
                else np.nan
            )

        rows.append(row)

    return pd.DataFrame(rows)


def grouped_wide_summary(
    dataframe: pd.DataFrame,
    group_columns: list[str],
    metrics: list[str],
) -> pd.DataFrame:
    group_columns = [
        column
        for column in group_columns
        if column in dataframe.columns
    ]

    metrics = available_metrics(
        dataframe,
        metrics,
    )

    if not group_columns or not metrics:
        return pd.DataFrame()

    aggregation: dict[str, list[str]] = {
        metric: [
            "count",
            "mean",
            "std",
            "median",
            "min",
            "max",
        ]
        for metric in metrics
    }

    summary = (
        dataframe.groupby(
            group_columns,
            dropna=False,
            sort=False,
        )
        .agg(aggregation)
        .reset_index()
    )

    summary.columns = [
        column
        if isinstance(column, str)
        else "_".join(
            str(part)
            for part in column
            if str(part)
        )
        for column in summary.columns
    ]

    return summary


def warmup_comparison(
    generation_data: pd.DataFrame,
    metrics: list[str],
    warmup_count: int,
) -> pd.DataFrame:
    if warmup_count <= 0:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    success = generation_data[
        generation_data["status"].eq("generated")
    ].copy()

    if success.empty:
        return pd.DataFrame()

    success["warmup_group"] = np.where(
        success["successful_generation_order"]
        <= warmup_count,
        "warmup",
        "steady_state",
    )

    for experiment_id, group in success.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        warmup = group[
            group["warmup_group"].eq("warmup")
        ]
        steady = group[
            group["warmup_group"].eq("steady_state")
        ]

        for metric in available_metrics(
            group,
            metrics,
        ):
            warmup_values = pd.to_numeric(
                warmup[metric],
                errors="coerce",
            ).dropna()

            steady_values = pd.to_numeric(
                steady[metric],
                errors="coerce",
            ).dropna()

            if warmup_values.empty:
                continue

            warmup_mean = float(
                warmup_values.mean()
            )

            steady_mean = (
                float(steady_values.mean())
                if not steady_values.empty
                else np.nan
            )

            rows.append(
                {
                    "experiment_id": str(experiment_id),
                    "metric": metric,
                    "warmup_count_configured": warmup_count,
                    "warmup_observations": int(
                        warmup_values.count()
                    ),
                    "steady_state_observations": int(
                        steady_values.count()
                    ),
                    "warmup_mean": warmup_mean,
                    "steady_state_mean": steady_mean,
                    "absolute_difference": (
                        warmup_mean - steady_mean
                        if pd.notna(steady_mean)
                        else np.nan
                    ),
                    "warmup_to_steady_ratio": (
                        warmup_mean / steady_mean
                        if (
                            pd.notna(steady_mean)
                            and steady_mean != 0
                        )
                        else np.nan
                    ),
                    "percent_difference_from_steady": (
                        (warmup_mean - steady_mean)
                        / abs(steady_mean)
                        * 100.0
                        if (
                            pd.notna(steady_mean)
                            and steady_mean != 0
                        )
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(rows)


def pairwise_correlations(
    dataframe: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    available = available_metrics(
        dataframe,
        metrics,
    )

    rows: list[dict[str, Any]] = []

    for experiment_id, group in dataframe.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        for index, metric_a in enumerate(available):
            for metric_b in available[index + 1:]:
                pair = group[
                    [metric_a, metric_b]
                ].apply(
                    pd.to_numeric,
                    errors="coerce",
                ).dropna()

                if len(pair) < 3:
                    continue

                if (
                    pair[metric_a].nunique(dropna=True) < 2
                    or pair[metric_b].nunique(dropna=True) < 2
                ):
                    continue

                pearson = pair[metric_a].corr(
                    pair[metric_b],
                    method="pearson",
                )

                spearman = pair[metric_a].corr(
                    pair[metric_b],
                    method="spearman",
                )

                rows.append(
                    {
                        "experiment_id": str(experiment_id),
                        "metric_a": metric_a,
                        "metric_b": metric_b,
                        "observations": int(len(pair)),
                        "pearson_correlation": pearson,
                        "spearman_correlation": spearman,
                    }
                )

    return pd.DataFrame(rows)


def detect_outliers(
    dataframe: pd.DataFrame,
    metrics: list[str],
    iqr_multiplier: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for experiment_id, group in dataframe.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        for metric in available_metrics(
            group,
            metrics,
        ):
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            )

            valid = values.dropna()

            if len(valid) < 4:
                continue

            q1 = float(valid.quantile(0.25))
            q3 = float(valid.quantile(0.75))
            iqr = q3 - q1
            lower = q1 - iqr_multiplier * iqr
            upper = q3 + iqr_multiplier * iqr

            mask = values.lt(lower) | values.gt(upper)

            for row_index in group.index[mask.fillna(False)]:
                source_row = dataframe.loc[row_index]

                rows.append(
                    {
                        "experiment_id": str(experiment_id),
                        "metric": metric,
                        "value": source_row.get(metric),
                        "lower_bound": lower,
                        "upper_bound": upper,
                        "iqr_multiplier": iqr_multiplier,
                        "generation_order": source_row.get(
                            "generation_order"
                        ),
                        "image_id": source_row.get(
                            "image_id",
                            "",
                        ),
                        "prompt_id": source_row.get(
                            "prompt_id",
                            "",
                        ),
                        "seed": source_row.get(
                            "seed",
                            "",
                        ),
                        "status": source_row.get(
                            "status",
                            "",
                        ),
                    }
                )

    return pd.DataFrame(rows)


def data_quality_report(
    generation_data: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in metrics:
        if metric not in generation_data.columns:
            rows.append(
                {
                    "metric": metric,
                    "column_present": False,
                    "rows": len(generation_data),
                    "non_null_count": 0,
                    "missing_count": len(generation_data),
                    "missing_percent": 100.0,
                    "zero_count": 0,
                    "negative_count": 0,
                }
            )
            continue

        values = pd.to_numeric(
            generation_data[metric],
            errors="coerce",
        )

        non_null_count = int(
            values.notna().sum()
        )

        missing_count = int(
            values.isna().sum()
        )

        rows.append(
            {
                "metric": metric,
                "column_present": True,
                "rows": int(len(generation_data)),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_percent": (
                    missing_count
                    / len(generation_data)
                    * 100.0
                    if len(generation_data)
                    else np.nan
                ),
                "zero_count": int(
                    values.eq(0).sum()
                ),
                "negative_count": int(
                    values.lt(0).sum()
                ),
            }
        )

    duplicate_image_ids = 0

    if "image_id" in generation_data.columns:
        duplicate_image_ids = int(
            generation_data["image_id"]
            .astype(str)
            .duplicated()
            .sum()
        )

    rows.append(
        {
            "metric": "__duplicate_image_ids__",
            "column_present": (
                "image_id" in generation_data.columns
            ),
            "rows": int(len(generation_data)),
            "non_null_count": np.nan,
            "missing_count": np.nan,
            "missing_percent": np.nan,
            "zero_count": duplicate_image_ids,
            "negative_count": np.nan,
        }
    )

    return pd.DataFrame(rows)


def save_dataframe(
    dataframe: pd.DataFrame,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
    )


def plot_metric_over_order(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if (
        metric not in dataframe.columns
        or not dataframe[metric].notna().any()
    ):
        return

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    for experiment_id, group in dataframe.groupby(
        "experiment_id",
        dropna=False,
        sort=False,
    ):
        axis.plot(
            group["generation_order"],
            group[metric],
            marker="o",
            label=str(experiment_id),
        )

    axis.set_xlabel("Generation order")
    axis.set_ylabel(metric)
    axis.set_title(
        f"{metric} by generation order"
    )

    if dataframe["experiment_id"].nunique() > 1:
        axis.legend()

    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_histogram(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if (
        metric not in dataframe.columns
        or not dataframe[metric].notna().any()
    ):
        return

    values = pd.to_numeric(
        dataframe[metric],
        errors="coerce",
    ).dropna()

    if values.empty:
        return

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.hist(
        values,
        bins="auto",
    )

    axis.set_xlabel(metric)
    axis.set_ylabel("Frequency")
    axis.set_title(
        f"Distribution of {metric}"
    )
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_duration_energy_scatter(
    dataframe: pd.DataFrame,
    output_path: Path,
) -> None:
    required = {
        "generation_total_seconds",
        "gpu_energy_j",
    }

    if not required.issubset(dataframe.columns):
        return

    plot_data = dataframe[
        [
            "generation_total_seconds",
            "gpu_energy_j",
        ]
    ].apply(
        pd.to_numeric,
        errors="coerce",
    ).dropna()

    if plot_data.empty:
        return

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    axis.scatter(
        plot_data["generation_total_seconds"],
        plot_data["gpu_energy_j"],
    )

    axis.set_xlabel(
        "Generation total seconds"
    )
    axis.set_ylabel(
        "GPU energy (J)"
    )
    axis.set_title(
        "Generation duration vs GPU energy"
    )
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_metric_by_prompt_style(
    dataframe: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    if (
        "prompt_style" not in dataframe.columns
        or metric not in dataframe.columns
    ):
        return

    styles: list[str] = []
    values: list[np.ndarray] = []

    for style, group in dataframe.groupby(
        "prompt_style",
        dropna=False,
        sort=False,
    ):
        metric_values = pd.to_numeric(
            group[metric],
            errors="coerce",
        ).dropna()

        if metric_values.empty:
            continue

        styles.append(str(style))
        values.append(
            metric_values.to_numpy()
        )

    if len(values) < 2:
        return

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.boxplot(
        values,
        tick_labels=styles,
    )

    axis.set_xlabel("Prompt style")
    axis.set_ylabel(metric)
    axis.set_title(
        f"{metric} by prompt style"
    )
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def plot_energy_breakdown(
    experiment_summary_data: pd.DataFrame,
    output_path: Path,
) -> None:
    required = {
        "experiment_id",
        "generation_gpu_energy_j_total",
        "model_load_gpu_energy_j_total",
    }

    if not required.issubset(
        experiment_summary_data.columns
    ):
        return

    plot_data = experiment_summary_data[
        [
            "experiment_id",
            "generation_gpu_energy_j_total",
            "model_load_gpu_energy_j_total",
        ]
    ].copy()

    plot_data[
        "generation_gpu_energy_j_total"
    ] = pd.to_numeric(
        plot_data[
            "generation_gpu_energy_j_total"
        ],
        errors="coerce",
    ).fillna(0)

    plot_data[
        "model_load_gpu_energy_j_total"
    ] = pd.to_numeric(
        plot_data[
            "model_load_gpu_energy_j_total"
        ],
        errors="coerce",
    ).fillna(0)

    positions = np.arange(
        len(plot_data)
    )

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.bar(
        positions,
        plot_data[
            "generation_gpu_energy_j_total"
        ],
        label="Generation",
    )

    axis.bar(
        positions,
        plot_data[
            "model_load_gpu_energy_j_total"
        ],
        bottom=plot_data[
            "generation_gpu_energy_j_total"
        ],
        label="Model load",
    )

    axis.set_xticks(
        positions,
        plot_data["experiment_id"].astype(str),
        rotation=30,
        ha="right",
    )
    axis.set_ylabel("GPU energy (J)")
    axis.set_title(
        "GPU energy breakdown"
    )
    axis.legend()
    axis.grid(True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=160,
    )
    plt.close(figure)


def create_plots(
    generation_data: pd.DataFrame,
    experiment_summary_data: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = generation_data[
        generation_data["status"].eq("generated")
    ].copy()

    for metric in [
        "generation_total_seconds",
        "inference_seconds",
        "gpu_energy_j",
        "gpu_power_mean_w",
        "gpu_utilization_mean_percent",
        "gpu_temperature_max_c",
        "nvml_peak_used_vram_mb",
        "process_cpu_mean_percent",
        "process_rss_peak_mb",
    ]:
        plot_metric_over_order(
            successful,
            metric,
            plots_dir
            / f"{sanitize_filename(metric)}_over_order.png",
        )

    for metric in [
        "generation_total_seconds",
        "gpu_energy_j",
        "gpu_power_mean_w",
        "gpu_utilization_mean_percent",
        "nvml_peak_used_vram_mb",
        "process_cpu_mean_percent",
        "process_rss_peak_mb",
    ]:
        plot_histogram(
            successful,
            metric,
            plots_dir
            / f"{sanitize_filename(metric)}_histogram.png",
        )

    plot_duration_energy_scatter(
        successful,
        plots_dir
        / "generation_duration_vs_gpu_energy.png",
    )

    for metric in [
        "generation_total_seconds",
        "gpu_energy_j",
        "gpu_power_mean_w",
        "gpu_utilization_mean_percent",
    ]:
        plot_metric_by_prompt_style(
            successful,
            metric,
            plots_dir
            / (
                f"{sanitize_filename(metric)}"
                "_by_prompt_style.png"
            ),
        )

    plot_energy_breakdown(
        experiment_summary_data,
        plots_dir
        / "gpu_energy_breakdown.png",
    )


def format_number(
    value: Any,
    digits: int = 4,
) -> str:
    if value is None:
        return "n/a"

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)

    if not math.isfinite(numeric):
        return "n/a"

    return f"{numeric:.{digits}f}"


def write_markdown_report(
    experiment_summary_data: pd.DataFrame,
    warmup_data: pd.DataFrame,
    quality_data: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "# SynthSE resource metrics analysis",
        "",
    ]

    for _, row in experiment_summary_data.iterrows():
        experiment_id = row.get(
            "experiment_id",
            "unknown",
        )

        lines.extend(
            [
                f"## Experiment {experiment_id}",
                "",
                f"- Model: `{row.get('model_huggingface_id', '')}`",
                f"- Quantization: `{row.get('quantization', '')}`",
                f"- Device: `{row.get('actual_device', '')}`",
                f"- GPU: `{row.get('gpu_name', '')}`",
                (
                    "- Successful images: "
                    f"{int(row.get('successful_images', 0))}"
                ),
                (
                    "- Failed images: "
                    f"{int(row.get('failed_images', 0))}"
                ),
                (
                    "- Mean generation time: "
                    f"{format_number(row.get('generation_total_seconds_mean'))} s"
                ),
                (
                    "- Median generation time: "
                    f"{format_number(row.get('generation_total_seconds_median'))} s"
                ),
                (
                    "- Mean GPU energy per generated image: "
                    f"{format_number(row.get('gpu_energy_j_mean'))} J"
                ),
                (
                    "- Total generation GPU energy: "
                    f"{format_number(row.get('generation_gpu_energy_j_total'))} J"
                ),
                (
                    "- Total model-load GPU energy: "
                    f"{format_number(row.get('model_load_gpu_energy_j_total'))} J"
                ),
                (
                    "- End-to-end GPU energy: "
                    f"{format_number(row.get('end_to_end_gpu_energy_j'))} J"
                ),
                (
                    "- End-to-end GPU energy per image: "
                    f"{format_number(row.get('end_to_end_gpu_energy_j_per_image'))} J"
                ),
                (
                    "- Mean GPU power: "
                    f"{format_number(row.get('gpu_power_mean_w_mean'))} W"
                ),
                (
                    "- Mean GPU utilization: "
                    f"{format_number(row.get('gpu_utilization_mean_percent_mean'))}%"
                ),
                (
                    "- Peak NVML VRAM: "
                    f"{format_number(row.get('nvml_peak_used_vram_mb_max'))} MiB"
                ),
                (
                    "- End-to-end throughput: "
                    f"{format_number(row.get('end_to_end_images_per_hour'))} images/hour"
                ),
                "",
            ]
        )

    if not warmup_data.empty:
        lines.extend(
            [
                "## Warm-up observations",
                "",
            ]
        )

        selected = warmup_data[
            warmup_data["metric"].isin(
                [
                    "generation_total_seconds",
                    "gpu_energy_j",
                ]
            )
        ]

        for _, row in selected.iterrows():
            lines.append(
                "- "
                f"{row['experiment_id']} / {row['metric']}: "
                f"warm-up mean {format_number(row['warmup_mean'])}, "
                f"steady-state mean {format_number(row['steady_state_mean'])}, "
                f"difference {format_number(row['percent_difference_from_steady'])}%."
            )

        lines.append("")

    missing = quality_data[
        quality_data["missing_count"].fillna(0).gt(0)
    ]

    if not missing.empty:
        lines.extend(
            [
                "## Data quality notes",
                "",
            ]
        )

        for _, row in missing.iterrows():
            lines.append(
                "- "
                f"`{row['metric']}`: "
                f"{int(row['missing_count'])} missing values "
                f"({format_number(row['missing_percent'], 2)}%)."
            )

        lines.append("")

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def resolve_output_dir(
    args: argparse.Namespace,
    generation_data: pd.DataFrame,
) -> Path:
    if args.output_dir:
        return Path(args.output_dir)

    if generation_data["experiment_id"].nunique() == 1:
        experiment_id = sanitize_filename(
            generation_data["experiment_id"].iloc[0]
        )
    else:
        experiment_id = "combined"

    first_generation_path = Path(
        args.generation_csv[0]
    )

    return (
        first_generation_path.parent
        / f"{experiment_id}_resource_analysis"
    )


def main() -> None:
    args = parse_args()

    if args.warmup_count < 0:
        raise ValueError(
            "--warmup-count must be greater than or equal to zero."
        )

    if args.outlier_iqr_multiplier <= 0:
        raise ValueError(
            "--outlier-iqr-multiplier must be greater than zero."
        )

    generation_data = read_csv_files(
        args.generation_csv,
        "_generation_source_file",
    )

    generation_data = normalize_generation_data(
        generation_data
    )

    model_load_data = read_csv_files(
        args.model_load_csv,
        "_model_load_source_file",
    )

    model_load_data = normalize_model_load_data(
        model_load_data
    )

    output_dir = resolve_output_dir(
        args,
        generation_data,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = generation_data[
        generation_data["status"].eq("generated")
    ].copy()

    generation_metrics = available_metrics(
        generation_data,
        GENERATION_METRICS,
    )

    generation_statistics = descriptive_statistics(
        successful,
        generation_metrics,
        group_columns=["experiment_id"],
    )

    prompt_summary = grouped_wide_summary(
        successful,
        PROMPT_GROUP_COLUMNS,
        PRIMARY_METRICS,
    )

    seed_summary = grouped_wide_summary(
        successful,
        SEED_GROUP_COLUMNS,
        PRIMARY_METRICS,
    )

    if model_load_data.empty:
        model_load_summary = (
            infer_model_loads_from_generation(
                generation_data
            )
        )
    else:
        model_load_summary = aggregate_model_loads(
            model_load_data
        )

    experiment_summary_data = experiment_summary(
        generation_data,
        model_load_summary,
    )

    warmup_data = warmup_comparison(
        generation_data,
        PRIMARY_METRICS,
        warmup_count=args.warmup_count,
    )

    correlation_data = pairwise_correlations(
        successful,
        PRIMARY_METRICS,
    )

    outlier_data = detect_outliers(
        successful,
        PRIMARY_METRICS,
        iqr_multiplier=args.outlier_iqr_multiplier,
    )

    quality_data = data_quality_report(
        generation_data,
        GENERATION_METRICS,
    )

    failure_data = generation_data[
        ~generation_data["status"].eq("generated")
    ].copy()

    save_dataframe(
        generation_data.drop(
            columns=[
                "_timestamp_parsed",
                "_original_row_order",
            ],
            errors="ignore",
        ),
        output_dir / "normalized_generation_metadata.csv",
    )

    save_dataframe(
        experiment_summary_data,
        output_dir / "experiment_summary.csv",
    )

    save_dataframe(
        generation_statistics,
        output_dir / "generation_metric_statistics.csv",
    )

    save_dataframe(
        prompt_summary,
        output_dir / "metrics_by_prompt.csv",
    )

    save_dataframe(
        seed_summary,
        output_dir / "metrics_by_seed.csv",
    )

    save_dataframe(
        model_load_summary,
        output_dir / "model_load_summary.csv",
    )

    save_dataframe(
        warmup_data,
        output_dir / "warmup_comparison.csv",
    )

    save_dataframe(
        correlation_data,
        output_dir / "metric_correlations.csv",
    )

    save_dataframe(
        outlier_data,
        output_dir / "metric_outliers.csv",
    )

    save_dataframe(
        quality_data,
        output_dir / "data_quality_report.csv",
    )

    save_dataframe(
        failure_data.drop(
            columns=[
                "_timestamp_parsed",
                "_original_row_order",
            ],
            errors="ignore",
        ),
        output_dir / "failed_generations.csv",
    )

    if not model_load_data.empty:
        save_dataframe(
            model_load_data,
            output_dir / "normalized_model_load_resources.csv",
        )

    write_markdown_report(
        experiment_summary_data,
        warmup_data,
        quality_data,
        output_dir / "analysis_report.md",
    )

    if not args.no_plots:
        create_plots(
            generation_data,
            experiment_summary_data,
            output_dir / "plots",
        )

    print("SynthSE metrics analysis completed.")
    print(f"Generation rows: {len(generation_data)}")
    print(f"Successful generations: {len(successful)}")
    print(f"Failed generations: {len(failure_data)}")
    print(f"Experiments: {generation_data['experiment_id'].nunique()}")
    print(f"Output directory: {output_dir}")
    print(
        "Main summary: "
        f"{output_dir / 'experiment_summary.csv'}"
    )
    print(
        "Markdown report: "
        f"{output_dir / 'analysis_report.md'}"
    )


if __name__ == "__main__":
    main()