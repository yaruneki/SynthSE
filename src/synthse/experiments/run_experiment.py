from __future__ import annotations

import argparse
import gc
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from synthse.config.config_loader import load_config
from synthse.generation.generator import generate_image
from synthse.generation.model_loader import load_text_to_image_pipeline
from synthse.tracking.metadata_logger import save_metadata
from synthse.tracking.resource_monitor import ResourceMonitor


REQUIRED_PROMPT_COLUMNS = {
    "prompt_id",
    "category",
    "task",
    "prompt_style",
    "prompt",
    "variant_type",
}


def load_prompts(prompts_path: str) -> pd.DataFrame:
    """
    Load prompts from a CSV file and validate the required columns.
    """
    path = Path(prompts_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_path}"
        )

    prompts = pd.read_csv(path)

    missing_columns = REQUIRED_PROMPT_COLUMNS - set(prompts.columns)

    if missing_columns:
        raise ValueError(
            "Missing columns in prompts CSV: "
            f"{sorted(missing_columns)}"
        )

    if prompts.empty:
        raise ValueError(
            f"The prompts CSV is empty: {prompts_path}"
        )

    return prompts


def as_list(value: Any) -> list[Any]:
    """
    Normalize a scalar value or a sequence into a list.
    """
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def build_resolution_pairs(
    config: dict[str, Any],
) -> list[tuple[int, int]]:
    """
    Build all width-height combinations configured in the YAML.

    Example:
        width: [256, 512]
        height: [256, 512]

    produces:
        [(256, 256), (256, 512), (512, 256), (512, 512)]
    """
    widths = [
        int(width)
        for width in as_list(config["generation"]["width"])
    ]

    heights = [
        int(height)
        for height in as_list(config["generation"]["height"])
    ]

    return [
        (width, height)
        for width in widths
        for height in heights
    ]


def validate_generation_config(
    config: dict[str, Any],
) -> None:
    """
    Validate the required experiment and generation configuration.
    """
    required_sections = {
        "project",
        "model",
        "generation",
        "paths",
    }

    missing_sections = required_sections - set(config)

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    required_model_keys = {
        "name",
        "huggingface_id",
        "device",
    }

    missing_model_keys = (
        required_model_keys
        - set(config["model"])
    )

    if missing_model_keys:
        raise ValueError(
            "Missing model config keys: "
            f"{sorted(missing_model_keys)}"
        )

    required_generation_keys = {
        "num_inference_steps",
        "guidance_scale",
        "width",
        "height",
        "seeds",
    }

    missing_generation_keys = (
        required_generation_keys
        - set(config["generation"])
    )

    if missing_generation_keys:
        raise ValueError(
            "Missing generation config keys: "
            f"{sorted(missing_generation_keys)}"
        )

    required_path_keys = {
        "prompts_file",
        "output_images_dir",
        "output_logs_dir",
    }

    missing_path_keys = (
        required_path_keys
        - set(config["paths"])
    )

    if missing_path_keys:
        raise ValueError(
            "Missing path config keys: "
            f"{sorted(missing_path_keys)}"
        )

    resolution_pairs = build_resolution_pairs(config)

    seeds = [
        int(seed)
        for seed in as_list(config["generation"]["seeds"])
    ]

    if not resolution_pairs:
        raise ValueError(
            "At least one resolution pair is required."
        )

    if not seeds:
        raise ValueError(
            "generation.seeds must contain at least one seed."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "generation.seeds contains duplicate values."
        )

    for width, height in resolution_pairs:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Image width and height must be greater than zero. "
                f"Invalid resolution: {width}x{height}"
            )

        if width % 8 != 0 or height % 8 != 0:
            raise ValueError(
                "Image width and height must be divisible by 8. "
                f"Invalid resolution: {width}x{height}"
            )

    num_inference_steps = int(
        config["generation"]["num_inference_steps"]
    )

    guidance_scale = float(
        config["generation"]["guidance_scale"]
    )

    if num_inference_steps <= 0:
        raise ValueError(
            "generation.num_inference_steps must be greater than zero."
        )

    if guidance_scale < 0:
        raise ValueError(
            "generation.guidance_scale must be greater than or equal to zero."
        )


def get_gpu_name(device: str) -> str:
    """
    Return the GPU name when CUDA is active.
    """
    if (
        str(device).lower().startswith("cuda")
        and torch.cuda.is_available()
    ):
        return torch.cuda.get_device_name(0)

    return ""


def get_row_value(
    row: pd.Series,
    column: str,
    default: Any = "",
) -> Any:
    """
    Read an optional pandas value, replacing missing and NaN values.
    """
    if column not in row.index:
        return default

    value = row[column]

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return value


def sanitize_filename_value(value: Any) -> str:
    """
    Convert a value into a filesystem-safe filename component.
    """
    text = str(value).strip()

    sanitized = "".join(
        character
        if character.isalnum() or character in {"-", "_"}
        else "_"
        for character in text
    )

    sanitized = sanitized.strip("_")

    return sanitized or "prompt"


def build_image_filename(
    prompt_id: Any,
    width: int,
    height: int,
    seed: int,
) -> str:
    """
    Build a deterministic filename for a generated image.
    """
    safe_prompt_id = sanitize_filename_value(prompt_id)

    return (
        f"{safe_prompt_id}_"
        f"w{int(width)}_h{int(height)}_seed{int(seed)}.png"
    )


def normalize_quantization(value: Any) -> str:
    """
    Normalize the configured quantization mode.
    """
    if value is None:
        return "none"

    normalized = str(value).strip().lower()

    if normalized in {"", "none", "null", "false", "no", "off"}:
        return "none"

    return normalized


def normalize_quantized_components(
    components: Any,
) -> list[str]:
    """
    Normalize configured quantized pipeline components.
    """
    if components is None:
        return []

    if isinstance(components, str):
        values = [
            component.strip().lower()
            for component in components.split(",")
            if component.strip()
        ]
    elif isinstance(components, (list, tuple, set)):
        values = [
            str(component).strip().lower()
            for component in components
            if str(component).strip()
        ]
    else:
        raise ValueError(
            "model.quantized_components must be a string "
            "or a list of strings."
        )

    return list(dict.fromkeys(values))


def build_base_record(
    config: dict[str, Any],
    row: pd.Series,
    seed: int,
    experiment_id: str,
    image_id: str,
    actual_device: str,
    width: int,
    height: int,
    model_load_seconds: float,
) -> dict[str, Any]:
    """
    Build base metadata shared by successful and failed generations.
    """
    model_config = config["model"]
    generation_config = config["generation"]

    quantization = normalize_quantization(
        model_config.get("quantization")
    )

    quantized_components = normalize_quantized_components(
        model_config.get("quantized_components")
    )

    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prompt_id": get_row_value(
            row,
            "prompt_id",
            "",
        ),
        "task_id": get_row_value(
            row,
            "task_id",
            "",
        ),
        "category": get_row_value(
            row,
            "category",
            "",
        ),
        "task": get_row_value(
            row,
            "task",
            "",
        ),
        "variant_type": get_row_value(
            row,
            "variant_type",
            "",
        ),
        "prompt_style": get_row_value(
            row,
            "prompt_style",
            "",
        ),
        "prompt": str(
            get_row_value(
                row,
                "prompt",
                "",
            )
        ),
        "seed": int(seed),
        "image_id": image_id,
        "model_name": model_config["name"],
        "model_huggingface_id": model_config["huggingface_id"],
        "quantization": quantization,
        "quantized_components": ",".join(
            quantized_components
        ),
        "requested_device": model_config["device"],
        "actual_device": actual_device,
        "gpu_name": get_gpu_name(actual_device),
        "num_inference_steps": int(
            generation_config["num_inference_steps"]
        ),
        "guidance_scale": float(
            generation_config["guidance_scale"]
        ),
        "width": int(width),
        "height": int(height),
        "resolution": f"{int(width)}x{int(height)}",
        "model_load_seconds": round(
            float(model_load_seconds),
            6,
        ),
    }


def clear_runtime_memory(
    actual_device: str,
    quantization: str | None = None,
) -> None:
    """
    Release Python and CUDA caches between generations.

    CUDA cache cleanup is skipped for TorchAO INT8 runs because it may
    trigger CUDA errors after otherwise successful generations.
    """
    gc.collect()

    normalized_quantization = normalize_quantization(
        quantization
    )

    if (
        not str(actual_device).lower().startswith("cuda")
        or not torch.cuda.is_available()
    ):
        return

    if normalized_quantization == "int8":
        return

    try:
        torch.cuda.synchronize()
    except Exception:
        pass

    try:
        torch.cuda.empty_cache()
    except Exception as error:
        print(
            "Warning: torch.cuda.empty_cache() failed during cleanup. "
            f"Reason: {error}"
        )

    try:
        torch.cuda.ipc_collect()
    except Exception as error:
        print(
            "Warning: torch.cuda.ipc_collect() failed during cleanup. "
            f"Reason: {error}"
        )


def build_failed_generation_metrics(
    image_path: Path,
    error: Exception,
) -> dict[str, Any]:
    """
    Build a complete metadata block for a failed generation.
    """
    return {
        "image_path": str(image_path),
        "execution_time_seconds": None,
        "inference_seconds": None,
        "image_save_seconds": None,
        "generation_total_seconds": None,
        "monitoring_enabled": None,
        "monitoring_phase": "generation",
        "monitoring_duration_seconds": None,
        "monitoring_sample_count": 0,
        "gpu_energy_j": None,
        "gpu_energy_wh": None,
        "gpu_energy_measurement_method": "unavailable",
        "gpu_power_mean_w": None,
        "gpu_power_max_w": None,
        "gpu_utilization_mean_percent": None,
        "gpu_utilization_max_percent": None,
        "gpu_temperature_start_c": None,
        "gpu_temperature_max_c": None,
        "gpu_temperature_end_c": None,
        "gpu_power_limit_w": None,
        "torch_peak_allocated_vram_mb": None,
        "torch_peak_reserved_vram_mb": None,
        "nvml_peak_used_vram_mb": None,
        "process_cpu_mean_percent": None,
        "process_cpu_max_percent": None,
        "process_cpu_user_seconds": None,
        "process_cpu_system_seconds": None,
        "process_rss_start_mb": None,
        "process_rss_peak_mb": None,
        "process_rss_end_mb": None,
        "system_ram_used_start_mb": None,
        "system_ram_used_peak_mb": None,
        "system_ram_used_end_mb": None,
        "joules_per_image": None,
        "wh_per_image": None,
        "images_per_hour": None,
        "peak_vram_mb": None,
        "status": "failed",
        "error_message": (
            f"{type(error).__name__}: {error}"
        ),
    }


def resolve_experiment_id(
    config: dict[str, Any],
) -> str:
    """
    Use run.experiment_id when supplied, otherwise create a timestamp ID.
    """
    configured_id = config.get(
        "run",
        {},
    ).get(
        "experiment_id"
    )

    if configured_id:
        return str(configured_id)

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a SynthSE experiment."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)
    validate_generation_config(config)

    prompts = load_prompts(
        config["paths"]["prompts_file"]
    )

    max_prompts = config.get(
        "run",
        {},
    ).get(
        "max_prompts"
    )

    if max_prompts is not None:
        max_prompts = int(max_prompts)

        if max_prompts <= 0:
            raise ValueError(
                "run.max_prompts must be greater than zero."
            )

        prompts = prompts.head(
            max_prompts
        )

    experiment_id = resolve_experiment_id(
        config
    )

    seeds = [
        int(seed)
        for seed in as_list(
            config["generation"]["seeds"]
        )
    ]

    resolution_pairs = build_resolution_pairs(
        config
    )

    num_inference_steps = int(
        config["generation"]["num_inference_steps"]
    )

    guidance_scale = float(
        config["generation"]["guidance_scale"]
    )

    quantization = normalize_quantization(
        config["model"].get("quantization")
    )

    quantized_components = normalize_quantized_components(
        config["model"].get(
            "quantized_components"
        )
    )

    output_images_dir = (
        Path(config["paths"]["output_images_dir"])
        / experiment_id
    )

    output_logs_dir = Path(
        config["paths"]["output_logs_dir"]
    )

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_generations = (
        len(prompts)
        * len(seeds)
        * len(resolution_pairs)
    )

    print("Configuration loaded successfully.")
    print(f"Project: {config['project']['name']}")
    print(
        "Model: "
        f"{config['model']['huggingface_id']}"
    )
    print(f"Quantization: {quantization}")
    print(
        "Quantized components: "
        f"{quantized_components or 'none'}"
    )
    print(
        "Requested device: "
        f"{config['model']['device']}"
    )
    print(f"Prompts loaded: {len(prompts)}")
    print(f"Seeds: {seeds}")
    print(
        f"Resolution pairs: {resolution_pairs}"
    )
    print(
        "Total planned generations: "
        f"{total_generations}"
    )
    print(f"Experiment ID: {experiment_id}")

    print("\nPrompt preview:")
    print(
        prompts[
            ["prompt_id", "task", "prompt"]
        ]
        .head()
        .to_string(index=False)
    )

    monitoring_config = config.get(
        "monitoring",
        {},
    )

    monitoring_enabled = bool(
        monitoring_config.get(
            "enabled",
            False,
        )
    )

    monitor_model_loading = bool(
        monitoring_config.get(
            "monitor_model_loading",
            True,
        )
    )

    model_load_monitor: ResourceMonitor | None = None

    if monitor_model_loading:
        model_load_monitor = ResourceMonitor(
            gpu_index=int(
                monitoring_config.get(
                    "gpu_index",
                    0,
                )
            ),
            sample_interval_seconds=float(
                monitoring_config.get(
                    "sample_interval_seconds",
                    0.2,
                )
            ),
            enabled=monitoring_enabled,
        )

        model_load_monitor.start(
            phase="model_load"
        )

    model_load_start = time.perf_counter()

    pipeline, actual_device = load_text_to_image_pipeline(
        model_id=config["model"]["huggingface_id"],
        device=config["model"]["device"],
        quantization=quantization,
        quantized_components=quantized_components,
    )

    if (
        str(actual_device).lower().startswith("cuda")
        and torch.cuda.is_available()
    ):
        torch.cuda.synchronize()

    model_load_seconds = (
        time.perf_counter()
        - model_load_start
    )

    model_load_resource_metrics: dict[str, Any] = {}

    if model_load_monitor is not None:
        model_load_resource_metrics = (
            model_load_monitor.stop()
        )

    print(f"\nActual device: {actual_device}")

    gpu_name = get_gpu_name(actual_device)

    if gpu_name:
        print(f"GPU: {gpu_name}")

    print(
        "Model load time: "
        f"{model_load_seconds:.3f} seconds"
    )

    metadata_records: list[
        dict[str, Any]
    ] = []

    with tqdm(
        total=total_generations,
        desc="Generating images",
    ) as progress_bar:
        for _, row in prompts.iterrows():
            # Fix for the previous NameError:
            # define the exact prompt before generate_image().
            actual_prompt = str(
                get_row_value(
                    row,
                    "prompt",
                    "",
                )
            )

            prompt_id = get_row_value(
                row,
                "prompt_id",
                "",
            )

            for width, height in resolution_pairs:
                for seed in seeds:
                    image_filename = (
                        build_image_filename(
                            prompt_id=prompt_id,
                            width=width,
                            height=height,
                            seed=seed,
                        )
                    )

                    image_id = (
                        image_filename.removesuffix(
                            ".png"
                        )
                    )

                    image_path = (
                        output_images_dir
                        / image_filename
                    )

                    base_record = build_base_record(
                        config=config,
                        row=row,
                        seed=seed,
                        experiment_id=experiment_id,
                        image_id=image_id,
                        actual_device=actual_device,
                        width=width,
                        height=height,
                        model_load_seconds=(
                            model_load_seconds
                        ),
                    )

                    try:
                        generation_result = generate_image(
                            pipeline=pipeline,
                            prompt=actual_prompt,
                            seed=seed,
                            output_path=str(
                                image_path
                            ),
                            num_inference_steps=(
                                num_inference_steps
                            ),
                            guidance_scale=(
                                guidance_scale
                            ),
                            width=width,
                            height=height,
                            device=actual_device,
                            monitoring_config=(
                                monitoring_config
                            ),
                        )

                        record = {
                            **base_record,
                            **generation_result,
                        }

                    except Exception as error:
                        tqdm.write(
                            "Generation failed for "
                            f"{image_id}: "
                            f"{type(error).__name__}: "
                            f"{error}"
                        )

                        record = {
                            **base_record,
                            **build_failed_generation_metrics(
                                image_path=image_path,
                                error=error,
                            ),
                        }

                    metadata_records.append(
                        record
                    )

                    progress_bar.update(1)

                    clear_runtime_memory(
                        actual_device=actual_device,
                        quantization=quantization,
                    )

    metadata_path = (
        output_logs_dir
        / (
            f"{experiment_id}_"
            "generation_metadata.csv"
        )
    )

    save_metadata(
        metadata_records,
        str(metadata_path),
    )

    if model_load_resource_metrics:
        model_load_record = {
            "experiment_id": experiment_id,
            "model_name": config["model"]["name"],
            "model_huggingface_id": (
                config["model"]["huggingface_id"]
            ),
            "quantization": quantization,
            "quantized_components": ",".join(
                quantized_components
            ),
            "requested_device": (
                config["model"]["device"]
            ),
            "actual_device": actual_device,
            "gpu_name": get_gpu_name(
                actual_device
            ),
            "model_load_seconds": round(
                model_load_seconds,
                6,
            ),
            **{
                f"model_load_{key}": value
                for key, value
                in model_load_resource_metrics.items()
            },
        }

        model_load_path = (
            output_logs_dir
            / (
                f"{experiment_id}_"
                "model_load_resources.csv"
            )
        )

        save_metadata(
            [model_load_record],
            str(model_load_path),
        )

        print(
            "Model-load resources saved to: "
            f"{model_load_path}"
        )

    generated_count = sum(
        1
        for record in metadata_records
        if record.get("status") == "generated"
    )

    failed_count = sum(
        1
        for record in metadata_records
        if record.get("status") == "failed"
    )

    print("\nExperiment completed.")
    print(
        "Total planned generations: "
        f"{total_generations}"
    )
    print(
        f"Generated images: {generated_count}"
    )
    print(
        f"Failed generations: {failed_count}"
    )
    print(
        f"Images directory: {output_images_dir}"
    )
    print(
        f"Metadata saved to: {metadata_path}"
    )


if __name__ == "__main__":
    main()