from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import torch

from synthse.tracking.resource_monitor import ResourceMonitor


def sanitize_filename_value(
    value: Any,
) -> str:
    """
    Convert a value into a filesystem-safe string.
    """
    text = str(value).strip()

    sanitized = "".join(
        character
        if character.isalnum()
        or character in {"-", "_"}
        else "_"
        for character in text
    )

    sanitized = sanitized.strip("_")

    return sanitized or "prompt"


def build_image_filename(
    prompt_id: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> str:
    """
    Build a deterministic image filename.

    New format:

        P001_pv01_w512_h512_seed42.png

    The legacy positional call remains supported:

        build_image_filename("P001", 42)

    which produces:

        P001_seed42.png
    """
    # Backward compatibility with:
    #
    # build_image_filename(prompt_id, seed)
    if (
        seed is None
        and width is not None
        and height is None
    ):
        seed = int(width)
        width = None

    if seed is None:
        raise ValueError(
            "A seed is required to build the image filename."
        )

    safe_prompt_id = sanitize_filename_value(
        prompt_id
    )

    seed = int(seed)

    if width is None and height is None:
        return (
            f"{safe_prompt_id}"
            f"_seed{seed}.png"
        )

    if width is None or height is None:
        raise ValueError(
            "Both width and height must be supplied when "
            "the resolution is included in the filename."
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive."
        )

    return (
        f"{safe_prompt_id}"
        f"_w{width}"
        f"_h{height}"
        f"_seed{seed}.png"
    )


def is_cuda_device(
    device: str,
) -> bool:
    """
    Return True when CUDA is currently available and selected.
    """
    return (
        str(device).lower().startswith("cuda")
        and torch.cuda.is_available()
    )


def synchronize_cuda(
    device: str,
) -> None:
    """
    Wait until queued CUDA operations have completed.
    """
    if is_cuda_device(device):
        torch.cuda.synchronize()


def clear_generation_memory() -> None:
    """
    Release unreachable Python objects after one generation.

    torch.cuda.empty_cache() and torch.cuda.ipc_collect() are not
    called here because they may cause instability with repeated
    TorchAO INT8 generations and pipelines using CPU offload.
    """
    gc.collect()


def normalize_monitoring_configuration(
    monitoring_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Normalize the optional monitoring configuration.
    """
    config = monitoring_config or {}

    return {
        "enabled": bool(
            config.get(
                "enabled",
                False,
            )
        ),
        "gpu_index": int(
            config.get(
                "gpu_index",
                0,
            )
        ),
        "sample_interval_seconds": float(
            config.get(
                "sample_interval_seconds",
                0.2,
            )
        ),
    }


def build_unavailable_resource_metrics() -> dict[str, Any]:
    """
    Return empty monitoring fields when resource monitoring is disabled
    or fails to initialize.
    """
    return {
        "monitoring_enabled": False,
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
    }


def generate_image(
    pipeline: Any,
    prompt: str,
    seed: int,
    output_path: str,
    num_inference_steps: int,
    guidance_scale: float,
    width: int,
    height: int,
    device: str,
    monitoring_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Generate one image, save it and return generation/resource metadata.

    The monitoring interval covers:

        inference + image saving

    Separate timestamps are still collected for:

        inference_seconds
        image_save_seconds
        generation_total_seconds

    Model loading must be monitored separately in run_experiment.py.
    """
    path = Path(
        output_path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = int(seed)
    width = int(width)
    height = int(height)
    num_inference_steps = int(
        num_inference_steps
    )
    guidance_scale = float(
        guidance_scale
    )

    # CPU generator works correctly with both full-CUDA pipelines and
    # pipelines using model CPU offload.
    generator = torch.Generator(
        device="cpu"
    ).manual_seed(
        seed
    )

    result = None
    image = None
    monitor: ResourceMonitor | None = None

    monitoring = normalize_monitoring_configuration(
        monitoring_config
    )

    resource_metrics = (
        build_unavailable_resource_metrics()
    )

    try:
        synchronize_cuda(
            device
        )

        if monitoring["enabled"]:
            monitor = ResourceMonitor(
                gpu_index=monitoring[
                    "gpu_index"
                ],
                sample_interval_seconds=monitoring[
                    "sample_interval_seconds"
                ],
                enabled=True,
            )

            monitor.start(
                phase="generation"
            )

        generation_total_start = (
            time.perf_counter()
        )

        inference_start = (
            time.perf_counter()
        )

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=(
                    num_inference_steps
                ),
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        synchronize_cuda(
            device
        )

        inference_seconds = (
            time.perf_counter()
            - inference_start
        )

        if (
            not hasattr(result, "images")
            or not result.images
        ):
            raise RuntimeError(
                "The diffusion pipeline returned no images."
            )

        image = result.images[0]

        image_save_start = (
            time.perf_counter()
        )

        image.save(
            path
        )

        image_save_seconds = (
            time.perf_counter()
            - image_save_start
        )

        generation_total_seconds = (
            time.perf_counter()
            - generation_total_start
        )

        if monitor is not None:
            resource_metrics = (
                monitor.stop()
            )

        gpu_energy_j = (
            resource_metrics.get(
                "gpu_energy_j"
            )
        )

        gpu_energy_wh = (
            resource_metrics.get(
                "gpu_energy_wh"
            )
        )

        images_per_hour = (
            3600.0
            / generation_total_seconds
            if generation_total_seconds > 0
            else None
        )

        generation_metadata = {
            "image_path": str(path),
            # Compatibility with the previous metadata name.
            "execution_time_seconds": round(
                generation_total_seconds,
                6,
            ),
            "inference_seconds": round(
                inference_seconds,
                6,
            ),
            "image_save_seconds": round(
                image_save_seconds,
                6,
            ),
            "generation_total_seconds": round(
                generation_total_seconds,
                6,
            ),
            **resource_metrics,
            # One call to generate_image() creates one image.
            "joules_per_image": gpu_energy_j,
            "wh_per_image": gpu_energy_wh,
            "images_per_hour": (
                round(
                    images_per_hour,
                    6,
                )
                if images_per_hour is not None
                else None
            ),
            # Compatibility with the old peak_vram_mb field.
            "peak_vram_mb": resource_metrics.get(
                "torch_peak_allocated_vram_mb"
            ),
            "status": "generated",
            "error_message": "",
        }

        return generation_metadata

    except Exception:
        # Stop the monitor before propagating the generation error.
        if monitor is not None:
            try:
                if getattr(
                    monitor,
                    "_running",
                    False,
                ):
                    monitor.stop()
            except Exception:
                pass

        raise

    finally:
        if image is not None:
            del image

        if result is not None:
            del result

        del generator

        clear_generation_memory()