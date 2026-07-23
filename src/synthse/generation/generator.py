from __future__ import annotations

import gc
import re
import time
from pathlib import Path
from typing import Any

import torch


def sanitize_filename_value(
    value: Any,
) -> str:
    """
    Convert an arbitrary value into a filesystem-safe string.

    Letters, numbers, underscores and hyphens are preserved.
    Every other sequence is replaced with one underscore.
    """
    sanitized_value = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        str(value).strip(),
    )

    sanitized_value = sanitized_value.strip("_")

    return sanitized_value or "prompt"


def build_image_filename(
    prompt_id: str,
    width: int | None = None,
    height: int | None = None,
    seed: int | None = None,
) -> str:
    """
    Build a deterministic filename for a generated image.

    Current usage:

        build_image_filename(
            prompt_id="P001_pv01",
            width=512,
            height=512,
            seed=42,
        )

    Result:

        P001_pv01_w512_h512_seed42.png

    The legacy call is also supported:

        build_image_filename("P001", 42)

    Result:

        P001_seed42.png
    """
    # Backward compatibility:
    #
    # build_image_filename("P001", 42)
    #
    # In this case, 42 is initially assigned to `width`.
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

    sanitized_prompt_id = sanitize_filename_value(
        prompt_id
    )

    seed = int(seed)

    if width is None and height is None:
        return (
            f"{sanitized_prompt_id}"
            f"_seed{seed}.png"
        )

    if width is None or height is None:
        raise ValueError(
            "Both width and height must be provided when "
            "the resolution is included in the filename."
        )

    width = int(width)
    height = int(height)

    if width <= 0 or height <= 0:
        raise ValueError(
            "Image width and height must be positive."
        )

    return (
        f"{sanitized_prompt_id}"
        f"_w{width}"
        f"_h{height}"
        f"_seed{seed}.png"
    )


def is_cuda_device(
    device: str,
) -> bool:
    """
    Return True when the supplied device represents CUDA.
    """
    return (
        str(device).lower().startswith("cuda")
        and torch.cuda.is_available()
    )


def synchronize_cuda(
    device: str,
) -> None:
    """
    Wait for all queued CUDA operations to finish.

    CUDA operations are asynchronous, so synchronization is needed
    before stopping execution-time measurements.
    """
    if is_cuda_device(device):
        torch.cuda.synchronize()


def reset_cuda_peak_memory(
    device: str,
) -> None:
    """
    Reset PyTorch peak-memory statistics before one generation.
    """
    if is_cuda_device(device):
        torch.cuda.reset_peak_memory_stats()


def read_cuda_peak_memory(
    device: str,
) -> tuple[float | None, float | None]:
    """
    Return peak allocated and reserved CUDA memory in MiB.
    """
    if not is_cuda_device(device):
        return None, None

    peak_allocated_mb = (
        torch.cuda.max_memory_allocated()
        / (1024**2)
    )

    peak_reserved_mb = (
        torch.cuda.max_memory_reserved()
        / (1024**2)
    )

    return (
        round(peak_allocated_mb, 3),
        round(peak_reserved_mb, 3),
    )


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
) -> dict[str, Any]:
    """
    Generate and save one image.

    The returned dictionary contains timing information, CUDA memory
    statistics and the generation status.

    A CPU-based torch.Generator is used deliberately because it works
    reliably with Diffusers model CPU offloading and keeps seed handling
    deterministic.
    """
    seed = int(seed)
    width = int(width)
    height = int(height)
    num_inference_steps = int(
        num_inference_steps
    )
    guidance_scale = float(
        guidance_scale
    )

    destination = Path(output_path)

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    generator = torch.Generator(
        device="cpu"
    ).manual_seed(seed)

    reset_cuda_peak_memory(
        device
    )

    synchronize_cuda(
        device
    )

    total_start_time = time.perf_counter()

    inference_start_time = time.perf_counter()

    with torch.inference_mode():
        pipeline_result = pipeline(
            prompt=prompt,
            generator=generator,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            width=width,
            height=height,
        )

    synchronize_cuda(
        device
    )

    inference_time_seconds = (
        time.perf_counter()
        - inference_start_time
    )

    if (
        not hasattr(pipeline_result, "images")
        or not pipeline_result.images
    ):
        raise RuntimeError(
            "The diffusion pipeline returned no images."
        )

    generated_image = pipeline_result.images[0]

    save_start_time = time.perf_counter()

    generated_image.save(
        destination
    )

    image_save_time_seconds = (
        time.perf_counter()
        - save_start_time
    )

    execution_time_seconds = (
        time.perf_counter()
        - total_start_time
    )

    (
        peak_allocated_vram_mb,
        peak_reserved_vram_mb,
    ) = read_cuda_peak_memory(
        device
    )

    result = {
        "image_path": str(destination),
        "execution_time_seconds": round(
            execution_time_seconds,
            6,
        ),
        "inference_time_seconds": round(
            inference_time_seconds,
            6,
        ),
        "image_save_time_seconds": round(
            image_save_time_seconds,
            6,
        ),
        # Compatibility with the previous metadata format.
        "peak_vram_mb": peak_allocated_vram_mb,
        "peak_allocated_vram_mb": (
            peak_allocated_vram_mb
        ),
        "peak_reserved_vram_mb": (
            peak_reserved_vram_mb
        ),
        "status": "generated",
        "error_message": "",
    }

    del generated_image
    del pipeline_result
    del generator

    # Do not use torch.cuda.empty_cache() here. It can cause problems
    # with some TorchAO quantized pipelines and repeated generations.
    gc.collect()

    return result