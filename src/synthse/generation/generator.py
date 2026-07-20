from pathlib import Path
from typing import Any
import gc
import time

import torch


def build_image_filename(prompt_id: str, seed: int) -> str:
    """
    Build a deterministic image filename.
    """
    return f"{prompt_id}_seed_{seed}.png"


def clear_generation_memory(device: str) -> None:
    """
    Release Python and CUDA memory after a generation.

    This is particularly useful for large SDXL pipelines using
    model CPU offload, such as JuggernautXL.
    """
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            # ipc_collect may be unavailable in some CUDA configurations.
            pass


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
) -> dict:
    """
    Generate a single image, save it to disk and return its metadata.

    The random generator is created on CPU because this works reliably
    both with pipelines loaded entirely on CUDA and with model CPU offload.

    Returns:
        A dictionary containing generation metadata.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # A CPU generator works more reliably when model CPU offload is enabled.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    result = None
    image = None

    peak_vram_mb = None

    try:
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            result = pipeline(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize()

        execution_time_seconds = round(
            time.perf_counter() - start_time,
            4,
        )

        image = result.images[0]
        image.save(path)

        if device == "cuda" and torch.cuda.is_available():
            peak_vram_mb = round(
                torch.cuda.max_memory_allocated() / (1024**2),
                2,
            )

        generation_metadata = {
            "image_path": str(path),
            "execution_time_seconds": execution_time_seconds,
            "peak_vram_mb": peak_vram_mb,
            "status": "generated",
            "error_message": "",
        }

        return generation_metadata

    finally:
        # Remove references to large objects before clearing the caches.
        if image is not None:
            del image

        if result is not None:
            del result

        del generator

        clear_generation_memory(device)