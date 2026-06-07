from pathlib import Path
import time

import torch


def build_image_filename(prompt_id: str, seed: int) -> str:
    """
    Build a deterministic image filename.
    """
    return f"{prompt_id}_seed_{seed}.png"


def generate_image(
    pipeline,
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
    Generate a single image and save it to disk.

    Returns:
        Metadata about the generation process.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    generator_device = "cuda" if device == "cuda" and torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=generator_device).manual_seed(seed)

    start_time = time.perf_counter()

    image = pipeline(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
        generator=generator,
    ).images[0]

    execution_time_seconds = round(time.perf_counter() - start_time, 4)

    image.save(path)

    return {
        "image_path": str(path),
        "execution_time_seconds": execution_time_seconds,
        "status": "generated",
        "error_message": "",
    }