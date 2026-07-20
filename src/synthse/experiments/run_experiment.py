import argparse
import gc
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


def load_prompts(prompts_path: str) -> pd.DataFrame:
    """
    Load prompts from a CSV file.
    """
    path = Path(prompts_path)

    if not path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = pd.read_csv(path)

    required_columns = {
        "prompt_id",
        "category",
        "task",
        "prompt_style",
        "prompt",
        "variant_type",
    }
    missing_columns = required_columns - set(prompts.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in prompts CSV: {missing_columns}"
        )

    return prompts


def as_list(value: Any) -> list:
    """
    Normalize a scalar value or a list value into a list.
    """
    if isinstance(value, list):
        return value

    return [value]


def build_resolution_pairs(
    config: dict,
) -> list[tuple[int, int]]:
    """
    Build all width-height combinations from the config.

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


def validate_generation_config(config: dict) -> None:
    """
    Validate basic generation constraints.

    Width and height can be either scalar values or lists.
    All values must be positive integers divisible by 8.
    """
    required_generation_keys = {
        "num_inference_steps",
        "guidance_scale",
        "width",
        "height",
        "seeds",
    }

    missing_keys = (
        required_generation_keys
        - set(config["generation"].keys())
    )

    if missing_keys:
        raise ValueError(
            f"Missing generation config keys: {missing_keys}"
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

    for width, height in resolution_pairs:
        if width <= 0:
            raise ValueError(
                f"Image width must be > 0. Invalid width: {width}"
            )

        if height <= 0:
            raise ValueError(
                f"Image height must be > 0. Invalid height: {height}"
            )

        if width % 8 != 0:
            raise ValueError(
                "Image width must be divisible by 8. "
                f"Invalid width: {width}"
            )

        if height % 8 != 0:
            raise ValueError(
                "Image height must be divisible by 8. "
                f"Invalid height: {height}"
            )

    num_inference_steps = int(
        config["generation"]["num_inference_steps"]
    )

    guidance_scale = float(
        config["generation"]["guidance_scale"]
    )

    if num_inference_steps <= 0:
        raise ValueError(
            "generation.num_inference_steps must be > 0."
        )

    if guidance_scale < 0:
        raise ValueError(
            "generation.guidance_scale must be >= 0."
        )


def get_gpu_name(device: str) -> str:
    """
    Return the GPU name if CUDA is used and available.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return ""


def sanitize_filename_value(value: Any) -> str:
    """
    Convert a value into a filesystem-safe filename component.
    """
    text = str(value)

    forbidden_chars = [
        "/",
        "\\",
        ":",
        "*",
        "?",
        '"',
        "<",
        ">",
        "|",
        " ",
    ]

    for char in forbidden_chars:
        text = text.replace(char, "_")

    return text


def build_image_filename(
    prompt_id: Any,
    width: int,
    height: int,
    seed: int,
) -> str:
    """
    Build a unique filename for each generated image.
    """
    safe_prompt_id = sanitize_filename_value(prompt_id)

    return (
        f"{safe_prompt_id}_"
        f"w{width}_h{height}_seed{seed}.png"
    )


def build_base_record(
    config: dict,
    row: pd.Series,
    seed: int,
    experiment_id: str,
    image_id: str,
    actual_device: str,
    width: int,
    height: int,
) -> dict:
    """
    Build the base metadata record for a generation.
    """
    quantization = config["model"].get(
        "quantization",
        "none",
    )

    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "task": row["task"],
        "variant_type": row["variant_type"],
        "prompt_style": row["prompt_style"],
        "prompt": row["prompt"],
        "seed": seed,
        "image_id": image_id,
        "model_name": config["model"]["name"],
        "model_huggingface_id": (
            config["model"]["huggingface_id"]
        ),
        "quantization": quantization,
        "quantized_components": ",".join(
            config["model"].get("quantized_components", [])
        )
        if config["model"].get("quantized_components")
        else "",
        "requested_device": config["model"]["device"],
        "actual_device": actual_device,
        "gpu_name": get_gpu_name(actual_device),
        "num_inference_steps": int(
            config["generation"]["num_inference_steps"]
        ),
        "guidance_scale": float(
            config["generation"]["guidance_scale"]
        ),
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
    }


def clear_runtime_memory(
    actual_device: str,
    quantization: str | None = None,
) -> None:
    """
    Release Python and CUDA caches between generations.

    For INT8 quantized TorchAO pipelines, avoid aggressive CUDA cache
    cleanup because it may trigger CUDA errors after otherwise successful
    generations.
    """
    gc.collect()

    normalized_quantization = (
        str(quantization).strip().lower()
        if quantization is not None
        else "none"
    )

    if actual_device != "cuda" or not torch.cuda.is_available():
        return

    # For TorchAO INT8 runs, skip CUDA cache cleanup.
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

    prompts_file = config["paths"]["prompts_file"]
    prompts = load_prompts(prompts_file)

    max_prompts = config.get("run", {}).get(
        "max_prompts"
    )

    if max_prompts is not None:
        prompts = prompts.head(int(max_prompts))

    experiment_id = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    seeds = [
        int(seed)
        for seed in as_list(
            config["generation"]["seeds"]
        )
    ]

    resolution_pairs = build_resolution_pairs(config)

    num_inference_steps = int(
        config["generation"]["num_inference_steps"]
    )

    guidance_scale = float(
        config["generation"]["guidance_scale"]
    )

    quantization = config["model"].get(
        "quantization"
    )

    quantization_label = (
        quantization
        if quantization is not None
        else "none"
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
        f"Model: "
        f"{config['model']['huggingface_id']}"
    )
    print(
        f"Quantization: {quantization_label}"
    )
    print(
        f"Requested device: "
        f"{config['model']['device']}"
    )
    print(f"Prompts loaded: {len(prompts)}")
    print(f"Seeds: {seeds}")
    print(
        f"Resolution pairs: {resolution_pairs}"
    )
    print(
        f"Total planned generations: "
        f"{total_generations}"
    )
    print(f"Experiment ID: {experiment_id}")

    print("\nPrompt preview:")
    print(
        prompts[
            ["prompt_id", "task", "prompt"]
        ].head()
    )

    pipeline, actual_device = load_text_to_image_pipeline(
        model_id=config["model"]["huggingface_id"],
        device=config["model"]["device"],
        quantization=quantization,
        quantized_components=config["model"].get("quantized_components"),
    )

    print(f"\nActual device: {actual_device}")

    gpu_name = get_gpu_name(actual_device)

    if gpu_name:
        print(f"GPU: {gpu_name}")

    metadata_records: list[dict] = []

    with tqdm(
        total=total_generations,
        desc="Generating images",
    ) as progress_bar:
        for _, row in prompts.iterrows():
            for width, height in resolution_pairs:
                for seed in seeds:
                    image_filename = (
                        build_image_filename(
                            prompt_id=row["prompt_id"],
                            width=width,
                            height=height,
                            seed=seed,
                        )
                    )

                    image_id = image_filename.removesuffix(
                        ".png"
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
                    )

                    try:
                        generation_result = generate_image(
                            pipeline=pipeline,
                            prompt=row["prompt"],
                            seed=seed,
                            output_path=str(image_path),
                            num_inference_steps=(
                                num_inference_steps
                            ),
                            guidance_scale=(
                                guidance_scale
                            ),
                            width=width,
                            height=height,
                            device=actual_device,
                        )

                        record = {
                            **base_record,
                            **generation_result,
                        }

                    except Exception as error:
                        record = {
                            **base_record,
                            "image_path": str(
                                image_path
                            ),
                            "execution_time_seconds": None,
                            "peak_vram_mb": None,
                            "status": "failed",
                            "error_message": str(error),
                        }

                    metadata_records.append(record)
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
        f"Total planned generations: "
        f"{total_generations}"
    )
    print(
        f"Generated images: {generated_count}"
    )
    print(
        f"Failed generations: {failed_count}"
    )
    print(
        f"Images directory: "
        f"{output_images_dir}"
    )
    print(
        f"Metadata saved to: {metadata_path}"
    )


if __name__ == "__main__":
    main()
