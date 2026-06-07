import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from synthse.config.config_loader import load_config
from synthse.generation.generator import build_image_filename, generate_image
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

    required_columns = {"prompt_id", "category", "task", "prompt", "variant_type"}
    missing_columns = required_columns - set(prompts.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in prompts CSV: {missing_columns}")

    return prompts


def validate_generation_config(config: dict) -> None:
    """
    Validate basic generation constraints.
    """
    width = config["generation"]["width"]
    height = config["generation"]["height"]

    if width % 8 != 0 or height % 8 != 0:
        raise ValueError("Image width and height must be divisible by 8.")


def get_gpu_name(device: str) -> str:
    """
    Return the GPU name if CUDA is used and available.
    """
    if device == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)

    return ""


def build_base_record(
    config: dict,
    row: pd.Series,
    seed: int,
    experiment_id: str,
    image_id: str,
    actual_device: str,
) -> dict:
    """
    Build the base metadata record for a generation.
    """
    return {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "prompt_id": row["prompt_id"],
        "category": row["category"],
        "task": row["task"],
        "variant_type": row["variant_type"],
        "prompt": row["prompt"],
        "seed": seed,
        "image_id": image_id,
        "model_name": config["model"]["name"],
        "model_huggingface_id": config["model"]["huggingface_id"],
        "requested_device": config["model"]["device"],
        "actual_device": actual_device,
        "gpu_name": get_gpu_name(actual_device),
        "num_inference_steps": config["generation"]["num_inference_steps"],
        "guidance_scale": config["generation"]["guidance_scale"],
        "width": config["generation"]["width"],
        "height": config["generation"]["height"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SynthSE experiment.")
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

    max_prompts = config.get("run", {}).get("max_prompts")
    if max_prompts is not None:
        prompts = prompts.head(int(max_prompts))

    experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("Configuration loaded successfully.")
    print(f"Project: {config['project']['name']}")
    print(f"Model: {config['model']['huggingface_id']}")
    print(f"Device: {config['model']['device']}")
    print(f"Prompts loaded: {len(prompts)}")
    print(f"Experiment ID: {experiment_id}")

    print("\nPrompt preview:")
    print(prompts[["prompt_id", "task", "prompt"]].head())

    pipeline, actual_device = load_text_to_image_pipeline(
        model_id=config["model"]["huggingface_id"],
        device=config["model"]["device"],
    )

    seeds = config["generation"]["seeds"]
    metadata_records = []

    output_images_dir = Path(config["paths"]["output_images_dir"]) / experiment_id

    total_generations = len(prompts) * len(seeds)

    for _, row in tqdm(
        prompts.iterrows(),
        total=len(prompts),
        desc="Generating images",
    ):
        for seed in seeds:
            image_filename = build_image_filename(row["prompt_id"], seed)
            image_id = image_filename.replace(".png", "")
            image_path = output_images_dir / image_filename

            base_record = build_base_record(
                config=config,
                row=row,
                seed=seed,
                experiment_id=experiment_id,
                image_id=image_id,
                actual_device=actual_device,
            )

            try:
                generation_result = generate_image(
                    pipeline=pipeline,
                    prompt=row["prompt"],
                    seed=seed,
                    output_path=str(image_path),
                    num_inference_steps=config["generation"]["num_inference_steps"],
                    guidance_scale=config["generation"]["guidance_scale"],
                    width=config["generation"]["width"],
                    height=config["generation"]["height"],
                    device=actual_device,
                )

                record = {
                    **base_record,
                    **generation_result,
                }

            except Exception as error:
                record = {
                    **base_record,
                    "image_path": str(image_path),
                    "execution_time_seconds": None,
                    "status": "failed",
                    "error_message": str(error),
                }

            metadata_records.append(record)

    metadata_path = (
        Path(config["paths"]["output_logs_dir"])
        / f"{experiment_id}_generation_metadata.csv"
    )

    save_metadata(metadata_records, str(metadata_path))

    generated_count = sum(1 for record in metadata_records if record["status"] == "generated")
    failed_count = sum(1 for record in metadata_records if record["status"] == "failed")

    print("\nExperiment completed.")
    print(f"Total planned generations: {total_generations}")
    print(f"Generated images: {generated_count}")
    print(f"Failed generations: {failed_count}")
    print(f"Images directory: {output_images_dir}")
    print(f"Metadata saved to: {metadata_path}")


if __name__ == "__main__":
    main()