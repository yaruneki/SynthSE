from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from synthse.augmentation.noise import apply_noise


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Empty config file: {config_path}")

    return config


def normalize_path(value: object) -> str:
    if pd.isna(value):
        return ""

    return str(value).replace("\\", "/")


def image_filename_from_row(row: pd.Series) -> str:
    image_path = normalize_path(row.get("image_path", ""))

    if image_path:
        return Path(image_path).name

    prompt_id = row.get("prompt_id", "unknown")
    seed = row.get("seed", "unknown")

    return f"{prompt_id}_seed_{seed}.png"


def build_corrupted_filename(
    original_filename: str,
    noise_type: str,
    level_name: str,
) -> str:
    path = Path(original_filename)
    return f"{path.stem}__{noise_type}_{level_name}{path.suffix}"


def apply_noise_benchmark(
    config: dict,
    max_images: int | None,
    overwrite: bool,
) -> None:
    source_config = config["source"]
    output_config = config["output"]
    noise_config = config["noise"]

    experiment_id = source_config["experiment_id"]
    metadata_file = Path(source_config["metadata_file"])
    images_dir = Path(source_config["images_dir"])

    corrupted_images_dir = Path(output_config["corrupted_images_dir"])
    output_metadata_file = Path(output_config["metadata_file"])

    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if overwrite and corrupted_images_dir.exists():
        shutil.rmtree(corrupted_images_dir)

    corrupted_images_dir.mkdir(parents=True, exist_ok=True)
    output_metadata_file.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.read_csv(metadata_file)

    if "status" in metadata.columns:
        metadata = metadata[metadata["status"].astype(str).str.lower() == "generated"]

    if max_images is not None:
        metadata = metadata.head(max_images)

    random_seed = int(noise_config.get("random_seed", 123))
    rng = np.random.default_rng(random_seed)

    output_records = []

    for _, row in metadata.iterrows():
        original_filename = image_filename_from_row(row)
        original_image_path = images_dir / original_filename

        if not original_image_path.exists():
            print(f"Skipping missing image: {original_image_path}")
            continue

        original_image = Image.open(original_image_path).convert("RGB")

        for corruption in noise_config["corruptions"]:
            noise_type = corruption["type"]

            for level in corruption["levels"]:
                level_name = level["name"]
                corrupted_filename = build_corrupted_filename(
                    original_filename=original_filename,
                    noise_type=noise_type,
                    level_name=level_name,
                )
                corrupted_image_path = corrupted_images_dir / corrupted_filename

                corrupted_image = apply_noise(
                    image=original_image,
                    noise_type=noise_type,
                    parameters=level,
                    rng=rng,
                )

                corrupted_image.save(corrupted_image_path)

                record = row.to_dict()
                record["experiment_id"] = experiment_id
                record["source_image_filename"] = original_filename
                record["source_image_path"] = str(original_image_path)
                record["source_image_id"] = Path(original_filename).stem

                record["image_path"] = str(corrupted_image_path)
                record["corrupted_image_filename"] = corrupted_filename
                record["corrupted_image_path"] = str(corrupted_image_path)

                record["noise_type"] = noise_type
                record["noise_level"] = level_name

                for key, value in level.items():
                    if key != "name":
                        record[f"noise_{key}"] = value

                output_records.append(record)

                print(
                    f"Created {corrupted_filename} "
                    f"from {original_filename} "
                    f"with {noise_type}/{level_name}"
                )

    output_metadata = pd.DataFrame(output_records)
    output_metadata.to_csv(output_metadata_file, index=False, encoding="utf-8")

    print("")
    print("Noise benchmark completed.")
    print(f"Source experiment: {experiment_id}")
    print(f"Generated corrupted images: {len(output_records)}")
    print(f"Corrupted images directory: {corrupted_images_dir}")
    print(f"Noise metadata saved to: {output_metadata_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply controlled image noise to SynthSE generated images."
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to noise benchmark YAML config.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional number of source images to process for testing.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing corrupted image directory before running.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))

    apply_noise_benchmark(
        config=config,
        max_images=args.max_images,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()