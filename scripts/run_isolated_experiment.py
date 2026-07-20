from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def as_list(value: Any) -> list[Any]:
    """
    Convert a scalar configuration value into a list.
    """
    if isinstance(value, list):
        return value
    return [value]


def sanitize_filename(value: Any) -> str:
    """
    Convert a value into a filesystem-safe filename component.
    """
    text = str(value)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "item"


def load_yaml(path: Path) -> dict[str, Any]:
    """
    Load a YAML configuration file.
    """
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def save_yaml(config: dict[str, Any], path: Path) -> None:
    """
    Save a YAML configuration file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            config,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def resolve_project_root(config_path: Path) -> Path:
    """
    Resolve the SynthSE project root.
    """
    absolute_config = config_path.resolve()

    if absolute_config.parent.name == "configs":
        return absolute_config.parent.parent

    current = Path.cwd().resolve()

    if (current / "src" / "synthse").exists():
        return current

    raise RuntimeError(
        "Unable to determine the SynthSE project root. "
        "Run this script from the SynthSE root directory."
    )


def resolve_path(project_root: Path, configured_path: str) -> Path:
    """
    Resolve a path from the YAML configuration.
    Relative paths are interpreted relative to the project root.
    """
    path = Path(configured_path)

    if path.is_absolute():
        return path

    return project_root / path


def load_prompts(
    project_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
    """
    Load prompts and optionally apply run.max_prompts.
    """
    prompts_path = resolve_path(
        project_root,
        config["paths"]["prompts_file"],
    )

    if not prompts_path.exists():
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    prompts = pd.read_csv(prompts_path)

    max_prompts = config.get("run", {}).get("max_prompts")

    if max_prompts is not None:
        prompts = prompts.head(int(max_prompts))

    if prompts.empty:
        raise ValueError("No prompts are available for the experiment.")

    return prompts


def find_new_metadata_file(
    logs_directory: Path,
    files_before_run: set[Path],
) -> Path | None:
    """
    Find the metadata CSV created by the subprocess.
    """
    files_after_run = set(logs_directory.glob("*_generation_metadata.csv"))
    new_files = files_after_run - files_before_run

    if not new_files:
        return None

    return max(new_files, key=lambda file: file.stat().st_mtime)


def build_single_generation_config(
    base_config: dict[str, Any],
    prompt_file: Path,
    seed: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    """
    Create a config that produces exactly one image.
    """
    config = copy.deepcopy(base_config)

    config["paths"]["prompts_file"] = str(prompt_file)
    config["generation"]["seeds"] = [seed]
    config["generation"]["width"] = width
    config["generation"]["height"] = height

    config.setdefault("run", {})
    config["run"]["max_prompts"] = 1

    return config


def run_single_generation(
    project_root: Path,
    generated_config_path: Path,
) -> subprocess.CompletedProcess:
    """
    Run one isolated SynthSE generation in a fresh Python process.
    """
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    command = [
        sys.executable,
        "-m",
        "synthse.experiments.run_experiment",
        "--config",
        str(generated_config_path),
    ]

    return subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    )


def ensure_unique_destination(destination: Path) -> Path:
    """
    If the destination file already exists, create a unique variant.
    """
    if not destination.exists():
        return destination

    stem = destination.stem
    suffix = destination.suffix
    parent = destination.parent

    counter = 1
    while True:
        candidate = parent / f"{stem}_{counter}{suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def consolidate_metadata_and_images(
    metadata_file: Path,
    unified_images_dir: Path,
    unified_experiment_id: str,
    project_root: Path,
) -> pd.DataFrame:
    """
    Read one subprocess metadata CSV, move its generated images into the
    unified output directory, and update the metadata with a path relative
    to the SynthSE project root.

    Example stored path:
        /outputs/images/20260718_235558/P001_w512_h512_seed44.png
    """
    metadata = pd.read_csv(metadata_file)

    updated_rows: list[dict[str, Any]] = []

    for _, row in metadata.iterrows():
        row_dict = row.to_dict()

        original_image_path = Path(str(row_dict["image_path"]))

        if original_image_path.exists():
            destination = unified_images_dir / original_image_path.name
            destination = ensure_unique_destination(destination)

            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(original_image_path), str(destination))

            relative_destination = destination.relative_to(project_root)

            row_dict["image_path"] = relative_destination.as_posix()

            source_image_dir = original_image_path.parent

            if source_image_dir.exists():
                try:
                    source_image_dir.rmdir()
                except OSError:
                    pass

        row_dict["experiment_id"] = unified_experiment_id
        updated_rows.append(row_dict)

    return pd.DataFrame(updated_rows)

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run each SynthSE image generation in a separate Python "
            "process, then consolidate outputs into a single images "
            "directory and a single metadata CSV."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the base YAML configuration.",
    )

    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue with later generations if one subprocess fails.",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    project_root = resolve_project_root(config_path)
    config_path = config_path.resolve()

    base_config = load_yaml(config_path)
    prompts = load_prompts(project_root, base_config)

    seeds = [int(seed) for seed in as_list(base_config["generation"]["seeds"])]
    widths = [int(width) for width in as_list(base_config["generation"]["width"])]
    heights = [int(height) for height in as_list(base_config["generation"]["height"])]

    combinations = [
        (row_index, row, width, height, seed)
        for row_index, row in prompts.iterrows()
        for width in widths
        for height in heights
        for seed in seeds
    ]

    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_logs_directory = resolve_path(
        project_root,
        base_config["paths"]["output_logs_dir"],
    )
    output_logs_directory.mkdir(parents=True, exist_ok=True)

    output_images_root = resolve_path(
        project_root,
        base_config["paths"]["output_images_dir"],
    )
    output_images_root.mkdir(parents=True, exist_ok=True)

    unified_images_dir = output_images_root / batch_id
    unified_images_dir.mkdir(parents=True, exist_ok=True)

    unified_metadata_frames: list[pd.DataFrame] = []
    process_records: list[dict[str, Any]] = []

    print("Isolated experiment initialized.")
    print(f"Project root: {project_root}")
    print(f"Base config: {config_path}")
    print(f"Model: {base_config['model']['huggingface_id']}")
    print(f"Prompts: {len(prompts)}")
    print(f"Seeds: {seeds}")
    print(f"Widths: {widths}")
    print(f"Heights: {heights}")
    print(f"Total isolated generations: {len(combinations)}")
    print(f"Unified experiment ID: {batch_id}")
    print(f"Unified images directory: {unified_images_dir}")

    temporary_parent = project_root / "outputs" / "isolated_tmp"
    temporary_parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=f"isolated_{batch_id}_",
        dir=temporary_parent,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)

        for position, (_, prompt_row, width, height, seed) in enumerate(
            combinations,
            start=1,
        ):
            prompt_id = sanitize_filename(prompt_row["prompt_id"])
            generation_label = f"{prompt_id}_w{width}_h{height}_seed{seed}"

            print()
            print("=" * 80)
            print(
                f"Generation {position}/{len(combinations)}: "
                f"{generation_label}"
            )
            print("=" * 80)

            single_prompt_path = temporary_path / f"{generation_label}_prompt.csv"
            single_config_path = temporary_path / f"{generation_label}_config.yaml"

            pd.DataFrame([prompt_row.to_dict()]).to_csv(
                single_prompt_path,
                index=False,
            )

            single_config = build_single_generation_config(
                base_config=base_config,
                prompt_file=single_prompt_path,
                seed=seed,
                width=width,
                height=height,
            )

            save_yaml(single_config, single_config_path)

            metadata_files_before_run = set(
                output_logs_directory.glob("*_generation_metadata.csv")
            )

            process_result = run_single_generation(
                project_root=project_root,
                generated_config_path=single_config_path,
            )

            return_code = process_result.returncode
            process_status = "completed" if return_code == 0 else "failed"

            metadata_file = find_new_metadata_file(
                logs_directory=output_logs_directory,
                files_before_run=metadata_files_before_run,
            )

            consolidated_metadata_file = ""

            if metadata_file is not None and metadata_file.exists():
                metadata_frame = consolidate_metadata_and_images(
                    metadata_file=metadata_file,
                    unified_images_dir=unified_images_dir,
                    unified_experiment_id=batch_id,
                    project_root=project_root,
                )

                unified_metadata_frames.append(metadata_frame)
                consolidated_metadata_file = str(metadata_file)

                try:
                    metadata_file.unlink()
                except OSError:
                    pass

            process_records.append(
                {
                    "experiment_id": batch_id,
                    "process_index": position,
                    "prompt_id": prompt_row["prompt_id"],
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "status": process_status,
                    "return_code": return_code,
                    "source_metadata_file": consolidated_metadata_file,
                }
            )

            if return_code != 0:
                print(f"Generation failed with return code: {return_code}")

                if return_code in {-9, 137}:
                    print(
                        "The subprocess was probably terminated by the operating system."
                    )

                if not args.continue_on_error:
                    print(
                        "Stopping the batch. Use --continue-on-error "
                        "to continue after failures."
                    )
                    break
            else:
                print(
                    "Generation completed and consolidated into the unified output."
                )

    unified_metadata_path = (
        output_logs_directory / f"{batch_id}_generation_metadata.csv"
    )

    if unified_metadata_frames:
        merged_metadata = pd.concat(
            unified_metadata_frames,
            ignore_index=True,
        )
        merged_metadata.to_csv(unified_metadata_path, index=False)
    else:
        pd.DataFrame().to_csv(unified_metadata_path, index=False)

    process_summary_path = (
        output_logs_directory / f"{batch_id}_isolated_process_summary.csv"
    )
    pd.DataFrame(process_records).to_csv(process_summary_path, index=False)

    completed_count = sum(
        record["status"] == "completed"
        for record in process_records
    )
    failed_count = sum(
        record["status"] == "failed"
        for record in process_records
    )

    print()
    print("Isolated experiment completed.")
    print(f"Completed generations: {completed_count}")
    print(f"Failed generations: {failed_count}")
    print(f"Unified images directory: {unified_images_dir}")
    print(f"Unified metadata CSV: {unified_metadata_path}")
    print(f"Process summary: {process_summary_path}")


if __name__ == "__main__":
    main()