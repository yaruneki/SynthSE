from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from synthse.config.config_loader import load_config
from synthse.generation.generator import (
    build_image_filename,
    generate_image,
)
from synthse.generation.model_loader import (
    load_text_to_image_pipeline,
)
from synthse.generation.prompt_variation import (
    expand_prompt_dataframe,
)
from synthse.tracking.metadata_logger import (
    save_metadata,
)


REQUIRED_PROMPT_COLUMNS = {
    "prompt_id",
    "category",
    "task",
    "prompt_style",
    "prompt",
    "variant_type",
}


NOISE_METADATA_PREFIXES = (
    "noise_",
    "prompt_noise_",
    "perturbation_",
    "corruption_",
)


NOISE_METADATA_COLUMNS = {
    "noise_type",
    "noise_level",
    "noise_strength",
    "noise_seed",
    "corruption_type",
    "corruption_level",
    "source_image_path",
    "clean_image_path",
    "noisy_image_path",
}


def load_prompts(
    prompts_path: str,
) -> pd.DataFrame:
    """
    Load and validate a prompt CSV file.
    """
    path = Path(
        prompts_path
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {prompts_path}"
        )

    prompts = pd.read_csv(
        path
    )

    missing_columns = (
        REQUIRED_PROMPT_COLUMNS
        - set(prompts.columns)
    )

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


def normalize_integer_values(
    value: Any,
    field_name: str,
) -> list[int]:
    """
    Convert a scalar integer or a sequence into a list of integers.

    Examples:

        512
        ->
        [512]

        [42, 43, 44]
        ->
        [42, 43, 44]
    """
    if isinstance(
        value,
        (list, tuple),
    ):
        values = [
            int(item)
            for item in value
        ]
    else:
        values = [
            int(value)
        ]

    if not values:
        raise ValueError(
            f"{field_name} must contain at least one value."
        )

    return values


def build_resolution_pairs(
    generation_config: dict[str, Any],
) -> list[tuple[int, int]]:
    """
    Build width-height pairs from scalar values or lists.

    Examples:

        width: 512
        height: 512

        ->
        [(512, 512)]

        width: [256, 256, 512, 512]
        height: [256, 512, 256, 512]

        ->
        [
            (256, 256),
            (256, 512),
            (512, 256),
            (512, 512),
        ]

    When one field contains only one value, that value is reused for
    every value contained in the other field.
    """
    widths = normalize_integer_values(
        generation_config["width"],
        "generation.width",
    )

    heights = normalize_integer_values(
        generation_config["height"],
        "generation.height",
    )

    if len(widths) == len(heights):
        return list(
            zip(
                widths,
                heights,
                strict=True,
            )
        )

    if len(widths) == 1:
        return [
            (
                widths[0],
                height,
            )
            for height in heights
        ]

    if len(heights) == 1:
        return [
            (
                width,
                heights[0],
            )
            for width in widths
        ]

    raise ValueError(
        "generation.width and generation.height must contain "
        "the same number of values, unless one of them contains "
        "only one value."
    )


def validate_generation_config(
    config: dict[str, Any],
) -> None:
    """
    Validate the required experiment configuration.
    """
    required_sections = {
        "project",
        "model",
        "generation",
        "paths",
    }

    missing_sections = (
        required_sections
        - set(config)
    )

    if missing_sections:
        raise ValueError(
            "Missing configuration sections: "
            f"{sorted(missing_sections)}"
        )

    required_model_fields = {
        "name",
        "huggingface_id",
        "device",
    }

    missing_model_fields = (
        required_model_fields
        - set(config["model"])
    )

    if missing_model_fields:
        raise ValueError(
            "Missing model configuration fields: "
            f"{sorted(missing_model_fields)}"
        )

    required_generation_fields = {
        "num_inference_steps",
        "guidance_scale",
        "width",
        "height",
        "seeds",
    }

    missing_generation_fields = (
        required_generation_fields
        - set(config["generation"])
    )

    if missing_generation_fields:
        raise ValueError(
            "Missing generation configuration fields: "
            f"{sorted(missing_generation_fields)}"
        )

    required_path_fields = {
        "prompts_file",
        "output_images_dir",
        "output_logs_dir",
    }

    missing_path_fields = (
        required_path_fields
        - set(config["paths"])
    )

    if missing_path_fields:
        raise ValueError(
            "Missing path configuration fields: "
            f"{sorted(missing_path_fields)}"
        )

    resolution_pairs = build_resolution_pairs(
        config["generation"]
    )

    for width, height in resolution_pairs:
        if width <= 0 or height <= 0:
            raise ValueError(
                "Image width and height must be positive."
            )

        if (
            width % 8 != 0
            or height % 8 != 0
        ):
            raise ValueError(
                "Image width and height must be divisible by 8. "
                f"Invalid resolution: {width}x{height}"
            )

    seeds = normalize_integer_values(
        config["generation"]["seeds"],
        "generation.seeds",
    )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "generation.seeds contains duplicate values."
        )

    num_inference_steps = int(
        config["generation"][
            "num_inference_steps"
        ]
    )

    if num_inference_steps <= 0:
        raise ValueError(
            "generation.num_inference_steps must be "
            "greater than zero."
        )

    guidance_scale = float(
        config["generation"][
            "guidance_scale"
        ]
    )

    if guidance_scale < 0:
        raise ValueError(
            "generation.guidance_scale must be "
            "greater than or equal to zero."
        )


def get_gpu_name(
    device: str,
) -> str:
    """
    Return the CUDA GPU name when applicable.
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
    Read an optional value from a pandas row.

    Missing columns and NaN values are replaced with `default`.
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


def normalize_quantization(
    value: Any,
) -> str:
    """
    Normalize the configured quantization mode.
    """
    if value is None:
        return "none"

    quantization = str(
        value
    ).strip().lower()

    if quantization in {
        "",
        "false",
        "null",
        "no",
        "off",
    }:
        return "none"

    return quantization


def normalize_quantized_components(
    components: Any,
) -> list[str]:
    """
    Normalize the quantized component configuration.
    """
    if components is None:
        return []

    if isinstance(
        components,
        str,
    ):
        normalized_components = [
            component.strip().lower()
            for component
            in components.split(",")
            if component.strip()
        ]

    elif isinstance(
        components,
        (list, tuple, set),
    ):
        normalized_components = [
            str(component).strip().lower()
            for component in components
            if str(component).strip()
        ]

    else:
        raise ValueError(
            "model.quantized_components must be "
            "a string or a list of strings."
        )

    return list(
        dict.fromkeys(
            normalized_components
        )
    )


def extract_noise_metadata(
    row: pd.Series,
) -> dict[str, Any]:
    """
    Preserve any noise or corruption metadata already present in the
    prompt dataframe.

    This does not apply image corruption itself. Image noise remains
    managed by the dedicated robustness scripts. It only ensures that
    existing noise-related columns are not lost from the metadata.
    """
    noise_metadata: dict[str, Any] = {}

    for column in row.index:
        column_name = str(
            column
        ).strip()

        is_noise_column = (
            column_name in NOISE_METADATA_COLUMNS
            or column_name.startswith(
                NOISE_METADATA_PREFIXES
            )
        )

        if not is_noise_column:
            continue

        noise_metadata[column_name] = (
            get_row_value(
                row,
                column_name,
                "",
            )
        )

    return noise_metadata


def build_base_record(
    config: dict[str, Any],
    row: pd.Series,
    seed: int,
    width: int,
    height: int,
    experiment_id: str,
    image_id: str,
    actual_device: str,
) -> dict[str, Any]:
    """
    Build the metadata shared by successful and failed generations.
    """
    model_config = config[
        "model"
    ]

    generation_config = config[
        "generation"
    ]

    quantization = normalize_quantization(
        model_config.get(
            "quantization",
            "none",
        )
    )

    quantized_components = (
        normalize_quantized_components(
            model_config.get(
                "quantized_components"
            )
        )
    )

    source_prompt_id = get_row_value(
        row,
        "source_prompt_id",
        get_row_value(
            row,
            "prompt_id",
        ),
    )

    prompt_variant_id = get_row_value(
        row,
        "prompt_variant_id",
        get_row_value(
            row,
            "prompt_id",
        ),
    )

    actual_prompt = str(
        get_row_value(
            row,
            "prompt",
            "",
        )
    )

    original_prompt = str(
        get_row_value(
            row,
            "original_prompt",
            actual_prompt,
        )
    )

    modified_prompt = str(
        get_row_value(
            row,
            "modified_prompt",
            actual_prompt,
        )
    )

    record: dict[str, Any] = {
        "experiment_id": experiment_id,
        "timestamp": datetime.now().isoformat(
            timespec="seconds"
        ),
        "prompt_id": get_row_value(
            row,
            "prompt_id",
        ),
        "source_prompt_id": source_prompt_id,
        "prompt_variant_id": prompt_variant_id,
        "task_id": get_row_value(
            row,
            "task_id",
            "",
        ),
        "category": get_row_value(
            row,
            "category",
        ),
        "task": get_row_value(
            row,
            "task",
        ),
        "variant_type": get_row_value(
            row,
            "variant_type",
        ),
        "prompt_style": get_row_value(
            row,
            "prompt_style",
        ),
        "original_prompt": original_prompt,
        "modified_prompt": modified_prompt,
        # This is the exact text sent to the model.
        "prompt": actual_prompt,
        "prompt_variation_type": get_row_value(
            row,
            "prompt_variation_type",
            "none",
        ),
        "prompt_variation_index": get_row_value(
            row,
            "prompt_variation_index",
            0,
        ),
        "prompt_variation_seed": get_row_value(
            row,
            "prompt_variation_seed",
            "",
        ),
        "prompt_is_original": get_row_value(
            row,
            "prompt_is_original",
            True,
        ),
        "seed": int(seed),
        "image_id": image_id,
        "model_name": model_config["name"],
        "model_huggingface_id": (
            model_config["huggingface_id"]
        ),
        "quantization": quantization,
        "quantized_components": ",".join(
            quantized_components
        ),
        "requested_device": model_config[
            "device"
        ],
        "actual_device": actual_device,
        "gpu_name": get_gpu_name(
            actual_device
        ),
        "num_inference_steps": int(
            generation_config[
                "num_inference_steps"
            ]
        ),
        "guidance_scale": float(
            generation_config[
                "guidance_scale"
            ]
        ),
        "width": int(width),
        "height": int(height),
    }

    record.update(
        extract_noise_metadata(
            row
        )
    )

    return record


def resolve_experiment_id(
    config: dict[str, Any],
) -> str:
    """
    Resolve the experiment identifier.

    `run.experiment_id` is used by the isolated runner so multiple
    subprocesses can write into the same unified experiment.
    """
    configured_experiment_id = (
        config.get(
            "run",
            {},
        ).get(
            "experiment_id"
        )
    )

    if configured_experiment_id:
        return str(
            configured_experiment_id
        )

    return datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )


def print_prompt_preview(
    prompts: pd.DataFrame,
) -> None:
    """
    Print the first expanded prompt rows.
    """
    preferred_columns = [
        "prompt_id",
        "prompt_variant_id",
        "task",
        "prompt_variation_type",
        "original_prompt",
        "modified_prompt",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in prompts.columns
    ]

    print("\nPrompt preview:")

    print(
        prompts[
            available_columns
        ].head().to_string(
            index=False
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a SynthSE text-to-image experiment."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file.",
    )

    arguments = parser.parse_args()

    config = load_config(
        arguments.config
    )

    validate_generation_config(
        config
    )

    prompts_file = config[
        "paths"
    ]["prompts_file"]

    source_prompts = load_prompts(
        prompts_file
    )

    max_prompts = (
        config.get(
            "run",
            {},
        ).get(
            "max_prompts"
        )
    )

    if max_prompts is not None:
        max_prompts = int(
            max_prompts
        )

        if max_prompts <= 0:
            raise ValueError(
                "run.max_prompts must be greater than zero."
            )

        # max_prompts limits source prompts, not expanded variants.
        source_prompts = source_prompts.head(
            max_prompts
        )

    source_prompt_count = len(
        source_prompts
    )

    prompts = expand_prompt_dataframe(
        prompts=source_prompts,
        configuration=config.get(
            "prompt_variation",
            {},
        ),
    )

    if prompts.empty:
        raise ValueError(
            "Prompt expansion produced no rows."
        )

    experiment_id = resolve_experiment_id(
        config
    )

    model_config = config[
        "model"
    ]

    quantization = normalize_quantization(
        model_config.get(
            "quantization",
            "none",
        )
    )

    quantized_components = (
        normalize_quantized_components(
            model_config.get(
                "quantized_components"
            )
        )
    )

    seeds = normalize_integer_values(
        config["generation"]["seeds"],
        "generation.seeds",
    )

    resolution_pairs = build_resolution_pairs(
        config["generation"]
    )

    total_generations = (
        len(prompts)
        * len(seeds)
        * len(resolution_pairs)
    )

    print(
        "Configuration loaded successfully."
    )
    print(
        f"Project: {config['project']['name']}"
    )
    print(
        "Model: "
        f"{model_config['huggingface_id']}"
    )
    print(
        f"Device: {model_config['device']}"
    )
    print(
        f"Quantization: {quantization}"
    )
    print(
        "Quantized components: "
        f"{quantized_components or 'none'}"
    )
    print(
        "Source prompts loaded: "
        f"{source_prompt_count}"
    )
    print(
        "Prompts after variation expansion: "
        f"{len(prompts)}"
    )
    print(
        f"Seeds: {seeds}"
    )
    print(
        f"Resolutions: {resolution_pairs}"
    )
    print(
        "Total planned generations: "
        f"{total_generations}"
    )
    print(
        f"Experiment ID: {experiment_id}"
    )

    print_prompt_preview(
        prompts
    )

    pipeline, actual_device = (
        load_text_to_image_pipeline(
            model_id=model_config[
                "huggingface_id"
            ],
            device=model_config[
                "device"
            ],
            quantization=quantization,
            quantized_components=(
                quantized_components
            ),
        )
    )

    output_images_dir = (
        Path(
            config["paths"][
                "output_images_dir"
            ]
        )
        / experiment_id
    )

    output_images_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_logs_dir = Path(
        config["paths"][
            "output_logs_dir"
        ]
    )

    output_logs_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_records: list[
        dict[str, Any]
    ] = []

    with tqdm(
        total=total_generations,
        desc="Generating images",
    ) as progress_bar:
        for _, row in prompts.iterrows():
            prompt_variant_id = str(
                get_row_value(
                    row,
                    "prompt_variant_id",
                    get_row_value(
                        row,
                        "prompt_id",
                    ),
                )
            )

            actual_prompt = str(
                get_row_value(
                    row,
                    "prompt",
                    "",
                )
            )

            for width, height in resolution_pairs:
                for seed in seeds:
                    image_filename = (
                        build_image_filename(
                            prompt_id=(
                                prompt_variant_id
                            ),
                            width=width,
                            height=height,
                            seed=seed,
                        )
                    )

                    image_id = Path(
                        image_filename
                    ).stem

                    image_path = (
                        output_images_dir
                        / image_filename
                    )

                    base_record = (
                        build_base_record(
                            config=config,
                            row=row,
                            seed=seed,
                            width=width,
                            height=height,
                            experiment_id=(
                                experiment_id
                            ),
                            image_id=image_id,
                            actual_device=(
                                actual_device
                            ),
                        )
                    )

                    try:
                        generation_result = (
                            generate_image(
                                pipeline=pipeline,
                                prompt=actual_prompt,
                                seed=seed,
                                output_path=str(
                                    image_path
                                ),
                                num_inference_steps=int(
                                    config[
                                        "generation"
                                    ][
                                        "num_inference_steps"
                                    ]
                                ),
                                guidance_scale=float(
                                    config[
                                        "generation"
                                    ][
                                        "guidance_scale"
                                    ]
                                ),
                                width=width,
                                height=height,
                                device=actual_device,
                            )
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
                            "inference_time_seconds": None,
                            "image_save_time_seconds": None,
                            "peak_vram_mb": None,
                            "peak_allocated_vram_mb": None,
                            "peak_reserved_vram_mb": None,
                            "status": "failed",
                            "error_message": str(
                                error
                            ),
                        }

                    metadata_records.append(
                        record
                    )

                    progress_bar.update(
                        1
                    )

    metadata_path = (
        output_logs_dir
        / (
            f"{experiment_id}"
            "_generation_metadata.csv"
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