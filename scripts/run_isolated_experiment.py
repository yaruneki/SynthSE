from __future__ import annotations

import argparse
import copy
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


STATE_FILE = "session_state.yaml"
PROCESS_FILE = "process_records.csv"
PARTIAL_METADATA_FILE = "consolidated_metadata.csv"
PARTIAL_MODEL_LOAD_FILE = "consolidated_model_load_resources.csv"


def as_list(value: Any) -> list[Any]:
    """Normalize a scalar or sequence into a list."""
    if isinstance(value, (list, tuple)):
        return list(value)

    return [value]


def sanitize_filename(value: Any) -> str:
    """Convert a value into a filesystem-safe component."""
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))
    return text.strip("_") or "item"


def normalize_quantization(value: Any) -> str:
    """Normalize the configured quantization mode."""
    if value is None:
        return "none"

    normalized = str(value).strip().lower()

    if normalized in {"", "none", "null", "false", "no", "off"}:
        return "none"

    return normalized


def normalize_quantized_components(value: Any) -> list[str]:
    """Normalize the configured quantized pipeline components."""
    if value is None:
        return []

    if isinstance(value, str):
        components = [
            component.strip().lower()
            for component in value.split(",")
            if component.strip()
        ]
    elif isinstance(value, (list, tuple, set)):
        components = [
            str(component).strip().lower()
            for component in value
            if str(component).strip()
        ]
    else:
        raise ValueError(
            "model.quantized_components must be a string "
            "or a list of strings."
        )

    return list(dict.fromkeys(components))


def validate_base_config(config: dict[str, Any]) -> None:
    """
    Validate the configuration before launching isolated subprocesses.

    This catches configuration errors once, before repeatedly loading
    the same model in separate child processes.
    """
    required_sections = {
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

    missing_model_keys = required_model_keys - set(config["model"])

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

    missing_path_keys = required_path_keys - set(config["paths"])

    if missing_path_keys:
        raise ValueError(
            "Missing path config keys: "
            f"{sorted(missing_path_keys)}"
        )

    seeds = [
        int(value)
        for value in as_list(config["generation"]["seeds"])
    ]
    widths = [
        int(value)
        for value in as_list(config["generation"]["width"])
    ]
    heights = [
        int(value)
        for value in as_list(config["generation"]["height"])
    ]

    if not seeds:
        raise ValueError(
            "generation.seeds must contain at least one seed."
        )

    if not widths or not heights:
        raise ValueError(
            "At least one width and one height are required."
        )

    for width in widths:
        if width <= 0 or width % 8 != 0:
            raise ValueError(
                "Every image width must be positive and divisible by 8. "
                f"Invalid width: {width}"
            )

    for height in heights:
        if height <= 0 or height % 8 != 0:
            raise ValueError(
                "Every image height must be positive and divisible by 8. "
                f"Invalid height: {height}"
            )

    if int(config["generation"]["num_inference_steps"]) <= 0:
        raise ValueError(
            "generation.num_inference_steps must be greater than zero."
        )

    if float(config["generation"]["guidance_scale"]) < 0:
        raise ValueError(
            "generation.guidance_scale must be greater than or equal to zero."
        )

    quantization = normalize_quantization(
        config["model"].get("quantization")
    )
    quantized_components = normalize_quantized_components(
        config["model"].get("quantized_components")
    )

    if quantization != "none" and not quantized_components:
        raise ValueError(
            "Quantization is enabled, but model.quantized_components "
            "is empty. For the current INT8 pipeline, configure for "
            "example: quantized_components: [unet]."
        )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML file: {path}")

    return data


def save_yaml(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            data,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def resolve_project_root(config_path: Path) -> Path:
    absolute_config = config_path.resolve()

    if absolute_config.parent.name == "configs":
        return absolute_config.parent.parent

    current = Path.cwd().resolve()

    if (current / "src" / "synthse").exists():
        return current

    raise RuntimeError(
        "Unable to determine the SynthSE project root. "
        "Run the script from the SynthSE root directory."
    )


def resolve_path(project_root: Path, configured_path: str) -> Path:
    path = Path(configured_path)
    return path if path.is_absolute() else project_root / path


def load_prompts(
    project_root: Path,
    config: dict[str, Any],
) -> pd.DataFrame:
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


def build_generation_label(
    prompt_id: Any,
    width: int,
    height: int,
    seed: int,
) -> str:
    return (
        f"{sanitize_filename(prompt_id)}_"
        f"w{width}_h{height}_seed{seed}"
    )


def build_combinations(
    prompts: pd.DataFrame,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    seeds = [int(value) for value in as_list(config["generation"]["seeds"])]
    widths = [int(value) for value in as_list(config["generation"]["width"])]
    heights = [int(value) for value in as_list(config["generation"]["height"])]

    combinations: list[dict[str, Any]] = []
    position = 0

    for _, prompt_row in prompts.iterrows():
        for width in widths:
            for height in heights:
                for seed in seeds:
                    position += 1
                    combinations.append(
                        {
                            "position": position,
                            "prompt_row": prompt_row,
                            "width": width,
                            "height": height,
                            "seed": seed,
                            "label": build_generation_label(
                                prompt_row["prompt_id"],
                                width,
                                height,
                                seed,
                            ),
                        }
                    )

    return combinations


def build_single_generation_config(
    base_config: dict[str, Any],
    prompt_file: Path,
    seed: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    config = copy.deepcopy(base_config)
    config["paths"]["prompts_file"] = str(prompt_file.resolve())
    config["generation"]["seeds"] = [seed]
    config["generation"]["width"] = width
    config["generation"]["height"] = height

    run_config = config.setdefault("run", {})
    run_config["max_prompts"] = 1

    # Every retry must create fresh child output files. Reusing an explicit
    # experiment ID could overwrite files and prevent new-file detection.
    run_config.pop("experiment_id", None)

    return config


def run_single_generation(
    project_root: Path,
    config_path: Path,
) -> subprocess.CompletedProcess[Any]:
    environment = os.environ.copy()
    environment["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"

    source_path = str(project_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join([source_path, existing_pythonpath])
    )

    return subprocess.run(
        [
            sys.executable,
            "-m",
            "synthse.experiments.run_experiment",
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        env=environment,
        check=False,
    )


def find_new_output_file(
    logs_dir: Path,
    pattern: str,
    files_before: set[Path],
) -> Path | None:
    """Return the newest output file created by one child process."""
    files_after = set(logs_dir.glob(pattern))
    new_files = files_after - files_before

    if not new_files:
        return None

    return max(
        new_files,
        key=lambda path: path.stat().st_mtime,
    )


def find_new_metadata_file(
    logs_dir: Path,
    files_before: set[Path],
) -> Path | None:
    return find_new_output_file(
        logs_dir=logs_dir,
        pattern="*_generation_metadata.csv",
        files_before=files_before,
    )


def find_new_model_load_file(
    logs_dir: Path,
    files_before: set[Path],
) -> Path | None:
    return find_new_output_file(
        logs_dir=logs_dir,
        pattern="*_model_load_resources.csv",
        files_before=files_before,
    )


def resolve_metadata_image_path(
    project_root: Path,
    value: Any,
) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def consolidate_metadata_and_images(
    metadata_file: Path,
    unified_images_dir: Path,
    batch_id: str,
    project_root: Path,
) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_file)
    rows: list[dict[str, Any]] = []

    for _, row in metadata.iterrows():
        record = row.to_dict()
        source = resolve_metadata_image_path(
            project_root,
            record.get("image_path", ""),
        )

        if source.exists():
            destination = unified_images_dir / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)

            if source.resolve() != destination.resolve():
                if destination.exists():
                    destination.unlink()
                shutil.move(str(source), str(destination))

            record["image_path"] = (
                destination.relative_to(project_root).as_posix()
            )

            if source.parent.exists() and source.parent != unified_images_dir:
                try:
                    source.parent.rmdir()
                except OSError:
                    pass

        record["experiment_id"] = batch_id
        rows.append(record)

    return pd.DataFrame(rows)


def consolidate_model_load_resources(
    model_load_file: Path,
    batch_id: str,
    combination: dict[str, Any],
) -> pd.DataFrame:
    """
    Add isolated-generation context to one child model-load CSV.

    Each child process loads the model once, so each resulting row is
    associated with exactly one prompt/seed/resolution combination.
    """
    dataframe = pd.read_csv(model_load_file)

    if dataframe.empty:
        return dataframe

    prompt_row = combination["prompt_row"]

    dataframe["source_experiment_id"] = dataframe.get(
        "experiment_id",
        "",
    )
    dataframe["experiment_id"] = batch_id
    dataframe["process_index"] = combination["position"]
    dataframe["generation_label"] = combination["label"]
    dataframe["prompt_id"] = prompt_row.get("prompt_id", "")
    dataframe["task_id"] = prompt_row.get("task_id", "")
    dataframe["seed"] = combination["seed"]
    dataframe["width"] = combination["width"]
    dataframe["height"] = combination["height"]
    dataframe["resolution"] = (
        f"{combination['width']}x{combination['height']}"
    )

    return dataframe


def model_load_key(row: pd.Series) -> str:
    """Return the unique key for one isolated model-load measurement."""
    generation_label = row.get("generation_label", "")

    if pd.notna(generation_label) and str(generation_label).strip():
        return str(generation_label)

    return "|".join(
        [
            str(row.get("prompt_id", "")),
            str(row.get("width", "")),
            str(row.get("height", "")),
            str(row.get("seed", "")),
        ]
    )


def upsert_model_load_resources(
    new_rows: pd.DataFrame,
    path: Path,
) -> None:
    """Insert or replace model-load measurements by generation label."""
    if new_rows.empty:
        return

    existing = load_dataframe(path)
    columns = list(
        dict.fromkeys(
            existing.columns.tolist()
            + new_rows.columns.tolist()
        )
    )

    existing = existing.reindex(columns=columns)
    new_rows = new_rows.reindex(columns=columns)

    new_keys = set(
        new_rows.apply(
            model_load_key,
            axis=1,
        )
    )

    if not existing.empty:
        existing = existing[
            ~existing.apply(
                model_load_key,
                axis=1,
            ).isin(new_keys)
        ]

    pd.concat(
        [existing, new_rows],
        ignore_index=True,
    ).to_csv(
        path,
        index=False,
    )


def load_dataframe(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def metadata_key(row: pd.Series) -> str:
    image_id = row.get("image_id", "")

    if pd.notna(image_id) and str(image_id).strip():
        return str(image_id)

    image_path = row.get("image_path", "")

    if pd.notna(image_path) and str(image_path).strip():
        return Path(str(image_path)).stem

    return build_generation_label(
        row.get("prompt_id", ""),
        int(row.get("width", 0)),
        int(row.get("height", 0)),
        int(row.get("seed", 0)),
    )


def upsert_metadata(new_rows: pd.DataFrame, path: Path) -> None:
    if new_rows.empty:
        return

    existing = load_dataframe(path)
    columns = list(
        dict.fromkeys(
            existing.columns.tolist() + new_rows.columns.tolist()
        )
    )
    existing = existing.reindex(columns=columns)
    new_rows = new_rows.reindex(columns=columns)
    new_keys = set(new_rows.apply(metadata_key, axis=1))

    if not existing.empty:
        existing = existing[
            ~existing.apply(metadata_key, axis=1).isin(new_keys)
        ]

    pd.concat([existing, new_rows], ignore_index=True).to_csv(
        path,
        index=False,
    )


def load_process_records(path: Path) -> list[dict[str, Any]]:
    dataframe = load_dataframe(path)
    return [] if dataframe.empty else dataframe.to_dict(orient="records")


def save_process_records(
    records: list[dict[str, Any]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def upsert_process_record(
    records: list[dict[str, Any]],
    new_record: dict[str, Any],
) -> list[dict[str, Any]]:
    label = str(new_record["generation_label"])
    records = [
        record
        for record in records
        if str(record.get("generation_label", "")) != label
    ]
    records.append(new_record)
    records.sort(key=lambda record: int(record.get("process_index", 0)))
    return records


def extract_batch_id(session_dir: Path) -> str:
    match = re.search(r"isolated_(\d{8}_\d{6})", session_dir.name)

    if not match:
        raise ValueError(
            "The resume directory name does not contain a valid batch ID: "
            f"{session_dir.name}"
        )

    return match.group(1)


def legacy_labels(session_dir: Path) -> set[str]:
    return {
        path.name.removesuffix("_config.yaml")
        for path in session_dir.glob("*_config.yaml")
    }


def session_model_id(session_dir: Path) -> str:
    state_path = session_dir / STATE_FILE

    if state_path.exists():
        return str(load_yaml(state_path).get("model_id", ""))

    configs = sorted(session_dir.glob("*_config.yaml"))

    if not configs:
        return ""

    return str(
        load_yaml(configs[0])
        .get("model", {})
        .get("huggingface_id", "")
    )


def validate_resume_directory(
    session_dir: Path,
    base_config: dict[str, Any],
    expected_labels: set[str],
) -> None:
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Resume directory not found: {session_dir}")

    saved_model = session_model_id(session_dir)
    expected_model = str(base_config["model"]["huggingface_id"])

    if saved_model and saved_model != expected_model:
        raise ValueError(
            f"Resume model mismatch: found '{saved_model}', "
            f"expected '{expected_model}'."
        )

    labels = legacy_labels(session_dir)

    if not labels and not (session_dir / STATE_FILE).exists():
        raise ValueError(
            "The resume directory contains neither session_state.yaml "
            "nor legacy *_config.yaml files."
        )

    incompatible = labels - expected_labels

    if incompatible:
        examples = ", ".join(sorted(incompatible)[:5])
        raise ValueError(
            "The resume directory is incompatible with the current YAML. "
            f"Unexpected generations include: {examples}"
        )


def find_compatible_sessions(
    temporary_parent: Path,
    base_config: dict[str, Any],
    expected_labels: set[str],
) -> list[Path]:
    candidates: list[Path] = []

    for directory in sorted(temporary_parent.glob("isolated_*")):
        if not directory.is_dir():
            continue

        state_path = directory / STATE_FILE

        if state_path.exists():
            try:
                if str(load_yaml(state_path).get("status", "")) == "completed":
                    continue
            except Exception:
                continue

        try:
            validate_resume_directory(
                directory,
                base_config,
                expected_labels,
            )
        except (FileNotFoundError, ValueError):
            continue

        candidates.append(directory)

    return candidates


def create_session(temporary_parent: Path) -> tuple[Path, str]:
    batch_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = temporary_parent / f"isolated_{batch_id}"
    counter = 1

    while session_dir.exists():
        session_dir = temporary_parent / f"isolated_{batch_id}_{counter}"
        counter += 1

    session_dir.mkdir(parents=True)
    return session_dir, batch_id


def save_state(
    state: dict[str, Any],
    session_dir: Path,
) -> None:
    state = copy.deepcopy(state)
    state["last_updated_at"] = datetime.now().isoformat(timespec="seconds")
    save_yaml(state, session_dir / STATE_FILE)


def recovered_metadata_row(
    base_config: dict[str, Any],
    combination: dict[str, Any],
    image_path: Path,
    project_root: Path,
    batch_id: str,
) -> dict[str, Any]:
    prompt_row = combination["prompt_row"]
    model = base_config["model"]
    components = model.get("quantized_components", [])

    if isinstance(components, list):
        components = ",".join(str(value) for value in components)

    return {
        "experiment_id": batch_id,
        "timestamp": datetime.fromtimestamp(
            image_path.stat().st_mtime
        ).isoformat(timespec="seconds"),
        "prompt_id": prompt_row.get("prompt_id", ""),
        "task_id": prompt_row.get("task_id", ""),
        "category": prompt_row.get("category", ""),
        "task": prompt_row.get("task", ""),
        "prompt_style": prompt_row.get(
            "prompt_style",
            prompt_row.get("variant_type", ""),
        ),
        "variant_type": prompt_row.get("variant_type", ""),
        "prompt": prompt_row.get("prompt", ""),
        "seed": combination["seed"],
        "image_id": combination["label"],
        "model_name": model.get("name", ""),
        "model_huggingface_id": model.get("huggingface_id", ""),
        "quantization": model.get("quantization", "none") or "none",
        "quantized_components": components,
        "requested_device": model.get("device", ""),
        "actual_device": "",
        "gpu_name": "",
        "num_inference_steps": int(
            base_config["generation"]["num_inference_steps"]
        ),
        "guidance_scale": float(
            base_config["generation"]["guidance_scale"]
        ),
        "width": combination["width"],
        "height": combination["height"],
        "resolution": (
            f"{combination['width']}x{combination['height']}"
        ),
        "image_path": image_path.relative_to(project_root).as_posix(),
        "model_load_seconds": None,
        "execution_time_seconds": None,
        "inference_seconds": None,
        "image_save_seconds": None,
        "generation_total_seconds": None,
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
        "status": "generated",
        "error_message": "",
        "recovered_from_legacy_session": True,
    }


def recover_existing_images(
    combinations: list[dict[str, Any]],
    unified_images_dir: Path,
    project_root: Path,
    batch_id: str,
    base_config: dict[str, Any],
    process_records: list[dict[str, Any]],
    metadata_path: Path,
) -> tuple[list[dict[str, Any]], set[str]]:
    completed: set[str] = set()
    recovered_rows: list[dict[str, Any]] = []
    existing_metadata = load_dataframe(metadata_path)
    existing_metadata_keys = (
        set(existing_metadata.apply(metadata_key, axis=1))
        if not existing_metadata.empty
        else set()
    )

    for combination in combinations:
        label = combination["label"]
        image_path = unified_images_dir / f"{label}.png"

        if not image_path.exists():
            continue

        completed.add(label)
        process_records = upsert_process_record(
            process_records,
            {
                "experiment_id": batch_id,
                "process_index": combination["position"],
                "generation_label": label,
                "prompt_id": combination["prompt_row"].get("prompt_id", ""),
                "seed": combination["seed"],
                "width": combination["width"],
                "height": combination["height"],
                "status": "completed",
                "return_code": 0,
                "source_metadata_file": "legacy_image_recovery",
            },
        )
        if label not in existing_metadata_keys:
            recovered_rows.append(
                recovered_metadata_row(
                    base_config,
                    combination,
                    image_path,
                    project_root,
                    batch_id,
                )
            )

    if recovered_rows:
        upsert_metadata(pd.DataFrame(recovered_rows), metadata_path)

    return process_records, completed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run SynthSE generations in separate processes and resume "
            "unfinished isolated_tmp sessions."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume-dir",
        help="Existing modern or legacy isolated_tmp directory to resume.",
    )
    parser.add_argument(
        "--force-new",
        action="store_true",
        help="Ignore compatible sessions and start a new one.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
    )
    args = parser.parse_args()

    if args.resume_dir and args.force_new:
        parser.error("--resume-dir and --force-new cannot be used together.")

    config_path = Path(args.config)
    project_root = resolve_project_root(config_path)
    config_path = config_path.resolve()
    base_config = load_yaml(config_path)
    validate_base_config(base_config)
    prompts = load_prompts(project_root, base_config)
    combinations = build_combinations(prompts, base_config)
    expected_labels = {item["label"] for item in combinations}

    logs_dir = resolve_path(
        project_root,
        base_config["paths"]["output_logs_dir"],
    )
    images_root = resolve_path(
        project_root,
        base_config["paths"]["output_images_dir"],
    )
    temporary_parent = project_root / "outputs" / "isolated_tmp"

    logs_dir.mkdir(parents=True, exist_ok=True)
    images_root.mkdir(parents=True, exist_ok=True)
    temporary_parent.mkdir(parents=True, exist_ok=True)

    resumed = False

    if args.resume_dir:
        session_dir = Path(args.resume_dir)

        if not session_dir.is_absolute():
            session_dir = project_root / session_dir

        session_dir = session_dir.resolve()
        validate_resume_directory(
            session_dir,
            base_config,
            expected_labels,
        )
        batch_id = extract_batch_id(session_dir)
        resumed = True

    elif not args.force_new:
        candidates = find_compatible_sessions(
            temporary_parent,
            base_config,
            expected_labels,
        )

        if len(candidates) > 1:
            options = "\n".join(f"- {path}" for path in candidates)
            raise RuntimeError(
                "Multiple compatible unfinished sessions were found:\n"
                f"{options}\n"
                "Specify the correct one with --resume-dir."
            )

        if len(candidates) == 1:
            session_dir = candidates[0].resolve()
            batch_id = extract_batch_id(session_dir)
            resumed = True
        else:
            session_dir, batch_id = create_session(temporary_parent)

    else:
        session_dir, batch_id = create_session(temporary_parent)

    unified_images_dir = images_root / batch_id
    unified_images_dir.mkdir(parents=True, exist_ok=True)

    process_path = session_dir / PROCESS_FILE
    partial_metadata_path = session_dir / PARTIAL_METADATA_FILE
    partial_model_load_path = session_dir / PARTIAL_MODEL_LOAD_FILE

    final_metadata_path = (
        logs_dir
        / f"{batch_id}_generation_metadata.csv"
    )
    final_model_load_path = (
        logs_dir
        / f"{batch_id}_model_load_resources.csv"
    )
    summary_path = (
        logs_dir
        / f"{batch_id}_isolated_process_summary.csv"
    )

    state_path = session_dir / STATE_FILE

    if state_path.exists():
        state = load_yaml(state_path)
    else:
        state = {
            "session_id": batch_id,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "legacy_session": resumed,
        }

    state.update(
        {
            "batch_id": batch_id,
            "base_config_path": str(config_path),
            "model_id": base_config["model"]["huggingface_id"],
            "total_generations": len(combinations),
            "status": "running",
        }
    )
    save_state(state, session_dir)

    process_records = load_process_records(process_path)
    process_records, completed = recover_existing_images(
        combinations,
        unified_images_dir,
        project_root,
        batch_id,
        base_config,
        process_records,
        partial_metadata_path,
    )
    save_process_records(process_records, process_path)

    seeds = [int(value) for value in as_list(base_config["generation"]["seeds"])]
    widths = [int(value) for value in as_list(base_config["generation"]["width"])]
    heights = [int(value) for value in as_list(base_config["generation"]["height"])]

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
    print(f"Session directory: {session_dir}")
    print(
        "Monitoring enabled: "
        f"{bool(base_config.get('monitoring', {}).get('enabled', False))}"
    )

    if resumed:
        print("Resume mode enabled.")
        print(f"Already completed generations: {len(completed)}")

    interrupted = False
    failed_and_stopped = False

    try:
        for item in combinations:
            position = item["position"]
            prompt_row = item["prompt_row"]
            width = item["width"]
            height = item["height"]
            seed = item["seed"]
            label = item["label"]
            expected_image = unified_images_dir / f"{label}.png"

            print()
            print("=" * 80)
            print(f"Generation {position}/{len(combinations)}: {label}")
            print("=" * 80)

            if label in completed and expected_image.exists():
                print("Skipping: image already exists in the resumed batch.")
                continue

            prompt_path = session_dir / f"{label}_prompt.csv"
            single_config_path = session_dir / f"{label}_config.yaml"

            if not prompt_path.exists():
                pd.DataFrame([prompt_row.to_dict()]).to_csv(
                    prompt_path,
                    index=False,
                )

            if not single_config_path.exists():
                single_config = build_single_generation_config(
                    base_config,
                    prompt_path,
                    seed,
                    width,
                    height,
                )
                save_yaml(single_config, single_config_path)

            state["current_generation"] = label
            state["current_position"] = position
            save_state(state, session_dir)

            metadata_before = set(
                logs_dir.glob("*_generation_metadata.csv")
            )
            model_load_before = set(
                logs_dir.glob("*_model_load_resources.csv")
            )

            result = run_single_generation(
                project_root,
                single_config_path,
            )

            metadata_file = find_new_metadata_file(
                logs_dir,
                metadata_before,
            )
            model_load_file = find_new_model_load_file(
                logs_dir,
                model_load_before,
            )

            metadata_frame = pd.DataFrame()
            model_load_frame = pd.DataFrame()
            source_metadata_file = ""
            source_model_load_file = ""

            if metadata_file is not None and metadata_file.exists():
                metadata_frame = consolidate_metadata_and_images(
                    metadata_file,
                    unified_images_dir,
                    batch_id,
                    project_root,
                )
                upsert_metadata(
                    metadata_frame,
                    partial_metadata_path,
                )
                source_metadata_file = str(metadata_file)

                try:
                    metadata_file.unlink()
                except OSError:
                    pass

            if model_load_file is not None and model_load_file.exists():
                model_load_frame = consolidate_model_load_resources(
                    model_load_file=model_load_file,
                    batch_id=batch_id,
                    combination=item,
                )
                upsert_model_load_resources(
                    model_load_frame,
                    partial_model_load_path,
                )
                source_model_load_file = str(model_load_file)

                try:
                    model_load_file.unlink()
                except OSError:
                    pass

            generated_status = (
                not metadata_frame.empty
                and "status" in metadata_frame.columns
                and (
                    metadata_frame["status"].astype(str) == "generated"
                ).any()
            )
            success = (
                result.returncode == 0
                and generated_status
                and expected_image.exists()
            )

            process_records = upsert_process_record(
                process_records,
                {
                    "experiment_id": batch_id,
                    "process_index": position,
                    "generation_label": label,
                    "prompt_id": prompt_row.get("prompt_id", ""),
                    "seed": seed,
                    "width": width,
                    "height": height,
                    "status": "completed" if success else "failed",
                    "return_code": result.returncode,
                    "source_metadata_file": source_metadata_file,
                    "source_model_load_file": source_model_load_file,
                },
            )
            save_process_records(process_records, process_path)

            if success:
                completed.add(label)
                print("Generation completed and consolidated.")
                continue

            print(
                "Generation failed or produced no valid image. "
                f"Return code: {result.returncode}"
            )

            if not args.continue_on_error:
                failed_and_stopped = True
                break

    except KeyboardInterrupt:
        interrupted = True
        print("\nExecution interrupted. Progress is being saved.")

    completed_count = len(
        {
            str(record.get("generation_label", ""))
            for record in process_records
            if str(record.get("status", "")) == "completed"
        }
    )
    failed_count = sum(
        str(record.get("status", "")) == "failed"
        for record in process_records
    )

    state["status"] = (
        "completed"
        if completed_count == len(combinations)
        else "interrupted"
    )
    save_state(state, session_dir)

    load_dataframe(partial_metadata_path).to_csv(
        final_metadata_path,
        index=False,
    )

    model_load_dataframe = load_dataframe(
        partial_model_load_path
    )

    if not model_load_dataframe.empty:
        model_load_dataframe.to_csv(
            final_model_load_path,
            index=False,
        )

    save_process_records(
        process_records,
        summary_path,
    )

    print()
    print("Isolated experiment finished.")
    print(f"Completed generations: {completed_count}/{len(combinations)}")
    print(f"Failed generation records: {failed_count}")
    print(f"Unified images directory: {unified_images_dir}")
    print(f"Unified metadata CSV: {final_metadata_path}")

    if final_model_load_path.exists():
        print(
            "Unified model-load resources CSV: "
            f"{final_model_load_path}"
        )

    print(f"Process summary: {summary_path}")
    print(f"Session state: {state_path}")

    if interrupted:
        raise SystemExit(130)

    if failed_and_stopped:
        raise SystemExit(1)


if __name__ == "__main__":
    main()