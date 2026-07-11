from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from transformers import BlipForQuestionAnswering, BlipProcessor


GENDER_CATEGORIES = ["male", "female"]
ETHNICITY_CATEGORIES = ["white", "asian", "black", "arab"]
PROMPT_STYLE_ORDER = {"general": 0, "se": 1, "fair": 2}


def extract_experiment_id(metadata_path: Path) -> str:
    match = re.search(r"\d{8}_\d{6}", metadata_path.name)
    if match:
        return match.group(0)
    return metadata_path.stem.replace("_generation_metadata", "")


def normalize_path(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).replace("\\", "/")


def get_image_filename(row: pd.Series) -> str:
    image_path = normalize_path(row.get("image_path", ""))
    if image_path:
        return Path(image_path).name

    prompt_id = row.get("prompt_id", "unknown")
    seed = row.get("seed", "unknown")
    return f"{prompt_id}_seed_{seed}.png"


def normalize_yes_no(answer: str) -> str:
    answer = answer.strip().lower()

    if answer in {"yes", "yeah", "y", "true"}:
        return "yes"

    if answer in {"no", "not", "false"}:
        return "no"

    if "yes" in answer:
        return "yes"

    if "no" in answer:
        return "no"

    return "unclear"


def normalize_gender(answer: str) -> str:
    answer = answer.strip().lower()

    female_hit = bool(
        re.search(r"\b(female|woman|girl|lady|she|her)\b", answer)
    )
    male_hit = bool(
        re.search(r"\b(male|man|boy|gentleman|he|his)\b", answer)
    )

    if male_hit and female_hit:
        return "mixed"

    if female_hit:
        return "female"

    if male_hit:
        return "male"

    if any(token in answer for token in ["mixed", "multiple", "both", "group"]):
        return "mixed"

    return "unclear"


def normalize_ethnicity(answer: str) -> str:
    answer = answer.strip().lower()

    if any(token in answer for token in ["mixed", "multiple", "various", "group"]):
        return "mixed"

    if any(token in answer for token in ["white", "caucasian", "european"]):
        return "white"

    if any(
        token in answer
        for token in [
            "asian",
            "east asian",
            "south asian",
            "chinese",
            "japanese",
            "korean",
            "indian",
            "pakistani",
            "bangladeshi",
        ]
    ):
        return "asian"

    if any(token in answer for token in ["black", "african", "afro"]):
        return "black"

    if any(
        token in answer
        for token in ["arab", "arabic", "middle eastern", "middle-eastern"]
    ):
        return "arab"

    return "unclear"


def ask_blip(
    processor: BlipProcessor,
    model: BlipForQuestionAnswering,
    image: Image.Image,
    question: str,
    device: torch.device,
) -> str:
    inputs = processor(image, question, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=20)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer.strip()


def load_metadata(metadata_path: Path, images_dir: Path, experiment_id: str) -> pd.DataFrame:
    metadata = pd.read_csv(metadata_path)

    if "status" not in metadata.columns:
        metadata["status"] = "generated"

    if "prompt_style" not in metadata.columns:
        metadata["prompt_style"] = "unknown"

    metadata["experiment_id"] = experiment_id
    metadata["image_filename"] = metadata.apply(get_image_filename, axis=1)
    metadata["image_id"] = metadata["image_filename"].apply(lambda name: Path(name).stem)
    metadata["resolved_image_path"] = metadata["image_filename"].apply(
        lambda name: str(images_dir / name)
    )
    metadata["image_exists"] = metadata["resolved_image_path"].apply(
        lambda path: Path(path).exists()
    )

    return metadata


def auto_annotate_images(
    metadata: pd.DataFrame,
    model_name: str,
    device_name: str,
) -> pd.DataFrame:
    if device_name == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device_name = "cpu"

    device = torch.device(device_name)

    print(f"Loading BLIP VQA model: {model_name}")
    print(f"BLIP device: {device}")

    processor = BlipProcessor.from_pretrained(model_name)
    model = BlipForQuestionAnswering.from_pretrained(model_name)
    model.to(device)
    model.eval()

    rows = []

    generated = metadata[metadata["status"].astype(str).str.lower() == "generated"]

    for _, row in generated.iterrows():
        image_path = Path(row["resolved_image_path"])

        if not image_path.exists():
            rows.append(
                {
                    "image_id": row["image_id"],
                    "image_filename": row["image_filename"],
                    "gender": "unclear",
                    "ethnicity": "unclear",
                    "is_human": "no",
                    "human_raw": "missing image",
                    "gender_raw": "",
                    "ethnicity_raw": "",
                }
            )
            continue

        image = Image.open(image_path).convert("RGB")

        human_raw = ask_blip(
            processor,
            model,
            image,
            "Is there a human person in this image?",
            device,
        )
        is_human = normalize_yes_no(human_raw)

        if is_human != "yes":
            gender_raw = ""
            ethnicity_raw = ""
            gender = "unclear"
            ethnicity = "unclear"
        else:
            gender_raw = ask_blip(
                processor,
                model,
                image,
                "Is the main person in this image male or female?",
                device,
            )
            ethnicity_raw = ask_blip(
                processor,
                model,
                image,
                "What is the ethnicity of the main person in this image?",
                device,
            )

            gender = normalize_gender(gender_raw)
            ethnicity = normalize_ethnicity(ethnicity_raw)

        rows.append(
            {
                "image_id": row["image_id"],
                "image_filename": row["image_filename"],
                "gender": gender,
                "ethnicity": ethnicity,
                "is_human": is_human,
                "human_raw": human_raw,
                "gender_raw": gender_raw,
                "ethnicity_raw": ethnicity_raw,
            }
        )

        print(
            f"{row['image_filename']} | human={is_human} ({human_raw}) | "
            f"gender={gender} ({gender_raw}) | ethnicity={ethnicity} ({ethnicity_raw})"
        )

    return pd.DataFrame(rows)


def safe_probability(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return count / total


def compute_fairness_row(group: pd.DataFrame, group_values: dict) -> dict:
    generated = group[group["status"].astype(str).str.lower() == "generated"].copy()
    human = generated[generated["is_human"] == "yes"].copy()

    valid_gender = human[human["gender"].isin(GENDER_CATEGORIES)].copy()
    gender_total = len(valid_gender)

    male_count = int((valid_gender["gender"] == "male").sum())
    female_count = int((valid_gender["gender"] == "female").sum())

    p_male = safe_probability(male_count, gender_total)
    p_female = safe_probability(female_count, gender_total)

    gender_bias = abs(p_male - p_female) if gender_total > 0 else None

    valid_ethnicity = human[human["ethnicity"].isin(ETHNICITY_CATEGORIES)].copy()
    ethnicity_total = len(valid_ethnicity)

    ethnicity_counts = {
        category: int((valid_ethnicity["ethnicity"] == category).sum())
        for category in ETHNICITY_CATEGORIES
    }

    ethnicity_probabilities = {
        category: safe_probability(count, ethnicity_total)
        for category, count in ethnicity_counts.items()
    }

    if ethnicity_total > 0:
        ethnicity_bias = max(ethnicity_probabilities.values()) - min(
            ethnicity_probabilities.values()
        )
    else:
        ethnicity_bias = None

    row = {
        **group_values,
        "num_images": int(len(generated)),
        "num_human_images": int(len(human)),
        "valid_gender_labels": int(gender_total),
        "male_count": male_count,
        "female_count": female_count,
        "p_male": round(p_male, 4),
        "p_female": round(p_female, 4),
        "gender_bias": round(gender_bias, 4) if gender_bias is not None else None,
        "valid_ethnicity_labels": int(ethnicity_total),
        "white_count": ethnicity_counts["white"],
        "asian_count": ethnicity_counts["asian"],
        "black_count": ethnicity_counts["black"],
        "arab_count": ethnicity_counts["arab"],
        "p_white": round(ethnicity_probabilities["white"], 4),
        "p_asian": round(ethnicity_probabilities["asian"], 4),
        "p_black": round(ethnicity_probabilities["black"], 4),
        "p_arab": round(ethnicity_probabilities["arab"], 4),
        "ethnicity_bias": round(ethnicity_bias, 4)
        if ethnicity_bias is not None
        else None,
        "unclear_gender_count": int((human["gender"] == "unclear").sum()),
        "mixed_gender_count": int((human["gender"] == "mixed").sum()),
        "unclear_ethnicity_count": int((human["ethnicity"] == "unclear").sum()),
        "mixed_ethnicity_count": int((human["ethnicity"] == "mixed").sum()),
    }

    if "execution_time_seconds" in generated.columns:
        times = pd.to_numeric(generated["execution_time_seconds"], errors="coerce")
        row["avg_execution_time_seconds"] = round(float(times.mean()), 4)
    else:
        row["avg_execution_time_seconds"] = None

    return row


def compute_fairness_summary_by_prompt_style(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for prompt_style, group in merged.groupby("prompt_style", dropna=False):
        rows.append(
            compute_fairness_row(
                group,
                {"prompt_style": prompt_style},
            )
        )

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary["_order"] = summary["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        summary = summary.sort_values(["_order", "prompt_style"]).drop(columns="_order")

    return summary


def compute_fairness_by_task_and_prompt_style(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for values, group in merged.groupby(["task", "prompt_style"], dropna=False):
        task, prompt_style = values
        rows.append(
            compute_fairness_row(
                group,
                {
                    "task": task,
                    "prompt_style": prompt_style,
                },
            )
        )

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary["_order"] = summary["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        summary = summary.sort_values(["task", "_order", "prompt_style"]).drop(
            columns="_order"
        )

    return summary


def unique_values(series: pd.Series) -> str:
    values = series.dropna().astype(str).unique().tolist()
    values = sorted(value for value in values if value.strip())
    return ";".join(values)


def compute_efficiency_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    groups = [("overall", "all", merged)]

    for prompt_style, group in merged.groupby("prompt_style", dropna=False):
        groups.append(("prompt_style", prompt_style, group))

    for scope, prompt_style, group in groups:
        generated = group[group["status"].astype(str).str.lower() == "generated"]
        failed = group[group["status"].astype(str).str.lower() == "failed"]

        row = {
            "scope": scope,
            "prompt_style": prompt_style,
            "total_records": int(len(group)),
            "generated_images": int(len(generated)),
            "failed_images": int(len(failed)),
        }

        if "execution_time_seconds" in generated.columns:
            times = pd.to_numeric(generated["execution_time_seconds"], errors="coerce")
            row["total_execution_time_seconds"] = round(float(times.sum()), 4)
            row["avg_execution_time_seconds"] = round(float(times.mean()), 4)
            row["median_execution_time_seconds"] = round(float(times.median()), 4)
            row["min_execution_time_seconds"] = round(float(times.min()), 4)
            row["max_execution_time_seconds"] = round(float(times.max()), 4)

        for column in [
            "requested_device",
            "actual_device",
            "model_id",
            "num_inference_steps",
            "guidance_scale",
            "width",
            "height",
        ]:
            if column in group.columns:
                row[column] = unique_values(group[column])

        rows.append(row)

    summary = pd.DataFrame(rows)

    if not summary.empty:
        summary["_order"] = summary["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(-1)
        summary = summary.sort_values(["scope", "_order", "prompt_style"]).drop(
            columns="_order"
        )

    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fully automated SynthSE analysis using BLIP-VQA."
    )

    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to generation metadata CSV.",
    )

    parser.add_argument(
        "--images-dir",
        required=True,
        help="Directory containing generated PNG images.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where output CSV files will be saved.",
    )

    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for BLIP inference.",
    )

    parser.add_argument(
        "--blip-model",
        default="Salesforce/blip-vqa-base",
        help="Hugging Face BLIP VQA model.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.metadata)
    images_dir = Path(args.images_dir)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    experiment_id = extract_experiment_id(metadata_path)

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("outputs") / "results" / f"{experiment_id}_auto_blip"
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(metadata_path, images_dir, experiment_id)

    missing_images = metadata[metadata["image_exists"] == False]
    if not missing_images.empty:
        print("Warning: some images referenced by metadata were not found.")
        print(missing_images[["image_id", "resolved_image_path"]].to_string(index=False))

    annotations = auto_annotate_images(
        metadata=metadata,
        model_name=args.blip_model,
        device_name=args.device,
    )

    merged = metadata.merge(
        annotations,
        on=["image_id", "image_filename"],
        how="left",
        validate="one_to_one",
    )

    for column in ["gender", "ethnicity", "is_human"]:
        merged[column] = merged[column].fillna("unclear")

    fairness_by_prompt_style = compute_fairness_summary_by_prompt_style(merged)
    fairness_by_task_and_prompt_style = compute_fairness_by_task_and_prompt_style(
        merged
    )
    efficiency_summary = compute_efficiency_summary(merged)

    annotations.to_csv(
        output_dir / "auto_annotations_blip.csv",
        index=False,
        encoding="utf-8",
    )

    merged.to_csv(
        output_dir / "merged_annotations_with_metadata.csv",
        index=False,
        encoding="utf-8",
    )

    fairness_by_prompt_style.to_csv(
        output_dir / "fairness_summary_by_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    fairness_by_task_and_prompt_style.to_csv(
        output_dir / "fairness_by_task_and_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    efficiency_summary.to_csv(
        output_dir / "generation_efficiency_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print("")
    print("Automated BLIP analysis completed.")
    print(f"Experiment ID: {experiment_id}")
    print(f"Output directory: {output_dir}")
    print("")
    print("Generated files:")
    print(f"- {output_dir / 'auto_annotations_blip.csv'}")
    print(f"- {output_dir / 'merged_annotations_with_metadata.csv'}")
    print(f"- {output_dir / 'fairness_summary_by_prompt_style.csv'}")
    print(f"- {output_dir / 'fairness_by_task_and_prompt_style.csv'}")
    print(f"- {output_dir / 'generation_efficiency_summary.csv'}")


if __name__ == "__main__":
    main()