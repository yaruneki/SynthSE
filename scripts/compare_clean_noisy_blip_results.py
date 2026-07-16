from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


GENDER_CATEGORIES = ["male", "female"]
ETHNICITY_CATEGORIES = ["white", "asian", "black", "arab"]
PROMPT_STYLE_ORDER = {"general": 0, "se": 1, "fair": 2}


def infer_source_image_id(noisy_image_id: str) -> str:
    return str(noisy_image_id).split("__")[0]


def safe_probability(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def compute_fairness_row(group: pd.DataFrame, group_values: dict) -> dict:
    if "status" in group.columns:
        generated = group[group["status"].astype(str).str.lower() == "generated"].copy()
    else:
        generated = group.copy()

    human = generated[generated["is_human"].astype(str).str.lower() == "yes"].copy()

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

    ethnicity_bias = (
        max(ethnicity_probabilities.values()) - min(ethnicity_probabilities.values())
        if ethnicity_total > 0
        else None
    )

    return {
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
        "ethnicity_bias": round(ethnicity_bias, 4) if ethnicity_bias is not None else None,
        "unclear_gender_count": int((human["gender"] == "unclear").sum()),
        "unclear_ethnicity_count": int((human["ethnicity"] == "unclear").sum()),
    }


def compute_clean_fairness(clean: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for prompt_style, group in clean.groupby("prompt_style", dropna=False):
        rows.append(
            compute_fairness_row(
                group,
                {"prompt_style": prompt_style},
            )
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["_order"] = result["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        result = result.sort_values(["_order", "prompt_style"]).drop(columns="_order")

    return result


def compute_noisy_fairness(noisy: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for values, group in noisy.groupby(
        ["noise_type", "noise_level", "prompt_style"],
        dropna=False,
    ):
        noise_type, noise_level, prompt_style = values

        rows.append(
            compute_fairness_row(
                group,
                {
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "prompt_style": prompt_style,
                },
            )
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["_order"] = result["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        result = result.sort_values(
            ["noise_type", "noise_level", "_order", "prompt_style"]
        ).drop(columns="_order")

    return result


def compute_fairness_delta(
    clean_fairness: pd.DataFrame,
    noisy_fairness: pd.DataFrame,
) -> pd.DataFrame:
    clean_subset = clean_fairness[
        [
            "prompt_style",
            "gender_bias",
            "ethnicity_bias",
            "valid_gender_labels",
            "valid_ethnicity_labels",
        ]
    ].rename(
        columns={
            "gender_bias": "clean_gender_bias",
            "ethnicity_bias": "clean_ethnicity_bias",
            "valid_gender_labels": "clean_valid_gender_labels",
            "valid_ethnicity_labels": "clean_valid_ethnicity_labels",
        }
    )

    noisy_subset = noisy_fairness.rename(
        columns={
            "gender_bias": "noisy_gender_bias",
            "ethnicity_bias": "noisy_ethnicity_bias",
            "valid_gender_labels": "noisy_valid_gender_labels",
            "valid_ethnicity_labels": "noisy_valid_ethnicity_labels",
        }
    )

    merged = noisy_subset.merge(clean_subset, on="prompt_style", how="left")

    merged["delta_gender_bias"] = (
        merged["noisy_gender_bias"] - merged["clean_gender_bias"]
    ).round(4)

    merged["delta_ethnicity_bias"] = (
        merged["noisy_ethnicity_bias"] - merged["clean_ethnicity_bias"]
    ).round(4)

    return merged


def prepare_paired_annotations(clean: pd.DataFrame, noisy: pd.DataFrame) -> pd.DataFrame:
    clean_subset = clean[
        [
            "image_id",
            "prompt_style",
            "task",
            "gender",
            "ethnicity",
            "is_human",
        ]
    ].rename(
        columns={
            "image_id": "source_image_id",
            "prompt_style": "clean_prompt_style",
            "task": "clean_task",
            "gender": "clean_gender",
            "ethnicity": "clean_ethnicity",
            "is_human": "clean_is_human",
        }
    )

    if "source_image_id" not in noisy.columns:
        noisy["source_image_id"] = noisy["image_id"].apply(infer_source_image_id)

    noisy_subset = noisy[
        [
            "image_id",
            "source_image_id",
            "prompt_style",
            "task",
            "noise_type",
            "noise_level",
            "gender",
            "ethnicity",
            "is_human",
        ]
    ].rename(
        columns={
            "image_id": "corrupted_image_id",
            "prompt_style": "noisy_prompt_style",
            "task": "noisy_task",
            "gender": "noisy_gender",
            "ethnicity": "noisy_ethnicity",
            "is_human": "noisy_is_human",
        }
    )

    paired = noisy_subset.merge(
        clean_subset,
        on="source_image_id",
        how="left",
        validate="many_to_one",
    )

    paired["same_gender"] = paired["clean_gender"] == paired["noisy_gender"]
    paired["same_ethnicity"] = paired["clean_ethnicity"] == paired["noisy_ethnicity"]
    paired["same_is_human"] = paired["clean_is_human"] == paired["noisy_is_human"]

    paired["gender_changed_to_unclear"] = (
        paired["clean_gender"].isin(GENDER_CATEGORIES)
        & (paired["noisy_gender"] == "unclear")
    )

    paired["ethnicity_changed_to_unclear"] = (
        paired["clean_ethnicity"].isin(ETHNICITY_CATEGORIES)
        & (paired["noisy_ethnicity"] == "unclear")
    )

    return paired


def summarize_label_stability(paired: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_definitions = [
        ("overall", []),
        ("noise_level", ["noise_type", "noise_level"]),
        ("noise_level_prompt_style", ["noise_type", "noise_level", "noisy_prompt_style"]),
    ]

    for scope, columns in group_definitions:
        if columns:
            grouped = paired.groupby(columns, dropna=False)
        else:
            grouped = [((), paired)]

        for values, group in grouped:
            if not isinstance(values, tuple):
                values = (values,)

            row = {"scope": scope}

            for column, value in zip(columns, values):
                row[column.replace("noisy_", "")] = value

            row["total_pairs"] = int(len(group))
            row["same_gender_rate"] = round(float(group["same_gender"].mean()), 4)
            row["same_ethnicity_rate"] = round(float(group["same_ethnicity"].mean()), 4)
            row["same_is_human_rate"] = round(float(group["same_is_human"].mean()), 4)

            row["gender_changed_to_unclear_rate"] = round(
                float(group["gender_changed_to_unclear"].mean()), 4
            )
            row["ethnicity_changed_to_unclear_rate"] = round(
                float(group["ethnicity_changed_to_unclear"].mean()), 4
            )

            row["changed_gender_count"] = int((~group["same_gender"]).sum())
            row["changed_ethnicity_count"] = int((~group["same_ethnicity"]).sum())
            row["changed_is_human_count"] = int((~group["same_is_human"]).sum())

            rows.append(row)

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare clean and noisy BLIP annotations for SynthSE experiments."
    )

    parser.add_argument(
        "--clean-merged",
        required=True,
        help="Path to clean merged_annotations_with_metadata.csv.",
    )

    parser.add_argument(
        "--noisy-merged",
        required=True,
        help="Path to noisy merged_annotations_with_metadata.csv.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where comparison CSV files will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    clean_path = Path(args.clean_merged)
    noisy_path = Path(args.noisy_merged)
    output_dir = Path(args.output_dir)

    if not clean_path.exists():
        raise FileNotFoundError(f"Clean merged file not found: {clean_path}")

    if not noisy_path.exists():
        raise FileNotFoundError(f"Noisy merged file not found: {noisy_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    clean = pd.read_csv(clean_path)
    noisy = pd.read_csv(noisy_path)

    if "source_image_id" not in noisy.columns:
        noisy["source_image_id"] = noisy["image_id"].apply(infer_source_image_id)

    clean_fairness = compute_clean_fairness(clean)
    noisy_fairness = compute_noisy_fairness(noisy)
    fairness_delta = compute_fairness_delta(clean_fairness, noisy_fairness)

    paired = prepare_paired_annotations(clean, noisy)
    label_stability = summarize_label_stability(paired)

    clean_fairness.to_csv(
        output_dir / "clean_fairness_by_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    noisy_fairness.to_csv(
        output_dir / "noisy_fairness_by_noise_and_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    fairness_delta.to_csv(
        output_dir / "fairness_delta_by_noise_and_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    paired.to_csv(
        output_dir / "paired_clean_noisy_annotations.csv",
        index=False,
        encoding="utf-8",
    )

    label_stability.to_csv(
        output_dir / "label_stability_summary.csv",
        index=False,
        encoding="utf-8",
    )

    print("Clean vs noisy comparison completed.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Generated files:")
    print(f"- {output_dir / 'clean_fairness_by_prompt_style.csv'}")
    print(f"- {output_dir / 'noisy_fairness_by_noise_and_prompt_style.csv'}")
    print(f"- {output_dir / 'fairness_delta_by_noise_and_prompt_style.csv'}")
    print(f"- {output_dir / 'paired_clean_noisy_annotations.csv'}")
    print(f"- {output_dir / 'label_stability_summary.csv'}")


if __name__ == "__main__":
    main()