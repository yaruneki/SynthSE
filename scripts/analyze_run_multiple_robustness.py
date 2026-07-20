from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd


GENDER_CATEGORIES = ["male", "female"]
ETHNICITY_CATEGORIES = ["white", "asian", "black", "arab"]
PROMPT_STYLE_ORDER = {"general": 0, "se": 1, "fair": 2}


def safe_probability(count: int, total: int) -> float:
    return 0.0 if total == 0 else count / total


def entropy_from_counts(counts: list[int]) -> float:
    total = sum(counts)

    if total == 0:
        return 0.0

    entropy = 0.0

    for count in counts:
        if count == 0:
            continue

        probability = count / total
        entropy -= probability * math.log2(probability)

    return entropy


def normalized_entropy(counts: list[int]) -> float:
    nonzero_categories = len(counts)

    if nonzero_categories <= 1:
        return 0.0

    max_entropy = math.log2(nonzero_categories)

    if max_entropy == 0:
        return 0.0

    return entropy_from_counts(counts) / max_entropy


def majority_label_and_rate(series: pd.Series, valid_categories: list[str]) -> tuple[str, float]:
    valid = series[series.isin(valid_categories)]

    if valid.empty:
        return "none", 0.0

    counts = valid.value_counts()
    majority_label = str(counts.idxmax())
    majority_rate = counts.max() / len(valid)

    return majority_label, majority_rate


def compute_prompt_run_summary(group: pd.DataFrame, group_values: dict) -> dict:
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

    gender_majority_label, gender_majority_rate = majority_label_and_rate(
        human["gender"],
        GENDER_CATEGORIES,
    )

    gender_entropy = normalized_entropy([male_count, female_count])

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

    ethnicity_majority_label, ethnicity_majority_rate = majority_label_and_rate(
        human["ethnicity"],
        ETHNICITY_CATEGORIES,
    )

    ethnicity_entropy = normalized_entropy(
        [ethnicity_counts[category] for category in ETHNICITY_CATEGORIES]
    )

    seeds = (
        generated["seed"].dropna().astype(str).unique().tolist()
        if "seed" in generated.columns
        else []
    )

    return {
        **group_values,
        "num_runs": int(len(generated)),
        "num_human_images": int(len(human)),
        "seeds": ";".join(sorted(seeds)),
        "valid_gender_labels": int(gender_total),
        "male_count": male_count,
        "female_count": female_count,
        "p_male": round(p_male, 4),
        "p_female": round(p_female, 4),
        "gender_bias": round(gender_bias, 4) if gender_bias is not None else None,
        "gender_majority_label": gender_majority_label,
        "gender_majority_rate": round(float(gender_majority_rate), 4),
        "gender_entropy": round(float(gender_entropy), 4),
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
        "ethnicity_majority_label": ethnicity_majority_label,
        "ethnicity_majority_rate": round(float(ethnicity_majority_rate), 4),
        "ethnicity_entropy": round(float(ethnicity_entropy), 4),
        "unclear_gender_count": int((human["gender"] == "unclear").sum()),
        "unclear_ethnicity_count": int((human["ethnicity"] == "unclear").sum()),
    }


def compute_by_prompt(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_columns = ["prompt_id", "task", "prompt_style"]

    for values, group in merged.groupby(group_columns, dropna=False):
        prompt_id, task, prompt_style = values

        rows.append(
            compute_prompt_run_summary(
                group,
                {
                    "prompt_id": prompt_id,
                    "task": task,
                    "prompt_style": prompt_style,
                },
            )
        )

    result = pd.DataFrame(rows)

    if not result.empty:
        result["_order"] = result["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        result = result.sort_values(["task", "_order", "prompt_id"]).drop(
            columns="_order"
        )

    return result


def compute_by_prompt_style(prompt_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for prompt_style, group in prompt_summary.groupby("prompt_style", dropna=False):
        row = {
            "prompt_style": prompt_style,
            "num_prompts": int(len(group)),
            "total_runs": int(group["num_runs"].sum()),
            "avg_gender_bias": round(float(group["gender_bias"].mean()), 4),
            "std_gender_bias": round(float(group["gender_bias"].std(ddof=0)), 4),
            "avg_ethnicity_bias": round(float(group["ethnicity_bias"].mean()), 4),
            "std_ethnicity_bias": round(float(group["ethnicity_bias"].std(ddof=0)), 4),
            "avg_gender_majority_rate": round(
                float(group["gender_majority_rate"].mean()), 4
            ),
            "avg_ethnicity_majority_rate": round(
                float(group["ethnicity_majority_rate"].mean()), 4
            ),
            "avg_gender_entropy": round(float(group["gender_entropy"].mean()), 4),
            "avg_ethnicity_entropy": round(float(group["ethnicity_entropy"].mean()), 4),
        }

        rows.append(row)

    result = pd.DataFrame(rows)

    if not result.empty:
        result["_order"] = result["prompt_style"].map(PROMPT_STYLE_ORDER).fillna(999)
        result = result.sort_values(["_order", "prompt_style"]).drop(columns="_order")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze robustness across multiple generation runs with different seeds."
    )

    parser.add_argument(
        "--merged",
        required=True,
        help="Path to merged_annotations_with_metadata.csv produced by BLIP analysis.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where run-multiple robustness CSV files will be saved.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    merged_path = Path(args.merged)
    output_dir = Path(args.output_dir)

    if not merged_path.exists():
        raise FileNotFoundError(f"Merged annotations file not found: {merged_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    merged = pd.read_csv(merged_path)

    prompt_summary = compute_by_prompt(merged)
    prompt_style_summary = compute_by_prompt_style(prompt_summary)

    prompt_summary.to_csv(
        output_dir / "run_multiple_robustness_by_prompt.csv",
        index=False,
        encoding="utf-8",
    )

    prompt_style_summary.to_csv(
        output_dir / "run_multiple_robustness_by_prompt_style.csv",
        index=False,
        encoding="utf-8",
    )

    print("Run multiple robustness analysis completed.")
    print(f"Output directory: {output_dir}")
    print("")
    print("Generated files:")
    print(f"- {output_dir / 'run_multiple_robustness_by_prompt.csv'}")
    print(f"- {output_dir / 'run_multiple_robustness_by_prompt_style.csv'}")


if __name__ == "__main__":
    main()