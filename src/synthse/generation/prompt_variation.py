from __future__ import annotations

import hashlib
import random
import re
from typing import Any

import pandas as pd


SUPPORTED_VARIATION_TYPES = {
    "typo_swap",
    "punctuation",
    "case",
    "article_drop",
    "adjacent_word_swap",
}


def build_stable_seed(
    base_seed: int,
    prompt_id: Any,
    variant_index: int,
    variation_type: str,
) -> int:
    """
    Build a deterministic seed for a prompt variation.

    The resulting seed depends on:
    - the base variation seed;
    - the source prompt identifier;
    - the variation index;
    - the variation type.

    SHA-256 is used instead of Python's hash() because hash() may
    produce different values in separate Python processes.
    """
    value = (
        f"{base_seed}|"
        f"{prompt_id}|"
        f"{variant_index}|"
        f"{variation_type}"
    )

    digest = hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()

    return int(digest[:8], 16)


def apply_typo_swap(
    prompt: str,
    random_generator: random.Random,
) -> str:
    """
    Swap two adjacent characters inside one eligible word.

    The first and last character of the selected word are preserved
    whenever possible.

    Example:
        Engineer -> Engiener
    """
    word_matches = list(
        re.finditer(
            r"\b[A-Za-z]{4,}\b",
            prompt,
        )
    )

    random_generator.shuffle(word_matches)

    for match in word_matches:
        word = match.group(0)

        possible_positions = [
            index
            for index in range(
                1,
                len(word) - 2,
            )
            if word[index] != word[index + 1]
        ]

        if not possible_positions:
            continue

        position = random_generator.choice(
            possible_positions
        )

        modified_word = (
            word[:position]
            + word[position + 1]
            + word[position]
            + word[position + 2:]
        )

        return (
            prompt[:match.start()]
            + modified_word
            + prompt[match.end():]
        )

    return prompt


def apply_punctuation_variation(
    prompt: str,
) -> str:
    """
    Add or remove final punctuation.

    If the prompt ends with punctuation, it is removed.
    Otherwise, a final period is added.
    """
    stripped_prompt = prompt.rstrip()

    if not stripped_prompt:
        return prompt

    if stripped_prompt[-1] in ".,;:!?":
        return stripped_prompt[:-1].rstrip()

    return f"{stripped_prompt}."


def apply_case_variation(
    prompt: str,
    random_generator: random.Random,
) -> str:
    """
    Change the capitalization of one randomly selected word.

    Example:
        Software -> software
        portrait -> Portrait
    """
    word_matches = list(
        re.finditer(
            r"\b[A-Za-z]{3,}\b",
            prompt,
        )
    )

    if not word_matches:
        return prompt

    match = random_generator.choice(
        word_matches
    )

    word = match.group(0)

    if word[0].isupper():
        modified_word = (
            word[0].lower()
            + word[1:]
        )
    else:
        modified_word = (
            word[0].upper()
            + word[1:]
        )

    return (
        prompt[:match.start()]
        + modified_word
        + prompt[match.end():]
    )


def apply_article_drop(
    prompt: str,
    random_generator: random.Random,
) -> str:
    """
    Remove one English article from the prompt.

    Supported articles:
    - a
    - an
    - the

    Example:
        a Software Engineer
        ->
        Software Engineer
    """
    article_matches = list(
        re.finditer(
            r"\b(?:a|an|the)\b\s*",
            prompt,
            flags=re.IGNORECASE,
        )
    )

    if not article_matches:
        return prompt

    match = random_generator.choice(
        article_matches
    )

    modified_prompt = (
        prompt[:match.start()]
        + prompt[match.end():]
    )

    modified_prompt = re.sub(
        r"\s{2,}",
        " ",
        modified_prompt,
    )

    return modified_prompt.strip()


def apply_adjacent_word_swap(
    prompt: str,
    random_generator: random.Random,
) -> str:
    """
    Swap two consecutive words in the prompt.

    Only words separated exclusively by whitespace are considered
    adjacent. Punctuation located between words is therefore not moved.

    Example:
        Software Engineer
        ->
        Engineer Software
    """
    word_matches = list(
        re.finditer(
            r"\b[A-Za-z]{2,}(?:'[A-Za-z]+)?\b",
            prompt,
        )
    )

    if len(word_matches) < 2:
        return prompt

    candidate_pairs: list[
        tuple[
            re.Match[str],
            re.Match[str],
            str,
        ]
    ] = []

    for index in range(
        len(word_matches) - 1
    ):
        first_match = word_matches[index]
        second_match = word_matches[index + 1]

        separator = prompt[
            first_match.end():
            second_match.start()
        ]

        if separator and separator.isspace():
            candidate_pairs.append(
                (
                    first_match,
                    second_match,
                    separator,
                )
            )

    if not candidate_pairs:
        return prompt

    first_match, second_match, separator = (
        random_generator.choice(
            candidate_pairs
        )
    )

    first_word = first_match.group(0)
    second_word = second_match.group(0)

    return (
        prompt[:first_match.start()]
        + second_word
        + separator
        + first_word
        + prompt[second_match.end():]
    )


def apply_prompt_variation(
    prompt: str,
    variation_type: str,
    variation_seed: int,
) -> str:
    """
    Apply one deterministic variation to a prompt.

    If the selected transformation cannot modify the prompt, a
    punctuation variation is used as a fallback.
    """
    normalized_type = (
        variation_type.strip().lower()
    )

    if normalized_type not in SUPPORTED_VARIATION_TYPES:
        raise ValueError(
            "Unsupported prompt variation type: "
            f"{variation_type}. "
            "Supported values: "
            f"{sorted(SUPPORTED_VARIATION_TYPES)}"
        )

    random_generator = random.Random(
        variation_seed
    )

    if normalized_type == "typo_swap":
        modified_prompt = apply_typo_swap(
            prompt=prompt,
            random_generator=random_generator,
        )

    elif normalized_type == "punctuation":
        modified_prompt = (
            apply_punctuation_variation(
                prompt
            )
        )

    elif normalized_type == "case":
        modified_prompt = apply_case_variation(
            prompt=prompt,
            random_generator=random_generator,
        )

    elif normalized_type == "article_drop":
        modified_prompt = apply_article_drop(
            prompt=prompt,
            random_generator=random_generator,
        )

    elif normalized_type == "adjacent_word_swap":
        modified_prompt = apply_adjacent_word_swap(
            prompt=prompt,
            random_generator=random_generator,
        )

    else:
        modified_prompt = prompt

    # Ensure that a requested variation is not identical to
    # the source prompt.
    if modified_prompt == prompt:
        modified_prompt = (
            apply_punctuation_variation(
                prompt
            )
        )

    return modified_prompt


def normalize_variation_types(
    variation_types: Any,
) -> list[str]:
    """
    Normalize and validate the configured variation types.
    """
    if variation_types is None:
        normalized_types = [
            "typo_swap",
            "punctuation",
            "case",
            "article_drop",
            "adjacent_word_swap",
        ]

    elif isinstance(
        variation_types,
        str,
    ):
        normalized_types = [
            variation_type.strip().lower()
            for variation_type
            in variation_types.split(",")
            if variation_type.strip()
        ]

    else:
        normalized_types = [
            str(variation_type)
            .strip()
            .lower()
            for variation_type
            in variation_types
            if str(variation_type).strip()
        ]

    if not normalized_types:
        raise ValueError(
            "At least one prompt variation type "
            "must be configured."
        )

    unsupported_types = (
        set(normalized_types)
        - SUPPORTED_VARIATION_TYPES
    )

    if unsupported_types:
        raise ValueError(
            "Unsupported prompt variation types: "
            f"{sorted(unsupported_types)}. "
            "Supported values: "
            f"{sorted(SUPPORTED_VARIATION_TYPES)}"
        )

    return normalized_types


def build_prompt_variants(
    original_prompt: str,
    prompt_id: Any,
    configuration: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """
    Build the original prompt and its dynamic variations.

    Returned fields:
    - original_prompt
    - modified_prompt
    - prompt_variation_type
    - prompt_variation_index
    - prompt_variation_seed
    - prompt_is_original
    """
    config = configuration or {}

    enabled = bool(
        config.get(
            "enabled",
            False,
        )
    )

    include_original = bool(
        config.get(
            "include_original",
            True,
        )
    )

    if not enabled:
        return [
            {
                "original_prompt": original_prompt,
                "modified_prompt": original_prompt,
                "prompt_variation_type": "none",
                "prompt_variation_index": 0,
                "prompt_variation_seed": None,
                "prompt_is_original": True,
            }
        ]

    variation_types = normalize_variation_types(
        config.get("types")
    )

    variants_per_prompt = int(
        config.get(
            "variants_per_prompt",
            len(variation_types),
        )
    )

    if variants_per_prompt < 0:
        raise ValueError(
            "prompt_variation.variants_per_prompt "
            "must be greater than or equal to zero."
        )

    if (
        variants_per_prompt == 0
        and not include_original
    ):
        raise ValueError(
            "Prompt variation configuration would "
            "produce no prompts. Enable include_original "
            "or request at least one variation."
        )

    base_seed = int(
        config.get(
            "seed",
            2026,
        )
    )

    variants: list[
        dict[str, Any]
    ] = []

    if include_original:
        variants.append(
            {
                "original_prompt": original_prompt,
                "modified_prompt": original_prompt,
                "prompt_variation_type": "original",
                "prompt_variation_index": 0,
                "prompt_variation_seed": None,
                "prompt_is_original": True,
            }
        )

    for variation_offset in range(
        variants_per_prompt
    ):
        variation_index = (
            variation_offset + 1
        )

        variation_type = variation_types[
            variation_offset
            % len(variation_types)
        ]

        variation_seed = build_stable_seed(
            base_seed=base_seed,
            prompt_id=prompt_id,
            variant_index=variation_index,
            variation_type=variation_type,
        )

        modified_prompt = apply_prompt_variation(
            prompt=original_prompt,
            variation_type=variation_type,
            variation_seed=variation_seed,
        )

        variants.append(
            {
                "original_prompt": original_prompt,
                "modified_prompt": modified_prompt,
                "prompt_variation_type": variation_type,
                "prompt_variation_index": variation_index,
                "prompt_variation_seed": variation_seed,
                "prompt_is_original": False,
            }
        )

    return variants


def is_preexpanded_prompt_dataframe(
    prompts: pd.DataFrame,
) -> bool:
    """
    Return True when the dataframe already contains prompt-variation
    metadata.

    This prevents the isolated runner from modifying an already
    modified prompt a second time.
    """
    required_columns = {
        "source_prompt_id",
        "prompt_variant_id",
        "original_prompt",
        "modified_prompt",
        "prompt_variation_type",
        "prompt_variation_index",
        "prompt_variation_seed",
        "prompt_is_original",
    }

    return required_columns.issubset(
        prompts.columns
    )


def expand_prompt_dataframe(
    prompts: pd.DataFrame,
    configuration: dict[str, Any] | None,
) -> pd.DataFrame:
    """
    Expand each source prompt into its original and modified versions.

    The original CSV row is preserved and enriched with prompt-variation
    metadata.

    The `prompt` field always contains the actual text sent to the
    image-generation pipeline.
    """
    if prompts.empty:
        return prompts.copy()

    if "prompt_id" not in prompts.columns:
        raise ValueError(
            "The prompt dataframe must contain "
            "the 'prompt_id' column."
        )

    if "prompt" not in prompts.columns:
        raise ValueError(
            "The prompt dataframe must contain "
            "the 'prompt' column."
        )

    if is_preexpanded_prompt_dataframe(
        prompts
    ):
        return prompts.copy()

    config = configuration or {}

    enabled = bool(
        config.get(
            "enabled",
            False,
        )
    )

    expanded_rows: list[
        dict[str, Any]
    ] = []

    for _, row in prompts.iterrows():
        row_dict = row.to_dict()

        source_prompt_id = row_dict[
            "prompt_id"
        ]

        original_prompt = str(
            row_dict["prompt"]
        )

        prompt_variants = build_prompt_variants(
            original_prompt=original_prompt,
            prompt_id=source_prompt_id,
            configuration=config,
        )

        for variant in prompt_variants:
            expanded_row = (
                row_dict.copy()
            )

            variation_index = int(
                variant[
                    "prompt_variation_index"
                ]
            )

            variation_type = str(
                variant[
                    "prompt_variation_type"
                ]
            )

            # Preserve the original prompt identifier when
            # prompt variation is disabled.
            if (
                not enabled
                and variation_type == "none"
            ):
                prompt_variant_id = str(
                    source_prompt_id
                )
            else:
                prompt_variant_id = (
                    f"{source_prompt_id}_"
                    f"pv{variation_index:02d}"
                )

            expanded_row.update(
                {
                    "source_prompt_id": (
                        source_prompt_id
                    ),
                    "prompt_variant_id": (
                        prompt_variant_id
                    ),
                    "original_prompt": (
                        variant[
                            "original_prompt"
                        ]
                    ),
                    "modified_prompt": (
                        variant[
                            "modified_prompt"
                        ]
                    ),
                    # This is the prompt actually passed
                    # to the diffusion pipeline.
                    "prompt": (
                        variant[
                            "modified_prompt"
                        ]
                    ),
                    "prompt_variation_type": (
                        variation_type
                    ),
                    "prompt_variation_index": (
                        variation_index
                    ),
                    "prompt_variation_seed": (
                        variant[
                            "prompt_variation_seed"
                        ]
                    ),
                    "prompt_is_original": (
                        variant[
                            "prompt_is_original"
                        ]
                    ),
                }
            )

            expanded_rows.append(
                expanded_row
            )

    return pd.DataFrame(
        expanded_rows
    )