import argparse
from pathlib import Path

import pandas as pd

from synthse.config.config_loader import load_config


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a SynthSE experiment.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML configuration file."
    )

    args = parser.parse_args()

    config = load_config(args.config)

    prompts_file = config["paths"]["prompts_file"]
    prompts = load_prompts(prompts_file)

    print("Configuration loaded successfully.")
    print(f"Project: {config['project']['name']}")
    print(f"Model: {config['model']['huggingface_id']}")
    print(f"Device: {config['model']['device']}")
    print(f"Prompts loaded: {len(prompts)}")

    print("\nPrompt preview:")
    print(prompts[["prompt_id", "task", "prompt"]].head())


if __name__ == "__main__":
    main()