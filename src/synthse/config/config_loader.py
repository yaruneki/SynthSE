from pathlib import Path
import yaml


def load_config(config_path: str) -> dict:
    """
    Load the experiment configuration from a YAML file.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        raise ValueError(f"Configuration file is empty: {config_path}")

    return config