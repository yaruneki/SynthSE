from pathlib import Path

import pandas as pd


def save_metadata(records: list[dict], output_path: str) -> None:
    """
    Save experiment metadata records to a CSV file.

    Args:
        records: List of metadata dictionaries.
        output_path: Path where the CSV file will be saved.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(records)
    metadata.to_csv(path, index=False, encoding="utf-8")