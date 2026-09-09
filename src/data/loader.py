from pathlib import Path
import pandas as pd


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at: {path.resolve()}")

    return pd.read_csv(path)
