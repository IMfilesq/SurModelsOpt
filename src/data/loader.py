from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_PATH = BASE_DIR / "data" / "raw.csv"

def load_raw_data(file_path: Path = DATA_PATH) -> pd.DataFrame:
    """Wczytuje surowe dane z pliku CSV."""
    if not file_path.exists():
        raise FileNotFoundError(f"Plik nie istnieje pod ścieżką: {file_path}")

    return pd.read_csv(file_path)

