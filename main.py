import hydra
import logging
import pandas as pd
from omegaconf import DictConfig
from src.data.loader import load_raw_data


logger = logging.getLogger(__name__)
@hydra.main(config_path="config",
            config_name="config",
            version_base=None)

def main(cfg: DictConfig) -> None:
    logger.info("Reading data from csv file")
    raw = pd.read_csv(cfg.data.data_path)

# loading data
    df = load_raw_data()
    print("Data loaded successfully.")
    print(f"DataFrame dimensions (rows, columns): {df.shape}")

if __name__ == "__main__":
    main()