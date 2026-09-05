import hydra
import logging
import pandas as pd
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
@hydra.main(config_path="config",
            config_name="config",
            version_base=None)

def main(cfg: DictConfig) -> None:
    logger.info("Reading data from csv file")
    raw = pd.read_csv(cfg.data.data_path)


if __name__ == "__main__":
    main()
