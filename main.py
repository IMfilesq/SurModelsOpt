import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.loader import load_raw_data
from src.models.base_model import BaseModel
from optimization.base_optimizer import BaseOptimizer

logger = logging.getLogger(__name__)
@hydra.main(config_path="config",
            config_name="config",
            version_base=None)

def main(cfg: DictConfig) -> None:
# loading data
    df = load_raw_data()
    print("Data loaded successfully.")
    print(f"DataFrame dimensions (rows, columns): {df.shape}")

    logger.info("Instantiating model from config")
    model : BaseModel = instantiate(cfg.model)
    logger.info("Running test prediction")
    raw_protocol = np.array([
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 2.0, 1.0],
            [4.0, 2.0, 1.0],
            [5.0, 0.0, 1.0],
            [6.0, 2.0, 1.0],
            [7.0, 2.0, 1.0],
            [8.0, 0.0, 1.0],
            [9.0, 2.0, 1.0],
            [10.0, 2.0, 1.0],
            [11.0, 0.0, 1.0],
            [12.0, 2.0, 1.0],
            [13.0, 2.0, 1.0],
            [14.0, 0.0, 1.0],
            [15.0, 2.0, 1.0],
            [16.0, 2.0, 1.0],
            [17.0, 0.0, 1.0],
            [18.0, 2.0, 1.0],
            [19.0, 2.0, 1.0],
        ], dtype=np.float64)
    
    prediction = model.predict(raw_protocol)
    print("Wynik predykcji:", prediction)

    print("instantiating optimizer")
    optimizer : BaseOptimizer = instantiate(cfg.optimizer,
                                            model = model)
    print("minimization result:")
    print(optimizer.minimize())


if __name__ == "__main__":
    main()