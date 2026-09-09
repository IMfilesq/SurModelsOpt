import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.filter import filter_data
from src.data.loader import load_raw_data
from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.simulation.simulation  import simulate

logger = logging.getLogger(__name__)
@hydra.main(config_path="config",
            config_name="config",
            version_base=None)

def main(cfg: DictConfig) -> None:
# loading data
    df = load_raw_data(cfg.data.data_path)
    logger.info("Data loaded successfully.")

    initial_series_count = df["series"].nunique()
    logger.info(f"Data loaded successfully. Total rows: {len(df)}, Unique series: {initial_series_count}.")
    logger.info("Filtering data based on clinical constraints...")
    filtered_df = filter_data(
        df=df,
        min_single_dose=cfg.optimizer.boundaries.min_single_dose,
        max_single_dose=cfg.optimizer.boundaries.max_single_dose,
        max_total_dose=cfg.optimizer.boundaries.max_total_dose,
        min_interval=cfg.optimizer.boundaries.min_interval,
        max_interval=cfg.optimizer.boundaries.max_interval,
    )
    remaining_series_count = filtered_df["series"].nunique()
    logger.info(
        f"Filtering complete. Retained {remaining_series_count}/{initial_series_count} series "
        f"({len(filtered_df)} rows remaining)."
    )


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
    print("launching optimization:")
    minimum = optimizer.minimize()
    print("minumum", minimum)
    print("running simulation")
    simulated = simulate(tuple_protocol=[(1,1)],
                         params_file = cfg.simulation.params_file,
                         tumor_file = cfg.simulation.tumor_file)
    print("simulated : ", simulated)



if __name__ == "__main__":
    main()