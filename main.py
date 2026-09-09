import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.filter import filter_data
from src.data.loader import load_raw_data
from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.simulation.simulation  import simulate
from src.utils.converter import Converter

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
    filtered_df = filter_data(df=df,
                              boundaries = cfg.optimizer.boundaries)
    remaining_series_count = filtered_df["series"].nunique()
    logger.info(
        f"Filtering complete. Retained {remaining_series_count}/{initial_series_count} series "
        f"({len(filtered_df)} rows remaining)."
    )

    logger.info("Instantiating model from config")
    model : BaseModel = instantiate(cfg.model)


    logger.info("Instantiating optimizer")
    optimizer: BaseOptimizer = instantiate(cfg.optimizer, model=model)

    logger.info("Seeking for optimal protocol")
    min_at, min_val = optimizer.minimize()
    print("min at", min_at)
    print("min val", min_val)
    logger.info("Running cancer growth simulation for found protocol")
    print("Tuple min protocol: ", Converter.matrix_to_tuples(min_at))
    simulated = simulate(protocol= Converter.matrix_to_tuples(min_at),
                         params_file = cfg.simulation.params_file,
                         tumor_file = cfg.simulation.tumor_file)
    print("simulated : ", simulated)



if __name__ == "__main__":
    main()