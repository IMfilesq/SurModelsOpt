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

    #logger.info("Loading raw traing data")
    # raw = load_raw_data()
    #logger.info("Filtering out data outside of optimization boundaries")
    # filtered = filter(raw, cfg.optimization.boundaries)
    #logger.info("Analyzing leftover data")
    # analysis = analyze(filtered)

    logger.info("Instantiating model from config")
    model : BaseModel = instantiate(cfg.model)


    logger.info("Instantiating optimizer")
    optimizer: BaseOptimizer = instantiate(cfg.optimizer, model=model)

    logger.info("Seeking for optimal protocol")
    min_result = optimizer.minimize()
    min_at = Converter.flat_to_tuples(min_result.x)
    print("minumum at ", min_at)
    logger.info("Running cancer growth simulation for found protocol")
    simulated = simulate(protocol= min_at,
                         params_file = cfg.simulation.params_file,
                         tumor_file = cfg.simulation.tumor_file)
    print("simulated : ", simulated)



if __name__ == "__main__":
    main()