import os
import sys

if "ipykernel" not in sys.modules and os.environ.get("MPLBACKEND") == "module://matplotlib_inline.backend_inline":
    os.environ.pop("MPLBACKEND", None)


import logging
from pathlib import Path
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.analyzer import analyze_data
from src.data.filter import filter_data
from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.simulation.simulation import simulate
from src.utils.check_bounds import check_bounds
from src.utils.converter import Converter
from src.utils.reporter import generate_report



logger = logging.getLogger(__name__)
@hydra.main(config_path="config",
            config_name="config",
            version_base=None)

def main(cfg: DictConfig) -> None:
    """
    Runs the whole analysis -> optimization -> simulation -> reporting pipeline.
    """
    logger.info("loading raw training data")
    df = pd.read_csv(cfg.data.data_path)

    logger.info("filtering raw data with respect to constrains")
    filtered_df = filter_data(df=df,
                              boundaries = cfg.optimizer.boundaries)
    
    logger.info("analyzing filtered dataset")
    analysis_result = analyze_data(filtered_df)

    logger.info("Instantiating model from config")
    model : BaseModel = instantiate(cfg.model)

    logger.info("Creating bounds stricter by safety_eps")
    stricter_bounds = Converter.make_stricter(bounds = instantiate(cfg.optimizer.boundaries),
                                              safety_eps = cfg.safety_eps)

    logger.info("Instantiating optimizer")
    optimizer: BaseOptimizer = instantiate(cfg.optimizer,
                                           model = model,
                                           boundaries = stricter_bounds)

    logger.info("Seeking for optimal protocol")
    opt_result = optimizer.minimize()

    logger.info("Running cancer growth simulation for found protocol")
    simulated = simulate(protocol= opt_result.min_protocol,
                         params_file = cfg.simulation.params_file,
                         tumor_file = cfg.simulation.tumor_file)

    logger.info("Acessing hydra output path")
    output_dir = HydraConfig.get().runtime.output_dir
    report_path = Path(output_dir) / "run_report.html"

    logger.info(f"Creating report at: {report_path}")
    generate_report(model_name = model.model_name,
                     opt_result= opt_result,
                     sim_val = simulated,
                     boundaries = cfg.optimizer.boundaries,
                     bounds_check = check_bounds(opt_result.min_protocol),
                     analysis_result = analysis_result,
                     filename=str(report_path))




if __name__ == "__main__":
    main()