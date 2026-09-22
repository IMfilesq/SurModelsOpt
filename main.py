import logging
from pathlib import Path
import matplotlib.pyplot as plt

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.data.filter import filter_data
from src.data.loader import load_raw_data
from src.models.base_model import BaseModel
from src.optimization.base_optimizer import BaseOptimizer
from src.simulation.simulation import simulate
from src.utils.check_bounds import check_bounds
from src.utils.reporter import generate_report
from src.data.analyzer import analyze_data

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

    analysis = analyze_data(filtered_df)
    output_dir = Path(HydraConfig.get().runtime.output_dir)

    if analysis.dose_histogram:
        dose_hist_path = output_dir / "dose_histogram.png"
        analysis.dose_histogram.savefig(dose_hist_path)
        plt.close(analysis.dose_histogram)  # zwolnienie pamięci
        logger.info(f"Saved dose histogram to {dose_hist_path}")

    if analysis.cancer_cells_histogram:
        cancer_hist_path = output_dir / "cancer_cells_histogram.png"
        analysis.cancer_cells_histogram.savefig(cancer_hist_path)
        plt.close(analysis.cancer_cells_histogram)  # zwolnienie pamięci
        logger.info(f"Saved cancer cells histogram to {cancer_hist_path}")

    if analysis.total_dose_histogram:
        total_hist_path = output_dir / "total_dose.png"
        analysis.total_dose_histogram.savefig(total_hist_path)
        plt.close(analysis.total_dose_histogram)
        logger.info(f"Saved total dose histogram to {total_hist_path}")


    logger.info("Instantiating model from config")
    model : BaseModel = instantiate(cfg.model)


    logger.info("Instantiating optimizer")
    optimizer: BaseOptimizer = instantiate(cfg.optimizer, model=model)

    logger.info("Seeking for optimal protocol")
    opt_result = optimizer.minimize()

    logger.info("Running cancer growth simulation for found protocol")
    simulated = simulate(protocol= opt_result.min_protocol,
                         params_file = cfg.simulation.params_file,
                         tumor_file = cfg.simulation.tumor_file)
    print("simulated : ", simulated)

    output_dir = HydraConfig.get().runtime.output_dir
    report_path = Path(output_dir) / "run_report.html"
    logger.info(f"Creating report at: {report_path}")

    generate_report(model_name = model.model_name,
                     opt_result=opt_result,
                     sim_val = simulated,
                     boundaries = cfg.optimizer.boundaries,
                     bounds_check = check_bounds(opt_result.min_protocol),
                     filename=str(report_path))




if __name__ == "__main__":
    main()