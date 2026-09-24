import logging
import os

import numpy as np

from src.schemas.protocols import TupleProtocol

logger = logging.getLogger(__name__)


def simulate(
    protocol: TupleProtocol,
    params_file: str = "/content/EMT6-Ro/data/default-parameters.json",
    tumor_file: str = "/content/EMT6-Ro/data/test_tumor.txt",
    n_sim: int = 50,
) -> float:
    """Runs the emt6ro simulation when the package is available.

    Use of colab.ipynb file in the Google Colab environment is advised.
    """
    try:
        import emt6ro.simulation as emt  # type: ignore
    except ImportError:
        logger.error(
            "Unable to find emt6ro simulation package (missing package), "
            "try to run in Google Colab environment. Returning dummy value."
        )
        return -999.0

    logger.info("Successfully imported emt6ro, running a simulation")

    if not os.path.exists(params_file) or not os.path.exists(tumor_file):
        raise FileNotFoundError(f"Missing config files: {params_file} or {tumor_file}")

    params = emt.load_parameters(params_file)
    tumor_state = emt.load_state(tumor_file, params)

    exp = emt.Experiment(params, [tumor_state], n_sim, 1)

    formatted_protocol = [(round(t), float(d)) for t, d in protocol]
    exp.add_irradiations([formatted_protocol])

    exp.run(144000)

    results = exp.get_results()
    return float(np.mean(results[0, 0, :]))