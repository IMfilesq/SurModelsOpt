import os
import numpy as np
from src.schemas.protocols import TupleProtocol

def simulate(protocol: TupleProtocol,
             params_file: str = "/content/EMT6-Ro/data/default-parameters.json",
             tumor_file: str = "/content/EMT6-Ro/data/test_tumor.txt") -> float:
    try:
        import emt6ro.simulation as emt # type: ignore

        if not os.path.exists(params_file) or not os.path.exists(tumor_file):
            raise FileNotFoundError("Brak plików konfiguracyjnych symulatora.")

        params = emt.load_parameters(params_file)
        tumor_state = emt.load_state(tumor_file, params)

        # Inicjalizacja dla 1 protokołu
        exp = emt.Experiment(params, [tumor_state], 50, 1)

        # Formatowanie i opakowanie w dodatkową listę (lista protokołów)
        formatted_protocol = [(int(round(t)), float(d)) for t, d in protocol]
        exp.add_irradiations([formatted_protocol]) 

        exp.run(144000)

        results = exp.get_results()
        return float(np.mean(results[0, 0, :]))

    except (ImportError, Exception) as e:
        print("Unable to find emt6ro simulation package, try to run collab.ipynb in google collab envirnoment. Returning dummy value")
        return -999.0