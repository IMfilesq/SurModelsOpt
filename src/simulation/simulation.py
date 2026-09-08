import os
import numpy as np

def simulate(tuple_protocol: list[tuple[float, float]],
             params_file: str = "/content/EMT6-Ro/data/default-parameters.json",
             tumor_file: str = "/content/EMT6-Ro/data/test_tumor.txt") -> float:
    try:
        import emt6ro.simulation as emt

        if not os.path.exists(params_file) or not os.path.exists(tumor_file):
            raise FileNotFoundError("Brak plików konfiguracyjnych symulatora.")

        params = emt.load_parameters(params_file)
        tumor_state = emt.load_state(tumor_file, params)

        # Inicjalizacja dla 1 protokołu
        exp = emt.Experiment(params, [tumor_state], 50, 1)

        # Formatowanie i opakowanie w dodatkową listę (lista protokołów)
        formatted_protocol = [(int(round(t)), float(d)) for t, d in tuple_protocol]
        exp.add_irradiations([formatted_protocol]) 

        exp.run(144000)

        results = exp.get_results()
        return float(np.mean(results[0, 0, :]))

    except (ImportError, Exception) as e:
        print(f"⚠️ Nie można uruchomić symulacji C++/CUDA ({e}). Zwracanie wartości atrapy (350.0).")
        return 350.0