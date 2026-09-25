# SurModelsOpt

**Optimal Cancer Treatment Protocols via Neural Network Surface Optimization**

Research repository dedicated to finding optimal cancer treatment protocols. Optimizers search for them on the surfaces of neural networks trained on data generated from cancer growth simulations.

**Main optimization pipeline:**

```
Model → Optimizer → Simulator (validation) + Dataset analysis → Report
```

---

## Authors

- **Igor Misterowicz**
- **Marta Hałas**

**Supervisor:** dr Paweł Gora

Created during the **QuantumAI 2026 Summer Internship**.

---

## Related Work & Credits

| Component              | Repository / Reference                                                                 | Authors                                                                 |
|------------------------|----------------------------------------------------------------------------------------|-------------------------------------------------------------------------|
| Simulation             | [EMT6-Ro](https://github.com/banasraf/EMT6-Ro.git)                                     | Paweł Gora, Rafał Banaś                                                 |
| Model architecture     | [CancerDLOptimization](https://github.com/AWarno/CancerDLOptimization)                 | Ania Warno                                                              |
| Dataset                | Generated for Bachelor’s Thesis                                                        | Antoni Goldstein, Michał Kardaś, Piotr Kowalkowski, Magdalena Molenda, Łukasz Piekarski – *A machine learning approach for cancer treatment optimization* |

---

## Installation & Usage

### Local run (without simulation)

The simulation module requires CUDA, GPU, and `pybind`. You can still run the main pipeline locally:

```bash
git lfs install
git clone https://github.com/IMfilesq/SurModelsOpt.git
cd SurModelsOpt
pip install uv
uv sync
uv run main.py
```

### Full pipeline with simulation (recommended)

To utilize the simulation module, open and run the provided notebook in **Google Colab**:

- File: `collab.ipynb`
- The notebook contains the complete configuration needed to execute `main.py` in a Colab environment with GPU support.

---

## Project Structure Overview

```
model → optimizer → simulator (validation) +  dataset analysis → report
```

1. **Dataset analysis** – exploratory analysis of simulation-generated data that satisfies optimization constraints  
2. **Model** – neural network trained on the dataset  
3. **Optimizer** – searches for optimal treatment protocols on the model surface  
4. **Simulator** – validates candidate protocols (requires GPU/CUDA)  
5. **Report** – presents the run information 

---

## License & Acknowledgments

This project was developed as part of the QuantumAI 2026 Summer Internship under the supervision of dr Paweł Gora.

Special thanks to the authors of the simulation engine, model architecture, and the original dataset.