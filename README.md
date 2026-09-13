# Coupled LSTM–rSAS model

Research code associated with **“Constrained Estimation of Export and Import Inter-basin Groundwater Flows by Coupling Deep Learning with a Water-Age Balance Approach”**, accepted in *Water Resources Research*.

This repository contains the Python source and cluster job script for the coupled model.

## Contents

- `main_LMCMA_HPC_1.py`: coupled optimization, validation, and predictive uncertainty workflow using LMCMA.
- `LSTM.py`: PyTorch model component.
- `model_data_es.py`: input preparation and model parameter/data container.
- `rsas.py` / `rsas_functions.py`: rSAS solver and functions. The supplied `rsas.py` credits Ciaran J. Harman; preserve third-party attribution and verify the applicable upstream license.
- `main_test_rsas.py`: separate rSAS diagnostic with specified physical parameters and zero inter-basin groundwater-flow terms; this is not a full coupled-model reproducibility test.
- `runjob`: original LSF cluster job example.

## Environment and execution

The code imports NumPy, pandas, SciPy, PyTorch, and `pypop7` (including `pypop7.optimizers.es.lmcma.LMCMA`). Python 3.10 is consistent with the supplied local cache filenames, but exact package versions from the original research environment were not supplied. Install these dependencies into an isolated Python environment and record the versions actually used for reproducing the paper.

From the repository root, the original coupled entry point is:

```sh
python main_LMCMA_HPC_1.py
```

For an LSF cluster, adapt the queue, module, environment, and thread settings in `runjob` to the local system. The current `main()` performs calibration (up to 1,000 objective evaluations), uncertainty analysis, and validation. It is computationally expensive and writes parameter/result files, including large arrays; use a working copy rather than overwriting an archival snapshot.

The independent diagnostic entry point is:

```sh
python main_test_rsas.py
```

It writes `Q_test.txt`. Do not interpret this diagnostic as validating the full LSTM–rSAS workflow. Repository preparation does not establish numerical reproduction of the paper's results.
