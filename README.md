MLOps_Transformers
==============================

Project for the MLOps course at DTU utilizing Transformers library for state-of-the-art NLP from Hugginface.

## 1. Project Organization
------------

    ├── .github                <- Github CI Actions definitions for unit, integration tests and pep8 checks
    ├── LICENSE
    ├── README.md              <- The top-level README for developers using this project.
    ├── data
    │   ├── processed          <- The final, canonical data sets for modeling.
    │   └── raw                <- The original, immutable data dump.
    │
    ├── models                 <- Trained and serialized models, model predictions, or model summaries
    │
    ├── reports                <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures            <- Generated graphics and figures to be used in reporting
    │
    ├── poetry.lock            <- TODO
    ├── pyproject.toml         <- TODO
    ├── requirements_gpu.txt   <- The requirements file for reproducing the analysis environment on GPU
    │
    └── src                    <- Source code for use in this project.
        │
        ├── data               <- Scripts to download or generate data
        │   └── make_dataset.py
        │
        ├── models             <- Scripts to train models and then use trained models to make
        │                         predictions
        │
        ├── tests              <- Unit and Integration tests
        │
        └── visualization      <- Scripts to create exploratory and results oriented visualizations
            └── visualize.py



## 2. How to run
------------
### Run model with Poetry (CPU only)
- Setup (install, virtualenv)
```bash

```
- Training
```bash
    
```
- Predict
```bash

```

### Run model with shell (CUDA 11.1)
- Setup (install, virtualenv)
```bash

```
- Training
```bash
    
```
- Predict
```bash

```

### Run tests
```bash
# unit & integration tests

# test coverage

# isort & flake8 compliance
```