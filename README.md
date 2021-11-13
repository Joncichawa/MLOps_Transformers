MLOps_Transformers
==============================

Project for the MLOps course at DTU utilizing Transformers library for state-of-the-art NLP from Hugginface

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
    |
    ├── experiments            <- Experiment yaml files describing model creation and training in ./models
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
   # install libraries
   poetry install
   # run virtual environment
   poetry shell
```
- Training
```bash
   python -m src.models.main train --config experiment-base.yaml
```
- Predict
```bash

```

### Run model with shell (CUDA 11.1)
- Setup (install, virtualenv)
```bash
   # 1. before running install python, virtualenv and cuda 11.1 prerequisites
   # 2. create virtural environment
   mkvirtualenv MLOps
   # 3. activate virtual environment (to find created env run 'workon' without arguments)
   workon MLOps
   # 4. install packages
   pip3 install -r requirements_gpu.txt
```
- Training
```bash
   python -m src.models.main train --config experiment-base.yaml   
```
- Predict
```bash

```

### Run tests
```bash
# unit & integration tests
coverage run -m pytest -vv
# show coverage
coverage report -m
# flake8 PEP8 compliance
flake8 src --max-line-length=120
# black linting
black src --check --diff
# isort linting
isort src --check
```

### Deploy to Azure

(not a final version)

az group create --name MLOps_App  --location northeurope
az storage account create --name MLOps_Storage -l northeurope --sku Standard_LRS -g MLOps_App
az functionapp create --name MLOps_App -g MLOps_App --consumption-plan-location northeurope --storage-account mlopsdeployment --runtime python --runtime-version 3.7 --functions-version 3 --disable-app-insights --os-type Linux

func azure functionapp publish MLOPs_App --no-build

