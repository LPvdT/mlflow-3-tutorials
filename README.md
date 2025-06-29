# MLflow 3 Tutorials

This is a collection of tutorials for [MLflow 3](https://mlflow.org/docs/latest/index.html).

- [MLflow 3 Tutorials](#mlflow-3-tutorials)
  - [Installation](#installation)
  - [Running scripts](#running-scripts)
  - [Running code](#running-code)
    - [First MLflow Model](#first-mlflow-model)
    - [Hyperparameter Tuning \& Deployment](#hyperparameter-tuning--deployment)

## Installation

You can install the project with `uv`, using the following command:

```sh
# Sync all dependency groups, download Python using uv and compile bytecode to improve runtime
uv sync --managed-python --all-groups --compile-bytecode
```

## Running scripts

- Check the `pyproject.toml` for available scripts
- Run them using `uv run <script_name>`

## Running code

### First MLflow Model

- `first_model.py`:
  - Run: `uv run first_model`
    - If you encounter an error that the experiment already exists, then delete it manually for now:
      - `mlruns/142873212763357834` and/or `mlruns/665733959038238543`
      > - Actual IDs are different for you

### Hyperparameter Tuning & Deployment

- _Work in progress_
