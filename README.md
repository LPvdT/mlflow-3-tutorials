# MLflow 3 Tutorials

This is a collection of tutorials for [MLflow 3](https://mlflow.org/docs/latest/index.html).

- [MLflow 3 Tutorials](#mlflow-3-tutorials)
  - [Installation (development)](#installation-development)
  - [Running projects](#running-projects)
    - [First MLflow Model](#first-mlflow-model)
    - [Hyperparameter Tuning \& Deployment](#hyperparameter-tuning--deployment)
    - [Deep Learning Quickstart](#deep-learning-quickstart)
    - [Model Registry Quickstart](#model-registry-quickstart)

## Installation (development)

You can install the project with `uv`, using the following command:

```sh
# Sync all dependency groups and download `uv`-managed Python
uv sync --managed-python --all-groups
```

> [!NOTE]
> **Running scripts**
>
> - Check `pyproject.toml` for available scripts
> - Run them using `uv run <script_name>`

## Running projects

> [!TIP]
> You can run `rm -rf .venv/ uv.lock` after each project, and before running the next required `uv sync` command to cut down on the size of your `.venv` folder.

### First MLflow Model

> [!IMPORTANT]
> Ensure you have run [the initial `uv sync`](#installation-development) command.

- Code: `first_model.py`
  - Run: `uv run first_model`
    - If you encounter an error that the experiment already exists:
      - Run: `uv run remove_experiments`

### Hyperparameter Tuning & Deployment

> [!IMPORTANT]
> Run:
>
> - `uv sync --managed-python --all-groups --extra tensorflow --extra mlflow_extras`

- Code: `tuning_deployment.py`
  - Run: `uv run tuning_deployment`
  - Serve: `uv run serve_wine_model`

### Deep Learning Quickstart

> [!IMPORTANT]
> Run:
>
> - `uv sync --managed-python --all-groups --extra pytorch`

- Code: `deep_learning_quickstart.py`
  - Run: `uv run deep_learning`

### Model Registry Quickstart

- > Work in progress
