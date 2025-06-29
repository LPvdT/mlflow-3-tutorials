import shlex
import subprocess

import mlflow
from loguru import logger

from .constants import (
    SERVER_ADDRESS,
    SERVER_PORT,
    TRACKING_URI,
)


def configure_tracking_server() -> None:
    """
    Configure MLflow to use the MLflow tracking server running on
    http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function sets the MLflow tracking URI to
    http://{SERVER_ADDRESS}:{SERVER_PORT} and logs the result.
    """

    logger.info(f"Setting MLflow tracking URI to: {TRACKING_URI}")
    mlflow.set_tracking_uri(TRACKING_URI)


def start_tracking_server() -> None:
    """
    Start an MLflow tracking server on http://{SERVER_ADDRESS}:{SERVER_PORT}.

    This function will block until the MLflow tracking server is stopped.
    """

    cmd_name = f"mlflow server --host {SERVER_ADDRESS} --port {SERVER_PORT}"
    cmd = shlex.split(cmd_name)

    try:
        logger.info(
            f"Starting MLflow tracking server on http://{SERVER_ADDRESS}:{SERVER_PORT}...",
        )
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.error(f"Failed to start MLflow tracking server: {e!s}")
        raise
    except KeyboardInterrupt:
        logger.warning("MLflow tracking server shutdown.")
        return


def uv_sync() -> None:
    """
    Execute the 'uv sync' command to synchronize all groups and compile bytecode.

    This function runs the 'uv sync --managed-python --all-groups --compile-bytecode'
    command to synchronize the environment and compile bytecode for all groups.
    It captures the command's output and logs any exceptions or interruptions
    that occur during its execution.
    """

    cmd_name = "uv sync --managed-python --all-groups --compile-bytecode"
    cmd = shlex.split(cmd_name)

    try:
        logger.info(f"Running: '{cmd_name}'...")
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.error(f"Failed to run 'uv sync': {e!s}")
        raise
    except KeyboardInterrupt:
        logger.warning("'uv sync' interrupted.")
        return


def run_precommit() -> None:
    """
    Run pre-commit commands to update and execute all hooks.

    This function executes two pre-commit commands:

    1. 'pre-commit autoupdate' to update all hooks to the latest versions.
    2. 'pre-commit run -a' to run all hooks against all files.

    It captures the output of each command, logs the progress, and handles
    any exceptions or interruptions that occur during their execution.
    """

    cmd_names = {
        "update": "pre-commit autoupdate",
        "run": "pre-commit run -a",
    }

    cmd_update = shlex.split(cmd_names["update"])
    cmd_run = shlex.split(cmd_names["run"])

    for cmd_name, cmd in zip(
        cmd_names.values(),
        [cmd_update, cmd_run],
        strict=True,
    ):
        try:
            logger.info(f"Running: '{cmd_name}'...")
            _ = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )
        except Exception as e:
            logger.error(f"Failed to run {cmd}: {e!s}")
            raise
        except KeyboardInterrupt:
            logger.warning(f"{cmd} interrupted.")
            return


def run_pyment() -> None:
    """
    Execute the 'pyment' command to run the Python code in the current directory.

    This function runs the 'pyment' command to run the Python code in the
    current directory. It captures the command's output and logs any exceptions
    or interruptions that occur during its execution.
    """

    cmd_name = "pyment -f false -o google ."
    cmd = shlex.split(cmd_name)

    try:
        logger.info(f"Running: '{cmd_name}'...")
        _ = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except Exception as e:
        logger.error(f"Failed to run 'pyment': {e!s}")
        raise
    except KeyboardInterrupt:
        logger.warning("'pyment' interrupted.")
        return
