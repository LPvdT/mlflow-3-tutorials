import shlex
import subprocess
import sys

from loguru import logger

from mlflow_3_tutorials.lib.constants import LOG_LEVEL

# Configure logger
logger.remove()
logger.bind(name=__file__).add(sys.stderr, level=LOG_LEVEL)


def run_command(
    cmd_str: str,
    description: str | None = None,
    timeout: float | None = 5.0,
    *,
    check: bool = True,
    show_output: bool = False,
) -> None:
    """
    Run a shell command with logging and error handling.

    Args:
        cmd_str (str): The full command string to execute.
        description (str | None): Optional description for logging context.
        timeout (float): Time in seconds before the command times out.
        check (bool): If True, raise an exception for non-zero exit codes.
        show_output (bool): If True, log stdout and stderr regardless of result.
    """

    description = description or cmd_str
    cmd = shlex.split(cmd_str)

    logger.info(f"Running: '{description}' - [{cmd_str}]...")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=False, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{description}' timed out.")
        return
    except KeyboardInterrupt:
        logger.warning(f"'{description}' interrupted by user.")
        return

    _log_process_output(result, always=show_output)

    if check and result.returncode != 0:
        _raise_called_process_error(result, cmd_str)

    if result.returncode == 0:
        logger.success(f"Command '{description}' completed successfully.")


def _log_process_output(
    result: subprocess.CompletedProcess[str], *, always: bool = False
) -> None:
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if always or stdout:
        logger.info(f"stdout:\n{stdout}")
    if always or stderr:
        logger.warning(f"stderr:\n{stderr}")


def _raise_called_process_error(
    result: subprocess.CompletedProcess[str], cmd_str: str
) -> None:
    logger.error(f"Command exited with code {result.returncode}")
    raise subprocess.CalledProcessError(
        result.returncode,
        cmd_str,
        result.stdout,
        result.stderr,
    )
