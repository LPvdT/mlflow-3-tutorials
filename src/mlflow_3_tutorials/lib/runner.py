import shlex
import subprocess

from loguru import logger


def run_command(
    cmd_str: str,
    description: str | None = None,
    *,
    check: bool = True,
    show_output: bool = False,
) -> None:
    """
    Run a shell command with logging and error handling.

    Args:
        cmd_str (str): The full command string to execute.
        description (str | None): Optional description for logging context.
        check (bool): If True, raise an exception for non-zero exit codes.
    """

    description = description or cmd_str
    cmd = shlex.split(cmd_str)

    try:
        logger.info(f"Running: '{description}' - [{cmd_str}]...")
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=check, timeout=5
        )

        if show_output:
            logger.info(f"stdout:\n{result.stdout.strip()}")
            if err := result.stderr:
                logger.warning(f"stderr:\n{err.strip()}")

        if check:
            _raise_called_process_error(result, cmd_str)
    except subprocess.TimeoutExpired:
        logger.error(f"Command '{description}' timed out.")
    except KeyboardInterrupt:
        logger.warning(f"'{description}' interrupted by user.")
    else:
        logger.success(f"Command '{description}' completed successfully.")


def _raise_called_process_error(
    result: subprocess.CompletedProcess[str],
    cmd_str: str,
) -> None:
    if (code := result.returncode) != 0:
        logger.error(f"Command exited with code {code}")
        logger.warning(f"stdout:\n{result.stdout.strip()}")
        logger.warning(f"stderr:\n{result.stderr.strip()}")

        raise subprocess.CalledProcessError(
            code,
            cmd_str,
            result.stdout,
            result.stderr,
        )
