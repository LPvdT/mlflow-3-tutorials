import shlex
import subprocess
from typing import Never

from loguru import logger


def run_command(
    cmd_str: str,
    description: str | None = None,
    *,
    check: bool = False,
) -> None:
    """
    Run a shell command with logging and error handling.

    Args:
        cmd_str (str): The full command string to execute.
        description (Optional[str]): Optional description for logging context.
        check (bool): If True, raise an exception for non-zero exit codes.
    """

    description = description or cmd_str
    cmd = shlex.split(cmd_str)

    try:
        logger.info(f"Running: '{cmd_str}' ({description})...")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )

        if result.returncode != 0:
            logger.warning(f"Command exited with code {result.returncode}")
            logger.warning(f"stdout:\n{result.stdout.strip()}")
            logger.warning(f"stderr:\n{result.stderr.strip()}")

        if check:
            _raise_called_process_error(result, cmd_str)
        else:
            logger.info(f"Command '{description}' completed successfully.")

    except KeyboardInterrupt:
        logger.warning(f"'{description}' interrupted by user.")
    except Exception as e:
        logger.error(f"Failed to run '{description}': {e!s}")
        raise


def _raise_called_process_error(
    result: subprocess.CompletedProcess[str],
    cmd_str: str,
) -> Never:
    raise subprocess.CalledProcessError(
        result.returncode,
        cmd_str,
        output=result.stdout,
        stderr=result.stderr,
    )
