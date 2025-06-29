import logging
import sys
from logging import Handler, LogRecord

from loguru import logger

from mlflow_3_tutorials.lib.constants import LOG_LEVEL


class InterceptHandler(Handler):
    def emit(self, record: LogRecord) -> None:  # noqa
        try:
            level: str | int = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame = logging.currentframe()
        depth = 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level,
            record.getMessage(),
        )


# Setup Loguru to replace standard logging
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)

# Redirect standard logging to Loguru
logging.basicConfig(handlers=[InterceptHandler()], level=LOG_LEVEL)
logging.getLogger().handlers = [InterceptHandler()]
