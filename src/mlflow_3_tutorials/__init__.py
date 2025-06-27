import sys

from loguru import logger

from .constants import LOG_LEVEL

# Remove the default logger and add a new one with the specified log level
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)
