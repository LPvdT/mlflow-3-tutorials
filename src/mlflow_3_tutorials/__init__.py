import sys

from loguru import logger

from .constants import LOG_LEVEL

logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL)
