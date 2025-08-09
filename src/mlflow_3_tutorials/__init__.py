import logging

import seaborn as sns
from loguru import logger

from mlflow_3_tutorials.lib.constants import LOG_LEVEL
from mlflow_3_tutorials.lib.utils import InterceptHandler

# Setup Loguru to replace standard logging
logger.remove()

# Redirect standard logging to Loguru
logging.basicConfig(handlers=[InterceptHandler()], level=LOG_LEVEL)
logging.getLogger().handlers = [InterceptHandler()]

# Set the default style for seaborn plots
sns.set_style("whitegrid")
