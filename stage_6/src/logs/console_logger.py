"""Console logger"""

import logging

from src.logs.handlers.console_handler import ConsoleLogHandler

LOGGER = logging.getLogger("logger")
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(ConsoleLogHandler())
