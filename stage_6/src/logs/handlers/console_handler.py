"""Handler for logger"""

import logging
import sys


class ConsoleLogHandler(logging.StreamHandler):
    """General-purpose console logger"""

    def __init__(self, stream=sys.stdout):
        super().__init__(stream)
        self.setLevel(logging.INFO)
        self.setFormatter(
            logging.Formatter(fmt="[%(asctime)s: %(levelname)s] %(message)s")
        )
