import logging
import os

logger = logging.getLogger("vulcan")
logger.setLevel(level=os.getenv("LOGLEVEL", logging.INFO))
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)
