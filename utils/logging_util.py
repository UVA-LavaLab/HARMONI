import logging
import os

path = os.environ['HARMONI_HOME']

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Create handlers (output destinations)
ch = logging.StreamHandler()  # Console handler
fh = logging.FileHandler(f'{path}/sim.log')  # File handler

# Set logging level for handlers
ch.setLevel(logging.INFO)
fh.setLevel(logging.INFO)
#fh.setLevel(logging.CRITICAL)

# Create formatters (define log message format)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Add formatter to handlers
ch.setFormatter(formatter)
fh.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(ch)
logger.addHandler(fh)

# Example import in external file
# from utils/logging_util import logger

# Example Log messages
# logger.debug('This is a debug message')
# logger.info('This is an info message')
# logger.warning('This is a warning message')
# logger.error('This is an error message')
# logger.critical('This is a critical message')
