import logging
import pandas as pd
import numpy as np

"""
	This example creates a logging file in a local repo called "logging.log" which has information in format -- Timestamp, levelname & message.
"""

LOGGING_LEVEL = logging.DEBUG
LOGGING_FILE = 'logging.log'

logging.basicConfig(level=LOGGING_LEVEL, filename = LOGGING_FILE, format="%(asctime)s:%(levelname)s:%(message)s")

print('1. test this log') # note not logged

df = pd.DataFrame(np.array([[1, 0], [2, 1], [0, 1]]),columns=['a', 'b'],dtype='float')
logging.debug('\n' + str(df.head()))
logging.info('3. made it to here')

a = 5
b = 0
try:
  c = a / b
except Exception as e:
  logging.exception("Exception occurred")

logging.info('finished the file')