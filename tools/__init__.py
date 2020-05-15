import logging
import tqdm
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from deepsfp.config import config, update_config
from deepsfp.utils import setup_experiment


class TqdmLoggingHandler(logging.Handler):
    '''Helper to enable logging without interfering with TQDM progress bar. Credit: 
        https://stackoverflow.com/questions/38543506/change-logging-print-function-to-tqdm-write-so-logging-doesnt-interfere-wit/38739634#38739634'''
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)