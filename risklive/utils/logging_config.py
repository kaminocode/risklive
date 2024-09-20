import logging
import os
from logging.handlers import RotatingFileHandler

def setup_logging(name):
    log_dir = 'logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, 'app.log')
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        c_handler = logging.StreamHandler()  
        f_handler = RotatingFileHandler(log_file, maxBytes=10*1024*1024, backupCount=5)  
        c_handler.setLevel(logging.INFO)
        f_handler.setLevel(logging.DEBUG)

        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
    
    return logger