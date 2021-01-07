from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
from datetime import datetime
    
def init_logger():    
    now = datetime.now()
    log_file = './logs/' + now.strftime("%d-%m-%Y_%H:%M:%S.log")
    
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    
    # StreamHandler: sends logging output to a stream, in this case the log_file
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    
    # FileHandler: opens, closes file, sends logging output to the file
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger