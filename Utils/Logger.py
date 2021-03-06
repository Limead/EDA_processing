import logging
import datetime

def logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter("[%(asctime)s|%(levelname)s|%(filename)s:%(lineno)s] %(message)s")
    
    logPath = "./EDA_log" + datetime.datetime.today().strftime("%Y%m%d") + ".log"
    fileHandler = logging.FileHandler(logPath)
    fileHandler.setFormatter(formatter)
    
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    
    logger.addHandler(fileHandler)
    logger.addHandler(streamHandler)
    
    return logger