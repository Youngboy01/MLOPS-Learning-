import logging
import os

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)#ensuring log_directory exists
logger = logging.getLogger("DataIngestion")#Made an object of name logger
logger.setLevel("DEBUG")#set level to debug , this will show the other levels too 
console_handler = logging.StreamHandler()#made an object named console_handler using StreamHandler function, this helps printing our logs directly in the terminal. Logger->handler->console_handler
console_handler.setLevel("DEBUG")
filepath = os.path.join(log_dir, "Data_ingestion.log")#Creating a file path for printing our logs in a file
file_handler = logging.FileHandler(filepath)#file_handler will print our logs directly in the file
file_handler.setLevel("DEBUG")

#handler done  now formatter
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)#its just the format in which my logs will be printed.
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

#LOGGING PART DONE
