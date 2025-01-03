# logger.py
import os
import logging

# base_path = "/mnt/c/Users/kshch/Projects/Habr_RAG"
# log_directory = os.path.join(base_path, "logs")

# def setup_logging(fname: str):
#     logging.basicConfig(
#         level=logging.INFO,
#         format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#         filename=os.path.join(log_directory, f"{fname}.log")  # Используйте os.path.join для создания пути
#     )

def setup_logging(fname: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"../logs/{fname}.log"),
            logging.StreamHandler()
        ]
    )