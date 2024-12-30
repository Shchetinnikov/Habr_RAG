# logger.py
import logging

def setup_logging(fname: str):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f"../logs/{fname}.log"),
            logging.StreamHandler()
        ]
    )