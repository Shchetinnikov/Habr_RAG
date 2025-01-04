import os
import json
from dotenv import load_dotenv

load_dotenv()

path = "./config.json"

with open(path, 'r') as file:
    config = json.load(file)

DATABASE_PATH = config["database_path"]
CORPUS_PATH = config["corpus_path"]
SERVER_HOST = os.environ.get("SERVER_HOST")
SERVER_PORT = os.environ.get("SERVER_PORT")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")


if __name__ == "__main__":
    print(config)