import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from langchain_milvus import Milvus

from logs.logger import setup_logging
from models.models import embedder, bm25
from config import config

setup_logging(os.path.basename(__file__).split('.')[0])
logger = logging.getLogger(__name__)

# Define encoder's functions
dense_embedding_func = embedder
sparse_embedding_func = bm25

# Define schema of database
dense_field = "dense_vector"
sparse_field = "sparse_vector"

logger.info("Milvus vector store loading...")
URI = config["database_path"]
vector_store = Milvus(
    embedding_function=[dense_embedding_func, bm25],
    collection_name="harb_collection",
    connection_args={"uri": URI},
    vector_field=[dense_field, sparse_field], 
    auto_id=True,
    drop_old=False
)

if __name__ == "__main__":
    print(vector_store.similarity_search_with_score("как создать приложение под ios?", k=5))