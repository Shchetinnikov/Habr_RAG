import os
import logging
from dotenv import load_dotenv
from datasets import load_dataset
from langchain_milvus.retrievers import MilvusCollectionHybridSearchRetriever
from pymilvus import (
    MilvusClient,
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    WeightedRanker,
    connections,
)

from models import embedder, bm25
from .utils import update_collection


load_dotenv()
logging.basicConfig(level=logging.INFO, filename="../logs/milvus.log", filemode="w")

# Load dataset variable
try:
    dataset = load_dataset('IlyaGusev/habr', split="train", streaming=True)
except:
    logging.error("Error occured during huggingface dataset loading on streaming mode")

# Connect to Milvus
logging.info("Milvus connection...")
CONNECTION_URI = os.getenv("MILVUS_CONNECTION_URI")
connections.connect(uri=CONNECTION_URI)

# Define encoder's functions
logging.info("Dense & sparse embedders...")
dense_embedding_func = embedder
sparse_embedding_func = bm25

# Define schema of database
logging.info("Scheme definition...")
pk_field = "doc_id"
dense_field = "dense_vector"
sparse_field = "sparse_vector"
text_field = "text"

fields = [
    FieldSchema(
        name=pk_field,
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=True,
        max_length=100,
    ),
    FieldSchema(name=dense_field, dtype=DataType.FLOAT_VECTOR, dim=1024),
    FieldSchema(name=sparse_field, dtype=DataType.SPARSE_FLOAT_VECTOR),
    FieldSchema(name=text_field, dtype=DataType.VARCHAR, max_length=65_535),
]

# Create database
logging.info("Database creation...")
collaction_name = "habr_collection"
schema = CollectionSchema(fields=fields, enable_dynamic_field=False)
collection = Collection(
    name=collaction_name, schema=schema, consistency_level="Strong"
)

# Create index
logging.info("Index creation...")
dense_index = {"index_type": "DISKANN", "metric_type": "IP"}
sparse_index = {"index_type": "SPARSE_INVERTED_INDEX", "metric_type": "IP"}

collection.create_index("dense_vector", dense_index)
collection.create_index("sparse_vector", sparse_index)

collection.flush()

# Add data
collection = update_collection(dataset, log_step=10000)

# Create retriever
logging.info("Retriever creation...")
sparse_search_params = {"metric_type": "IP"}
dense_search_params = {"metric_type": "IP", "params": {}}

retriever = MilvusCollectionHybridSearchRetriever(
    collection=collection,
    rerank=WeightedRanker(0.8, 0.2),
    anns_fields=[dense_field, sparse_field],
    field_embeddings=[dense_embedding_func, sparse_embedding_func],
    field_search_params=[dense_search_params, sparse_search_params],
    top_k=5,
    text_field=text_field,
)
