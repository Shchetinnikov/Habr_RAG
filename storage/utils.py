import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
from langchain_core.documents import Document

from logs.logger import setup_logging
from config import config

setup_logging(os.path.basename(__file__).split('.')[0])
logger = logging.getLogger(__name__)

# Load data for BM25 corpus
def load_corpus():
    logger.info("Data loading...")
    corpus = pd.read_csv(config["corpus_path"])["texts"].tolist()
    logger.info("The data downloading is completed.")
    return corpus

# Format documents to context
def format_docs(docs: Document) -> str:
    logger.info("Formating documents to context...")
    context = []
    for doc in docs:
         context.append(f"""
                        URL: {doc.metadata["url"]}
                        Title: {doc.metadata["title"]}
                        Content: {doc.page_content}
                        Author: {doc.metadata["author"]}
                        Original author: {doc.metadata["original_author"]}
                        Original url: {doc.metadata["original_url"]}
                        """) 
    logger.info("Formating is completed.")
    return "\n\n".join(context)


# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# 
# # Parse content to document
# def parse_doc(doc):
#     return Document(
#         page_content=doc["text_markdown"],
#         metadata={"title": doc["title"] if doc["title"] is not None else "",
#                   "author": doc["author"] if doc["author"] is not None else "",
#                   "url": doc["url"] if doc["url"] is not None else "",
#                   "original_author": doc["original_author"] if doc["original_author"] is not None else "",
#                   "original_url": doc["original_url"] if doc["original_url"] is not None else "",
#                   "labels": ", ".join(doc["labels"]),
#                   "flows": ", ".join(doc["flows"]),
#                   "hubs": ", ".join(doc["hubs"]),
#                   "tags": ", ".join(doc["tags"]),
#                   "format": doc['format'] if doc["format"] is not None else ""}
#     )

# # Update retriever collection
# def update_collection(vector_store, docs, log_step=10):
#     logging.info("Data loading...")

#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=100)
#     docs_splitted = text_splitter.split_documents(docs)
#     vector_store.add_documents(docs_splitted)
    
#     logging.info("The data downloading is completed.")

#     return vector_store