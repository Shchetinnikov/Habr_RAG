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
