import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
# from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_groq import ChatGroq

from server.logs.logger import setup_logging
from storage.utils import load_corpus
from config import GROQ_API_KEY

setup_logging(os.path.basename(__file__).split('.')[0])
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "deepvk/USER-bge-m3"
model_kwargs = {"device": "cuda"}
encode_kwargs = {"normalize_embeddings": True}

logger.info("Embedder loading...")
embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name, 
    model_kwargs=model_kwargs, 
    encode_kwargs=encode_kwargs
)

# logger.info("BM25 loading...")
# bm25 = BM25SparseEmbedding(corpus=load_corpus())

logger.info("Reranker loading...")
reranker = SentenceTransformer('intfloat/multilingual-e5-large',
                               device=device)

logger.info("LLM loading...")
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Вы являетесь помощником, который находит соответствующие URL-адреса для ответа на вопрос пользователя на основе предоставленных документов.
    Вот контекст с метаданными: {context}
    
    Вопрос: {question}
    Пожалуйста, предоставьте список URL-адресов наиболее релевантных документов.
    """
    )
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY)


if __name__ == "__main__":
    print(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))