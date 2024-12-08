from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_milvus.utils.sparse import BM25SparseEmbedding
from langchain_groq import ChatGroq

from storage.load_data import load_docs

load_dotenv(".env")

model_name = "deepvk/USER-bge-m3"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}

embedder = HuggingFaceBgeEmbeddings(
    model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
)
bm25 = BM25SparseEmbedding(corpus=load_docs())
reranker = SentenceTransformer('intfloat/multilingual-e5-large')
llm = ChatGroq(model="llama-3.3-70b-versatile")
