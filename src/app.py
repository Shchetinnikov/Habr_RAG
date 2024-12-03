import numpy as np
from ..storage.milvus_store import retriever
from ..models.utils import rerank

query = '...'

chunks = retriever.invoke(query)
texts = [chunk.page_content for chunk in chunks]
reranked_texts = rerank(query, texts)

llm = ...