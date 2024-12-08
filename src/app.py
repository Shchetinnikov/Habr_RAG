from langchain import hub

import numpy as np
from storage.milvus_store import retriever
from models.utils import rerank
from models import llm


prompt = hub.pull("rlm/rag-prompt")

query = '...'

chunks = retriever.invoke(query)
texts = [chunk.page_content for chunk in chunks]
reranked_texts = rerank(query, texts)

docs_content = "\n\n".join(doc for doc in reranked_texts)
messages = prompt.invoke({"question": query, "context": docs_content})

response = llm.invoke(messages)
print(response.content)
