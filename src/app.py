from langchain import hub

import numpy as np
from storage.milvus_store import retriever
from models.utils import rerank, step_back
from models import llm


prompt = hub.pull("rlm/rag-prompt")

query = '...'
step_back_query = step_back(query, llm)

chunks = retriever.invoke(query)
step_back_chunks = retriever.invoke(step_back_query)

texts = [chunk.page_content for chunk in chunks] + \
        [chunk.page_content for chunk in step_back_chunks]
reranked_texts = rerank(query, texts)[:5]

docs_content = "\n\n".join(doc for doc in reranked_texts)
messages = prompt.invoke({"question": query, "context": docs_content})

response = llm.invoke(messages)
print(response.content)
