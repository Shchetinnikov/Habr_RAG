from langchain import hub

from storage.milvus_store import retriever
from storage.parsing.create_habr import parse_docs
from storage.utils import update_collection
from models.utils import rerank, step_back
from models.models import llm


# Query and Step-back query
query = '...'
step_back_query = step_back(query, llm)

# Get chunks
chunks = retriever.invoke(query)
step_back_chunks = retriever.invoke(step_back_query)

# Get new docs and chunks from Habr if nessesary
"""
 !Здесь запускать в параллель!
 +
 Решить проблему с импортами
"""
new_docs = parse_docs(query, step_back_query)
update_collection(new_docs)
new_chunks = retriever.invoke(query)
new_step_back_chunks = retriever.invoke(step_back_query)

# Reranking
texts = list(
                set([chunk.page_content for chunk in chunks] + \
                    [chunk.page_content for chunk in step_back_chunks] + \
                    [chunk.page_content for chunk in new_chunks] + \
                    [chunk.page_content for chunk in new_step_back_chunks]
                    )
        )
reranked_texts = rerank(query, texts)[:5]
docs_content = "\n\n".join(doc for doc in reranked_texts)

# LLM-inference
"""
    Степень уверенности ответа
"""
prompt = hub.pull("rlm/rag-prompt")
messages = prompt.invoke({"question": query, "context": docs_content})

response = llm.invoke(messages)
print(response.content)
