import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from fastapi import FastAPI

from server.logs.logger import setup_logging
from server.models.models import llm, prompt
from server.models.utils import rerank, step_back
from storage.milvus_store import vector_store
from storage.utils import format_docs
from server.schemas import Message

app = FastAPI(
    title="Server"
)

setup_logging(os.path.basename(__file__).split('.')[0])
logger = logging.getLogger(__name__)

@app.post("/api/user_message")
def process_user_message(data: Message):
    # Query and Step-back query
    logger.info("Query reading and stepback prompting...")
    query = str(data.content).encode('utf-8', 'ignore').decode('utf-8')
    step_back_query = step_back(query)

    # Get chunks
    logger.info("Query to vector store...")
    chunks = vector_store.similarity_search_with_score(query, k=5)
    step_back_chunks = vector_store.similarity_search_with_score(step_back_query, k=5)

    # Reranking
    logger.info("Chunks reranking...")
    reranked_chunks = rerank(query, chunks + step_back_chunks)[:3]

    # LLM-inference
    logger.info("LLM inference...")

    try:
        context = format_docs(reranked_chunks)   
        messages = prompt.invoke({"question": query, 
                                "context": context})
        response = llm.invoke(messages)
    except:
        context = format_docs([reranked_chunks[0]])   
        messages = prompt.invoke({"question": query, 
                                "context": context})
        response = llm.invoke(messages)

    return {"status": 200,
            "data": {
                "role": "assistant",
                "content": response.content
            }}


if __name__ == "__main__":
    pass