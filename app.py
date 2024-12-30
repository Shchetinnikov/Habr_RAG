import os
import sys
import logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logs.logger import setup_logging
from models.models import llm, prompt
from models.utils import rerank, step_back
from storage.milvus_store import vector_store
from storage.utils import format_docs


if __name__ == "__main__":
    setup_logging(os.path.basename(__file__).split('.')[0])
    logger = logging.getLogger(__name__)

    # Query and Step-back query
    logger.info("Query reading and stepback prompting...")
    query = '...'
    step_back_query = step_back(query)

    # Get chunks
    logger.info("Query to vector store...")
    chunks = vector_store.similarity_search_with_score(query, k=5)
    step_back_chunks = vector_store.similarity_search_with_score(step_back_query, k=5)


    # Get new docs and chunks from Habr if nessesary
    # from storage.parsing.create_habr import parse_docs
    # from storage.utils import update_collection

    # new_docs = parse_docs(query, step_back_query)
    # update_collection(new_docs)
    # new_chunks = retriever.invoke(query)
    # new_step_back_chunks = retriever.invoke(step_back_query)



    # Reranking
    logger.info("Chunks reranking...")
    reranked_chunks = rerank(query, [chunks + step_back_chunks])[:5]
    context = format_docs(reranked_chunks)   

    # LLM-inference
    logger.info("LLM inference...")
    messages = prompt.invoke({"question": query, 
                              "context": context})
    response = llm.invoke(messages)


    # from langchain_core.runnables import RunnablePassthrough
    # from langchain_core.output_parsers import StrOutputParser

    # qa_chain = (
    #     {
    #         "context": vector_store.as_retriever() | format_docs,
    #         "question": RunnablePassthrough(),
    #     }
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )

    # print(qa_chain.invoke(query))
