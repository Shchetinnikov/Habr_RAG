from langchain_milvus import Milvus
from models.models import embedder, llm
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""
Вы являетесь помощником, который находит соответствующие URL-адреса для ответа на вопрос пользователя на основе предоставленных документов.
Вот контекст с метаданными:
{context}

Вопрос: {question}
Пожалуйста, предоставьте список URL-адресов наиболее релевантных документов.
"""
)

URI = "./milvus_example_1.db"

vector_store = Milvus(
    embedding_function=embedder,
    connection_args={"uri": URI},
)

def format_docs(documents):
    context = []
    for doc in documents:
        url = doc.metadata.get("url", "No URL available")
        title = doc.metadata.get("title", "No Title")
        content = doc.page_content
        context.append(f"Title: {title}\nURL: {url}\nContent: {content}") 
    return "\n\n".join(context)

qa_chain = (
    {
        "context": vector_store.as_retriever() | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print(qa_chain.invoke("что такое python?"))
