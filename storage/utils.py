import logging
from functools import reduce

from milvus_model import (
    dataset, 
    collection, 
    dense_field, 
    sparse_field, 
    text_field, 
    dense_embedding_func, 
    sparse_embedding_func)


# Document parsing
def doc_to_text(doc):
    text = ""
    if doc['title'] is not None:
        text += f"Заголовок:\n {doc['title']}\n\n"
    if doc['text_markdown'] is not None:
        text += f"Текст:\n {doc['text_markdown']}\n\n"
    if doc['author'] is not None:
        text += f"Автор статьи: {doc['author']}\n"
    if doc['url'] is not None:
        text += f"Ссылка на статью: {doc['url']}\n\n"
    if doc['original_author'] is not None:
        text += f"Автор оригинальной статьи: {doc['original_author']}\n"
    if doc['original_url'] is not None:
        text += f"Ссылка на оригинальную статью: {doc['original_url']}\n\n"
    if len(doc['labels']) != 0:
        text += f"Метки: {reduce(lambda a, b: a + " " + b, doc['labels'])}\n "
    if len(doc['flows']) != 0:
        text += f"Потоки: {reduce(lambda a, b: a + " " + b, doc['flows'])}\n "
    if len(doc['hubs']) != 0:
        text += f"Хабы: {reduce(lambda a, b: a + " " + b, doc['hubs'])}\n"
    if len(doc['tags']) != 0:
        text += f"Теги: {reduce(lambda a, b: a + " " + b, doc['tags'])}\n"
    if doc['format'] is not None:
        text += f"Формат: {doc['format']}"

    return text

# Load data for BM25 corpus
def load_docs():
    num_docs = 10000
    docs = []

    logging.info("Data loading...")
    for index, item in enumerate(dataset):
        docs.append(doc_to_text(item))
        
        if index // 10 == 0 or index == 0:
            logging.info("Iteration: ", index + 1)
        if index > num_docs:
            logging.info("The data download is complete.")
            return docs


# Update retriever collection
def update_collection(docs, log_step=10):
    logging.info("Data loading...")
    entities = []
    for index, doc in enumerate(docs):
        text = doc_to_text(doc)
        entity = {
            dense_field: dense_embedding_func.embed_query(text),
            sparse_field: sparse_embedding_func.embed_query(text),
            text_field: text,
        }
        entities.append(entity)
        if index // log_step == 0:
            logging.info("Iteration: ", index + 1)
            collection.insert(entities)
            del entities
            entities = []

    collection.insert(entities)
    collection.load()
    logging.info("The data download is complete.")

    return collection