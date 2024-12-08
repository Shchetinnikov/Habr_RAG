import logging
from functools import reduce
from datasets import load_dataset

# Document parsing
def parse_doc(doc):
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
        docs.append(parse_doc(item))
        
        if index // 10 == 0 or index == 0:
            logging.info("Iteration: ", index + 1)
        if index > num_docs:
            logging.info("The data download is complete.")
            return docs



# Logging 
logging.basicConfig(level=logging.INFO, filename="../logs/data_loading.log", filemode="w")

# Load dataset variable
try:
    dataset = load_dataset('IlyaGusev/habr', split="train", streaming=True)
except:
    logging.error("Error occured during huggingface dataset loading on streaming mode")