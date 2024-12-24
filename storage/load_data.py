from functools import reduce
from datasets import load_dataset
import os
import logging

logs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../logs'))
os.makedirs(logs_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO, filename=os.path.join(logs_dir, 'data_loading.log'), filemode="w")

# Load dataset variable
try:
    dataset = dataset = load_dataset('IlyaGusev/habr', split="train", streaming=True, trust_remote_code=True)
    logging.info("Dataset successfully loaded.")
except Exception as e:
    logging.error(f"Error occurred during dataset loading: {e}")
    dataset = None


# Document parsing
def parse_doc(doc):
    # Проверяем наличие тегов
    tags = " ".join(doc.get('tags', [])) if doc.get('tags') else "Нет тегов"

    if doc.get('original_author') and doc.get('original_url'):
        return f"""
                {doc.get('title', 'Без названия')}\n
                {doc.get('text_markdown', '')}\n
                \n
                Автор статьи: {doc.get('author', 'Неизвестен')}\n
                Ссылка на статью: {doc.get('url', 'Нет ссылки')}\n
                \n
                Автор оригинальной статьи: {doc.get('original_author', 'Неизвестен')}\n
                Ссылка на оригинальную статью: {doc.get('original_url', 'Нет ссылки')}\n
                \n
                Теги: {tags}
            """
    else:
        return f"""
                {doc.get('title', 'Без названия')}\n
                {doc.get('text_markdown', '')}\n
                \n
                Автор статьи: {doc.get('author', 'Неизвестен')}\n
                Ссылка на статью: {doc.get('url', 'Нет ссылки')}\n
                \n
                Теги: {tags}
            """

# Load data for BM25 corpus
def load_docs():

    if dataset is None:
        logging.error("Dataset is not available. Exiting load_docs.")
        return []

    num_docs = 10000
    docs = []

    logging.info("Data loading...")
    for index, item in enumerate(dataset):
        docs.append(parse_doc(item))
        
        if index % 10 == 0 or index == 0:
            logging.info(f"Iteration: {index + 1}")
        if index > num_docs:
            logging.info("The data download is complete.")
            return docs