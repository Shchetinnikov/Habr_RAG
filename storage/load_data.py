import logging
from functools import reduce
from datasets import load_dataset

# Document parsing
def parse_doc(doc):
    if doc['original_author'] and doc['original_url']:
        return f"""
                {doc['title']}\n
                {doc['text_markdown']}\n
                \n
                Автор статьи: {doc['author']}\n
                Ссылка на статью: {doc['url']}\n
                \n
                Автор оригинальной статьи: {doc['original_author']}\n
                Ссылка на оригинальную статью: {doc['original_url']}\n
                \n
                Теги: {reduce(lambda a, b: a + " " + b, doc['tags'])}
            """
    else:
        return f"""
                {doc['title']}\n
                {doc['text_markdown']}\n
                \n
                Автор статьи: {doc['author']}\n
                Ссылка на статью: {doc['url']}\n
                \n
                Теги: {reduce(lambda a, b: a + " " + b, doc['tags'])}
            """

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