import numpy as np
from models.models import reranker

def rerank(query, texts):
    input_texts = [f'query: {query}'] + \
                  [f"passage: {text}" for text in texts]
    embeddings = reranker.encode(input_texts, normalize_embeddings=True)

    query_emb = embeddings[0]
    passage_embs = embeddings[1:]

    ranks = np.array(passage_embs.shape[0])
    for index, passage_emb in enumerate(passage_embs):
        ranks[index] = reranker.similarity(query_emb, passage_emb).item()
    indexes = ranks.argsort()[::-1]

    return indexes
