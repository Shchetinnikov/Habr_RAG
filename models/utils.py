import numpy as np
from models.models import reranker
from langchain_core.prompts import ChatPromptTemplate

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

    return [texts[idx] for idx in indexes]


def step_back(query, llm):
    system = """Вы являетесь экспертом в области мировых знаний. 
                Ваша задача - выполнить step-back, перефразировать вопрос в более общий, на который проще ответить. 

                Вот несколько примеров:
                Оригинальный вопрос: Какую должность занимал Нокс Каннингем с мая 1955 по апрель 1956 года?
                Step-back вопрос: Какие должности занимал Нок Каннингем в своей карьере?

                Оригинальный вопрос: Кто был супругом Анны Карины с 1968 по 1974 год?
                Step-back вопрос: Кем были супруги Анны Карины?

                Оригинальный вопрос:: За какую команду Тьерри Одель играл с 2007 по 2008 год?
                Step-back вопрос: За какие команды Тьерри Одель играл в своей карьере?

                Напишите только вопрос без каких-либо пояснений.
                """

    prompt = (
        """
        Оригинальный вопрос: {question}
        Step-back вопрос: <ваш ответ>
        """
    )


    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            ("human", prompt)
        ]
    )

    messages = chat_template.format_messages(question=query)
    step_back_query = llm.invoke(messages).content

    return step_back_query