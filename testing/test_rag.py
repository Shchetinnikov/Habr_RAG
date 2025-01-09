import requests
from urllib.parse import urljoin
from langchain import hub

from testing.test_utils import server_url, llm_judge, validation_inputs

from tqdm import tqdm


# Функция для отправки запроса к RAG API
def query_rag_system(query):
    response = requests.post(urljoin(server_url, "/api/user_message"), json={"role": "user", "content": query})
    if response.status_code == 200:
        return {"content": response.json()["data"]["content"], "context": response.json()["data"]["context"]}
    else:
        print(f"Ошибка при обращении к RAG API: {response.status_code}")
        return None

# Функция для оценки ответа с помощью LangChain LLM
def evaluate_response_helpfulness(validation_inputs):
    prompt = hub.pull("langchain-ai/rag-answer-helpfulness")
    evaluation_chain = prompt | llm_judge
    scores = []
    for query in tqdm(validation_inputs):
        generated_response = query_rag_system(query)
        if generated_response:
            score = evaluation_chain.invoke({"question": query,
                                            "student_answer": generated_response})
            scores.append(score["Score"])
    
    return sum(scores) / len(scores)

def evaluate_response_helpfulness(validation_inputs):
    prompt = hub.pull("langchain-ai/rag-answer-helpfulness")
    evaluation_chain = prompt | llm_judge
    scores = []
    for query in tqdm(validation_inputs):
        generated_response = query_rag_system(query)
        if generated_response:
            score = evaluation_chain.invoke({"question": query,
                                            "student_answer": generated_response["content"]})
            scores.append(score["Score"])
    
    return sum(scores) / len(scores)

def evaluate_response_hallucination(validation_inputs):
    prompt = hub.pull("langchain-ai/rag-answer-hallucination")
    evaluation_chain = prompt | llm_judge
    scores = []
    for query in tqdm(validation_inputs):
        generated_response = query_rag_system(query)
        if generated_response:
            score = evaluation_chain.invoke({"documents": generated_response["context"],
                                            "student_answer": generated_response["content"]})
            scores.append(score["Score"])
    
    return sum(scores) / len(scores)

def evaluate_response_relevance(validation_inputs):
    prompt = hub.pull("langchain-ai/rag-document-relevance")
    evaluation_chain = prompt | llm_judge
    scores = []
    for query in tqdm(validation_inputs):
        generated_response = query_rag_system(query)
        if generated_response:
            score = evaluation_chain.invoke({"question": query,
                                            "documents": generated_response["context"]})
            scores.append(score["Score"])
    
    return sum(scores) / len(scores)


if __name__ == "__main__":
    helpfulness_score = evaluate_response_helpfulness(validation_inputs)
    print("Helpfulness score", helpfulness_score)

    hallucination_score = evaluate_response_hallucination(validation_inputs)
    print("Hallucination score", hallucination_score)

    relevance_score = evaluate_response_relevance(validation_inputs)
    print("Relevance score", relevance_score)
