from typing import Any
import tiktoken
from elasticsearch import Elasticsearch
import httpx
from tqdm import tqdm

INDEX_NAME = "course-questions"


def get_documents():
    docs_url = "https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1"
    docs_response = httpx.get(docs_url, follow_redirects=True)
    documents_raw = docs_response.json()

    documents = []

    for course in documents_raw:
        course_name = course["course"]

        for doc in course["documents"]:
            doc["course"] = course_name
            documents.append(doc)
    return documents


def index_documents(documents, es_client):
    index_settings = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
            }
        },
    }

    es_client.indices.create(index=INDEX_NAME, body=index_settings)

    for doc in tqdm(documents):
        es_client.index(index=INDEX_NAME, document=doc)


def elastic_search(query: str, es_client: Elasticsearch, course_filter: str = ""):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields",
                    }
                },
            }
        },
    }

    if course_filter:
        search_query["query"]["bool"]["filter"] = {"term": {"course": course_filter}}

    response = es_client.search(index=INDEX_NAME, body=search_query)

    return response["hits"]["hits"]


def build_prompt(question: str, context: list[dict[str, Any]]) -> str:
    context_template = """Q: {question}\nA: {text}""".strip()

    llm_context = "\n\n".join(
        context_template.format(
            question=doc["_source"]["question"], text=doc["_source"]["text"]
        )
        for doc in context
    )

    prompt_template = """You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
    """.strip()

    result = prompt_template.format(question=question, context=llm_context)
    return result


def main():
    # documents = get_documents()
    es_client = Elasticsearch("http://127.0.0.1:9200")
    # index_documents(documents, es_client)

    query = "How do execute a command on a Kubernetes pod?"
    result = elastic_search(query, es_client)
    print(result)
    query = "How do copy a file to a Docker container?"
    result = elastic_search(
        query, es_client, course_filter="machine-learning-zoomcamp"
    )[:3]
    print("=" * 80)
    print(result[2])

    prompt_result = build_prompt("How do copy a file to a Docker container?", result)
    print("=" * 80)
    print(prompt_result)
    print(len(prompt_result))

    print("=" * 80)
    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt_result)
    print(len(tokens))


if __name__ == "__main__":
    main()
