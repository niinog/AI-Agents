import json
import numpy as np

from tqdm.auto import tqdm
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer


def load_chunks(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


section_chunks = load_chunks("chiphuyen_aie-book_section_chunks.json")


text_index = Index(
    text_fields=["chunk", "section_title", "filename"],
    keyword_fields=[],
)

text_index.fit(section_chunks)


embedding_model = SentenceTransformer("multi-qa-distilbert-cos-v1")

embeddings = []

for d in tqdm(section_chunks):
    text = d["section_title"] + " " + d["chunk"]
    vector = embedding_model.encode(text)
    embeddings.append(vector)

embeddings = np.array(embeddings)


vector_index = VectorSearch()
vector_index.fit(embeddings, section_chunks)


def text_search(query, num_results=5):
    return text_index.search(query, num_results=num_results)


def vector_search(query, num_results=5):
    query_vector = embedding_model.encode(query)
    return vector_index.search(query_vector, num_results=num_results)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "What are best practices for prompt engineering?"

    print(f"\nQuery: {query}")

    print("\nTEXT SEARCH RESULTS")
    text_results = text_search(query)

    for result in text_results:
        print("Score:", result.get("score", result.get("_score", "N/A")))
        print("Section:", result["section_title"])
        print("File:", result["filename"])
        print(result["chunk"][:500])
        print("-" * 80)

    print("\nVECTOR SEARCH RESULTS")
    vector_results = vector_search(query)

    for result in vector_results:
        print("Score:", result.get("score", result.get("_score", "N/A")))
        print("Section:", result["section_title"])
        print("File:", result["filename"])
        print(result["chunk"][:500])
        print("-" * 80)

