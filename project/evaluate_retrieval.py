from typing import Any

from agent_building import CHUNKS_FILE, build_index, load_chunks


def precision_at_k(results: list[dict[str, Any]], relevant_filenames: set[str], k: int = 5) -> float:
    top_k = results[:k]

    if not top_k:
        return 0.0

    relevant_retrieved = sum(
        1 for result in top_k
        if result["filename"] in relevant_filenames
    )

    return relevant_retrieved / len(top_k)


def recall_at_k(results: list[dict[str, Any]], relevant_filenames: set[str], k: int = 5) -> float:
    if not relevant_filenames:
        return 0.0

    top_k = results[:k]

    relevant_retrieved = {
        result["filename"]
        for result in top_k
        if result["filename"] in relevant_filenames
    }

    return len(relevant_retrieved) / len(relevant_filenames)


def hit_rate_at_k(results: list[dict[str, Any]], relevant_filenames: set[str], k: int = 5) -> float:
    top_k = results[:k]

    return float(any(
        result["filename"] in relevant_filenames
        for result in top_k
    ))


def reciprocal_rank_at_k(results: list[dict[str, Any]], relevant_filenames: set[str], k: int = 5) -> float:
    top_k = results[:k]

    for rank, result in enumerate(top_k, start=1):
        if result["filename"] in relevant_filenames:
            return 1.0 / rank

    return 0.0


def evaluate_retrieval(index, question: str, relevant_filenames: set[str], k: int = 5) -> dict[str, float]:
    results = index.search(question, num_results=k)

    return {
        "precision_at_k": precision_at_k(results, relevant_filenames, k),
        "recall_at_k": recall_at_k(results, relevant_filenames, k),
        "hit_rate_at_k": hit_rate_at_k(results, relevant_filenames, k),
        "mrr_at_k": reciprocal_rank_at_k(results, relevant_filenames, k),
    }


def main():
    chunks = load_chunks(CHUNKS_FILE)
    index = build_index(chunks)

    eval_dataset = [
        {
            "question": "What are best practices for prompt engineering?",
            "relevant_filenames": {"aie-book-main/prompt-examples.md"},
        },
    ]

    all_metrics = []

    for item in eval_dataset:
        metrics = evaluate_retrieval(
            index=index,
            question=item["question"],
            relevant_filenames=item["relevant_filenames"],
            k=5,
        )
        all_metrics.append(metrics)

        print("Question:", item["question"])
        print(metrics)
        print("-" * 80)

    average_metrics = {
        "precision_at_k": sum(m["precision_at_k"] for m in all_metrics) / len(all_metrics),
        "recall_at_k": sum(m["recall_at_k"] for m in all_metrics) / len(all_metrics),
        "hit_rate_at_k": sum(m["hit_rate_at_k"] for m in all_metrics) / len(all_metrics),
        "mrr_at_k": sum(m["mrr_at_k"] for m in all_metrics) / len(all_metrics),
    }

    print("Average metrics:")
    print(average_metrics)


if __name__ == "__main__":
    main()
