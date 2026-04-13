from typing import Any


class SearchTool:
    def __init__(self, index, num_results: int = 8):
        self.index = index
        self.num_results = num_results

    def search(self, query: str) -> list[dict[str, Any]]:
        """
        Search the indexed AI Engineering book chunks.
        """
        results = self.index.search(query, num_results=self.num_results)

        cleaned_results = []

        for result in results:
            cleaned_results.append({
                "section_title": result.get("section_title", ""),
                "filename": result.get("filename", ""),
                "content": result.get("chunk", ""),
                "score": result.get("score", result.get("_score", None)),
            })

        return cleaned_results
