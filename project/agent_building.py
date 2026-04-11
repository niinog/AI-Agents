import json
import os
from pathlib import Path
from typing import Any

from minsearch import Index
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider


CHUNKS_FILE = Path(__file__).parent / "chiphuyen_aie-book_section_chunks.json"



def load_chunks(filename: str | Path) -> list[dict[str, Any]]:
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(chunks: list[dict[str, Any]]) -> Index:
    index = Index(
        text_fields=["chunk", "section_title", "filename"],
        keyword_fields=[],
    )
    index.fit(chunks)
    return index


def build_agent() -> Agent:
    section_chunks = load_chunks(CHUNKS_FILE)
    index = build_index(section_chunks)

    def text_search(query: str) -> list[dict[str, Any]]:
        return index.search(query, num_results=5)

    system_prompt = """
You are a helpful assistant for answering questions about Chip Huyen's AI Engineering book and related materials.

Use the text_search tool to find relevant information from the section chunks before answering.

If search returns relevant results, answer using those results.
Mention the relevant section title and filename when possible.

If search does not return relevant information, say that you could not find the answer in the provided materials and then give general guidance.
"""

    provider = GoogleProvider(api_key="AIzaSyDaIzpzJsH2lD9sIio--Gtr2ALmWTiP7Tk")

    model = GoogleModel(
        "gemini-3-flash-preview",
        provider=provider,
    )

    return Agent(
        model=model,
        name="aie_book_agent",
        instructions=system_prompt,
        tools=[text_search],
    )
