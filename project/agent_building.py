import os
from functools import lru_cache

from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from ingest import index_data
from search_tools import SearchTool


REPO_OWNER = "chiphuyen"
REPO_NAME = "aie-book"


SYSTEM_PROMPT_TEMPLATE = """
You are a practical AI Engineering assistant answering questions using Chip Huyen's AI Engineering materials.

Always use the search tool before answering.

When search results are relevant:
- Give a direct answer first.
- Extract concrete examples, rules, or patterns from the retrieved materials.
- Avoid saying generic phrases like "the search results do not explicitly say..." unless truly necessary.
- If the user asks for examples, provide examples.
- If the user asks for rules or best practices, provide a clear checklist.
- Cite the source filename for every major point.
- Format source references as GitHub links.

If search results are weak or unrelated:
- Say you could not find enough direct evidence in the materials.
- Then provide general guidance, clearly labeled as general guidance.

Preferred answer format:
1. Short direct answer
2. Practical rules/checklist
3. Concrete examples from the sources
4. References
"""



@lru_cache(maxsize=1)
def build_index_once():
    """
    Build the search index once and reuse it.

    This prevents downloading the GitHub repo and rebuilding the index
    every time build_agent() is called during the same app process.
    """
    return index_data(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
        chunk=True,
        chunking_strategy="section",
        chunking_params={"level": 2},
    )


@lru_cache(maxsize=1)
def build_agent() -> Agent:
    """
    Build the PydanticAI agent once and reuse it.

    In deployment, call build_agent() at app startup or reuse the cached result
    instead of rebuilding for every user request.
    """
    index = build_index_once()
    search_tool = SearchTool(index, num_results=8)

    provider = GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"])

    model = GoogleModel(
        "gemini-2.5-flash-lite",
        provider=provider,
    )

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        repo_owner=REPO_OWNER,
        repo_name=REPO_NAME,
    )

    return Agent(
        model=model,
        name="aie_book_agent",
        instructions=system_prompt,
        tools=[search_tool.search],
    )
