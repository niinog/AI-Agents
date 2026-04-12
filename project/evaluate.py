import json
import os
from pydantic import BaseModel
from pydantic_ai import Agent
from pathlib import Path

from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider







class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    summary: str
    checklist: list[EvaluationCheck]



def build_eval_agent():
    provider = GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"])


    model = GoogleModel(
        "gemini-2.5-flash-lite",
        provider=provider,
    )

    eval_system_prompt = """
You are an evaluator for a RAG course assistant.

Your job is to judge whether the assistant answered correctly using the provided conversation log.

Evaluate the answer using these checks:
- instructions_follow: Did the assistant follow the system instructions?
- instructions_avoid: Did the assistant avoid unsupported or forbidden behavior?
- answer_relevant: Is the answer relevant to the user's question?
- answer_clear: Is the answer clear and easy to understand?
- answer_citations: Did the answer mention useful source information such as section title or filename when available?
- completeness: Does the answer cover the important points from the retrieved context?
- tool_call_search: Did the assistant use the search tool before answering?

Be fair but strict. If the answer misses required citations or ignores search results, mark the related check as false.
"""

    return Agent(
        model=model,
        name="aie_book_eval_agent",
        instructions=eval_system_prompt,
    )


def make_eval_prompt(instructions: str, question: str, answer: str, log: str) -> str:
    return f"""
Evaluate this AI assistant interaction.

Original system instructions:
{instructions}

User question:
{question}

Assistant answer:
{answer}

Full conversation log:
{log}

Return an evaluation checklist.
"""




def get_latest_log_file() -> Path:
    log_dir = Path(__file__).parent / "logs"
    log_files = list(log_dir.glob("*.json"))

    if not log_files:
        raise FileNotFoundError(f"No log files found in {log_dir}")

    return max(log_files, key=lambda path: path.stat().st_mtime)




def load_log_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_log_file(filepath):
    log_record = load_log_file(filepath)

    instructions = log_record["system_prompt"]
    question = log_record["messages"][0]["parts"][0]["content"]
    answer = log_record["messages"][-1]["parts"][0]["content"]
    log = json.dumps(log_record["messages"], indent=2)

    user_prompt = make_eval_prompt(
        instructions=instructions,
        question=question,
        answer=answer,
        log=log,
    )

    eval_agent = build_eval_agent()

    result = eval_agent.run_sync(
        user_prompt,
        output_type=EvaluationChecklist,
    )

    return result.output


def main():
    filepath = get_latest_log_file()
    print(f"Evaluating log file: {filepath}")

    checklist = evaluate_log_file(filepath)

    print("\nEVALUATION SUMMARY")
    print(checklist.summary)

    print("\nCHECKLIST")
    for check in checklist.checklist:
        print(check)


if __name__ == "__main__":
    main()