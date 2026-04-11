import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent_building import build_agent


LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)


class EvaluationCheck(BaseModel):
    check_name: str
    justification: str
    check_pass: bool


class EvaluationChecklist(BaseModel):
    summary: str
    checklist: list[EvaluationCheck]


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def log_entry(agent, messages, source="user"):
    return {
        "agent_name": agent.name,
        "source": source,
        "system_prompt": agent.instructions,
        "messages": ModelMessagesTypeAdapter.dump_python(
            messages,
            mode="json",
        ),
    }


def log_interaction_to_file(agent, messages, source="user"):
    entry = log_entry(agent, messages, source)

    ts = entry["messages"][-1]["timestamp"]

    if isinstance(ts, datetime):
        ts_obj = ts
    else:
        ts_obj = datetime.fromisoformat(ts.replace("Z", "+00:00"))

    ts_str = ts_obj.strftime("%Y%m%d_%H%M%S")
    rand_hex = secrets.token_hex(3)

    filename = f"{agent.name}_{ts_str}_{rand_hex}.json"
    filepath = LOG_DIR / filename

    with filepath.open("w", encoding="utf-8") as f_out:
        json.dump(entry, f_out, indent=2, default=serializer)

    return filepath


def build_eval_agent():
    provider = GoogleProvider(api_key=os.environ["GOOGLE_API_KEY"])

    model = GoogleModel(
        "gemini-2.0-flash-lite",
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




def main():
    question = "What are best practices for prompt engineering?"

    agent = build_agent()

    result = agent.run_sync(user_prompt=question)
    answer = result.output

    print("ANSWER")
    print(answer)

    log_path = log_interaction_to_file(agent, result.new_messages())
    print(f"\nSaved log to: {log_path}")

    log_record = json.loads(log_path.read_text(encoding="utf-8"))

    instructions = log_record["system_prompt"]
    question_from_log = log_record["messages"][0]["parts"][0]["content"]
    answer_from_log = log_record["messages"][-1]["parts"][0]["content"]
    log = json.dumps(log_record["messages"], indent=2)

    user_prompt = make_eval_prompt(
        instructions=instructions,
        question=question_from_log,
        answer=answer_from_log,
        log=log,
    )

    eval_agent = build_eval_agent()

    eval_result = eval_agent.run_sync(
        user_prompt,
        output_type=EvaluationChecklist,
    )

    checklist = eval_result.output

    print("\nEVALUATION SUMMARY")
    print(checklist.summary)

    print("\nCHECKLIST")
    for check in checklist.checklist:
        print(check)




if __name__ == "__main__":
    main()
