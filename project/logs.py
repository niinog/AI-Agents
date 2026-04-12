import json
import os
import secrets
from datetime import datetime
from pathlib import Path

from pydantic_ai.messages import ModelMessagesTypeAdapter


LOG_DIR = Path(os.getenv("LOGS_DIRECTORY", "logs"))
LOG_DIR.mkdir(exist_ok=True)


def get_agent_tools(agent) -> list[str]:
    tools = []

    for toolset in getattr(agent, "toolsets", []):
        tool_map = getattr(toolset, "tools", {})
        tools.extend(tool_map.keys())

    return tools


def get_model_name(agent) -> str:
    model = getattr(agent, "model", None)

    if model is None:
        return "unknown"

    return (
        getattr(model, "model_name", None)
        or getattr(model, "_model_name", None)
        or str(model)
    )


def log_entry(agent, messages, system_prompt: str, source: str = "user") -> dict:
    dict_messages = ModelMessagesTypeAdapter.dump_python(
        messages,
        mode="json",
    )

    return {
        "agent_name": agent.name,
        "system_prompt": system_prompt,
        "provider": "google",
        "model": get_model_name(agent),
        "tools": get_agent_tools(agent),
        "messages": dict_messages,
        "source": source,
    }


def serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()

    raise TypeError(f"Type {type(obj)} not serializable")


def log_interaction_to_file(agent, messages, system_prompt: str, source: str = "user") -> Path:
    entry = log_entry(
        agent=agent,
        messages=messages,
        system_prompt=system_prompt,
        source=source,
    )

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
