# AI Engineering Assistant

An AI agent that answers questions about Chip Huyen's AI Engineering materials using retrieval, PydanticAI, Gemini, and Streamlit.

## Overview

AI Engineering Assistant is a small RAG-style application that searches the public [`chiphuyen/aie-book`](https://github.com/chiphuyen/aie-book) repository and uses the retrieved context to produce grounded answers with source references.

The project solves a common problem: when learning from a large collection of markdown notes, examples, and resources, it can be hard to quickly find the right section and turn it into a useful answer. This assistant builds a searchable index over the repository, lets an agent call that search index as a tool, and returns answers through a Streamlit chat interface.

The app currently supports:

- GitHub repository ingestion from markdown and MDX files.
- Section-based document chunking.
- Text search with `minsearch`.
- PydanticAI agent orchestration.
- Gemini model integration through Google AI Studio / Google Generative Language API.
- Streamlit chat UI.
- Conversation logging for later evaluation.
- LLM-as-judge evaluation and retrieval evaluation scripts.

## Installation

### Requirements

- Python 3.11 or newer
- A Google AI Studio Gemini API key
- `uv` or `pip`

The app expects a `GOOGLE_API_KEY` environment variable.

### Clone And Enter The Project

```bash
git clone <your-repo-url>
cd aihero/project
```

### Install Dependencies With `uv`

```bash
uv add streamlit pydantic-ai google-genai minsearch requests python-frontmatter
```

If you use `pip` instead:

```bash
pip install streamlit pydantic-ai google-genai minsearch requests python-frontmatter
```

### Set Your Gemini API Key

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
```

Run Streamlit from the same terminal session where you set the key.

## Usage

### Run The Streamlit App

From the `project` directory:

```bash
streamlit run streamlit_app.py
```

From the repository root:

```bash
streamlit run project/streamlit_app.py
```

Open the local Streamlit URL in your browser and ask questions such as:

```text
What are the most important steps to write good prompts?
```

```text
When should I use RAG instead of prompt engineering?
```

```text
What does the book say about evaluating AI applications?
```

### How The App Works

The main flow is:

```text
User question
-> PydanticAI agent
-> search tool
-> minsearch index over AI Engineering chunks
-> retrieved context
-> Gemini answer
-> Streamlit response
-> JSON log file
```

### Run Answer Evaluation

`evaluate.py` evaluates the latest saved log file in `project/logs`.

```bash
uv run python evaluate.py
```

This uses an LLM-as-judge checklist to evaluate whether the assistant followed instructions, gave a relevant answer, cited sources, and used the search tool.

### Run Retrieval Evaluation

`evaluate_retrieval.py` evaluates search quality with retrieval metrics such as precision, recall, hit rate, and MRR.

```bash
uv run python evaluate_retrieval.py
```

## Features

- **Repository ingestion:** Downloads a GitHub repository ZIP and parses markdown/MDX files with frontmatter.
- **Chunking strategies:** Supports simple character chunks, paragraph sliding windows, and markdown section chunks.
- **Search tool:** Wraps the `minsearch` index so the PydanticAI agent can search the indexed materials.
- **Grounded answers:** The prompt asks the agent to use retrieved context and cite source files.
- **Streamlit chat UI:** Provides an interactive browser-based interface.
- **Caching:** Uses caching so the index and agent are not rebuilt on every Streamlit rerun.
- **Logging:** Saves agent interactions as JSON logs for debugging and evaluation.
- **Evaluation:** Includes both LLM-as-judge answer evaluation and retrieval-focused metrics.

### Roadmap

- Improve answer quality with better chunk ranking and citation formatting.
- Add a persistent index so the app does not need to rebuild after server restarts.
- Add support for user-selected GitHub repositories.
- Add tests for ingestion, chunking, search tools, and logging.
- Add deployment configuration for Streamlit Community Cloud or another hosting platform.

## Contributing

Contributions are welcome. If you want to improve the project:

- Open an issue describing the problem or improvement.
- Keep functions small and focused.
- Avoid hardcoding API keys or secrets.
- Prefer readable, typed Python functions.
- Test changes locally before opening a pull request.

There is no separate `CONTRIBUTING.md` yet.

## Tests

There is not a formal test suite yet. For now, use the scripts below as smoke tests.

### Test Ingestion And Search

```bash
uv run python evaluate_retrieval.py
```

### Test The Streamlit App

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run streamlit_app.py
```

### Test Answer Evaluation

First generate at least one log through the Streamlit app, then run:

```bash
uv run python evaluate.py
```

## Deployment

For deployment, make sure:

- `GOOGLE_API_KEY` is configured as a secret or environment variable.
- The app is started with `streamlit run streamlit_app.py`.
- The index is cached so it is not rebuilt on every user request.
- Logs are written to a directory that exists and is writable.

For Streamlit Community Cloud, add the Gemini key through the app secrets/settings rather than hardcoding it in source code.

## FAQ / Troubleshooting

### `KeyError: 'GOOGLE_API_KEY'`

The Gemini API key is not visible to the running process. Set it before running the app:

```bash
export GOOGLE_API_KEY="your-gemini-api-key"
streamlit run streamlit_app.py
```

### `429 quota exceeded`

Your Gemini free-tier quota or rate limit has been reached. Wait for quota reset, switch models, reduce calls, or enable billing.

### `API key expired`

The Google API key is invalid, expired, deleted, or blocked. Create a new key in Google AI Studio and update `GOOGLE_API_KEY`.

### Logging Fails

Make sure `logs.py` contains `log_interaction_to_file()` and that the `logs/` directory exists. Restart Streamlit after changing logging code because Python may keep older imports cached.

### Answers Feel Too Generic

Try improving the system prompt, increasing `num_results` in `SearchTool`, cleaning search results before returning them to the model, or using a stronger Gemini model if quota allows.

### `main.py` Function Name Mismatch

The Streamlit app uses `agent_building.build_agent()`. If you use the CLI-style `main.py`, make sure it calls the function that exists in your current `agent_building.py`.
