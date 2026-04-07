# project/chunking.py
import re


def sliding_window(seq, size, step):
    if size <= 0 or step <= 0:
        raise ValueError("size and step must be positive")

    chunks = []

    for start in range(0, len(seq), step):
        chunk = seq[start:start + size]

        if not chunk:
            break

        chunks.append({
            "start": start,
            "chunk": chunk,
        })

        if start + size >= len(seq):
            break

    return chunks


def simple_character_chunking(docs, size=2000, step=1000):
    chunked_docs = []

    for doc in docs:
        doc_copy = doc.copy()
        content = doc_copy.pop("content", "")

        chunks = sliding_window(content, size=size, step=step)

        for chunk_id, chunk in enumerate(chunks):
            chunk_doc = doc_copy.copy()
            chunk_doc.update(chunk)
            chunk_doc["chunk_id"] = chunk_id
            chunk_doc["chunking_strategy"] = "simple_character"
            chunked_docs.append(chunk_doc)

    return chunked_docs


def split_paragraphs(text):
    return [
        paragraph.strip()
        for paragraph in re.split(r"\n\s*\n", text.strip())
        if paragraph.strip()
    ]


def paragraph_sliding_window_chunking(docs, window_size=5, step=3):
    chunked_docs = []

    for doc in docs:
        doc_copy = doc.copy()
        content = doc_copy.pop("content", "")
        paragraphs = split_paragraphs(content)

        for chunk_id, start in enumerate(range(0, len(paragraphs), step)):
            window = paragraphs[start:start + window_size]

            if not window:
                break

            chunk_doc = doc_copy.copy()
            chunk_doc["start_paragraph"] = start
            chunk_doc["chunk"] = "\n\n".join(window)
            chunk_doc["chunk_id"] = chunk_id
            chunk_doc["chunking_strategy"] = "paragraph_sliding_window"
            chunked_docs.append(chunk_doc)

            if start + window_size >= len(paragraphs):
                break

    return chunked_docs


def split_markdown_by_level(text, level=2):
    header_pattern = r"^(#{" + str(level) + r"} )(.+)$"
    pattern = re.compile(header_pattern, re.MULTILINE)

    parts = pattern.split(text)
    sections = []

    intro = parts[0].strip()
    if intro:
        sections.append({
            "title": "Introduction",
            "chunk": intro,
        })

    for i in range(1, len(parts), 3):
        header = (parts[i] + parts[i + 1]).strip()
        content = ""

        if i + 2 < len(parts):
            content = parts[i + 2].strip()

        section = f"{header}\n\n{content}" if content else header

        sections.append({
            "title": parts[i + 1].strip(),
            "chunk": section,
        })

    return sections


def section_chunking(docs, level=2):
    chunked_docs = []

    for doc in docs:
        doc_copy = doc.copy()
        content = doc_copy.pop("content", "")

        sections = split_markdown_by_level(content, level=level)

        if not sections and content.strip():
            sections = [{
                "title": "Document",
                "chunk": content.strip(),
            }]

        for chunk_id, section in enumerate(sections):
            chunk_doc = doc_copy.copy()
            chunk_doc["section_title"] = section["title"]
            chunk_doc["chunk"] = section["chunk"]
            chunk_doc["chunk_id"] = chunk_id
            chunk_doc["chunking_strategy"] = "section"
            chunked_docs.append(chunk_doc)

    return chunked_docs
