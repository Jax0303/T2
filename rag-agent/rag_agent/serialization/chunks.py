# SPDX-License-Identifier: MIT
"""Explicit character/token splitting with source cell spans.

Only a complete value inside a chunk is credited as delivered evidence.
Headers and partial value fragments do not acquire a data-cell coordinate.
"""
from __future__ import annotations


def markdown_source(tab, table, name):
    text = f"Table name: {name}\n"
    spans = []
    for r, row in enumerate(tab.raw.get("texts") or []):
        values = [str(v) for v in row]
        if not any(v.strip() for v in values):
            continue
        text += "| "
        for c, value in enumerate(values):
            start = len(text)
            text += value
            i, j = tab.row_map.get(r), tab.col_map.get(c)
            if i is not None and j is not None and str(table.data[i][j]).strip():
                spans.append((start, len(text), (i, j)))
            text += " |" if c == len(values) - 1 else " | "
        text += "\n"
    return text.rstrip("\n"), spans


def split_chunks(text, size, overlap, tokenizer=None):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("chunk size must be positive and 0 <= overlap < size")
    if tokenizer is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap,
                                                  add_start_index=True)
        for doc in splitter.create_documents([text]):
            start = doc.metadata["start_index"]
            if start < 0 or text[start:start + len(doc.page_content)] != doc.page_content:
                raise ValueError("splitter failed to preserve a source span")
            yield doc.page_content, start, start + len(doc.page_content)
        return
    # Fixed windows over the declared tokenizer, with original character offsets.
    offsets = tokenizer(text, add_special_tokens=False, truncation=False,
                        return_offsets_mapping=True)["offset_mapping"]
    first = 0
    while first < len(offsets):
        end = min(first + size, len(offsets))
        start_char = offsets[first][0]
        end_char = offsets[end - 1][1]
        # Slicing in the middle of a word can change tokenization. Enforce the
        # actual chunk length, not just the length of its parent token window.
        while end > first + 1 and len(tokenizer(text[start_char:end_char],
                add_special_tokens=False)["input_ids"]) > size:
            end -= 1
            end_char = offsets[end - 1][1]
        chunk = text[start_char:end_char]
        if len(tokenizer(chunk, add_special_tokens=False)["input_ids"]) > size:
            raise ValueError("one source token cannot fit the declared token budget")
        yield chunk, start_char, end_char
        if end == len(offsets):
            break
        first = max(first + 1, end - overlap)


def table_chunks(tab, table, name, size=1000, overlap=200, tokenizer=None):
    text, spans = markdown_source(tab, table, name)
    for chunk, begin, end in split_chunks(text, size, overlap, tokenizer):
        cells = [cell for start, stop, cell in spans if begin <= start and stop <= end]
        yield f"File name: {name}\n{chunk}", cells
