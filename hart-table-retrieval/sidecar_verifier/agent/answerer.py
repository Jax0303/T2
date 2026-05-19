"""Local LLM answer generation over the top-1 verified table.

Uses Hugging Face transformers; default model is Qwen2.5-3B-Instruct.
Loads on first use, kept warm across queries.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import torch

from ..store.table_store import TableRecord


def _clean_path(p):
    """Drop synthetic root tags like <TOP>/<LEFT>; keep meaningful segments."""
    return [s for s in p if s and s not in ("<TOP>", "<LEFT>")]


def _format_table_for_llm(rec: TableRecord, max_rows: int = 40) -> str:
    """Render the table so an LLM can locate rows by left-header and cols by full top-header path.

    Fixes two failure modes seen on HiTab:
      - top_header_paths have multiple levels (e.g. 'black male workers' > 'immigrant'),
        but the original render only used the leaf -> 8 duplicate 'immigrant' columns.
      - left_header_paths were not surfaced into the markdown -> rows were just 0..N.
    """
    import pandas as pd

    df = rec.df.copy()
    n_rows = min(df.shape[0], max_rows)
    df = df.iloc[:n_rows].reset_index(drop=True)

    # 1) Build column headers from full top_header_paths.
    top_paths = [_clean_path(p) for p in rec.top_header_paths]
    if len(top_paths) == df.shape[1]:
        df.columns = [
            " :: ".join(p) if len(p) > 1 else (p[0] if p else f"col_{i}")
            for i, p in enumerate(top_paths)
        ]

    # 2) Prepend left-header leaf (or full path if helpful) as a real column.
    left_paths = [_clean_path(p) for p in rec.left_header_paths]
    if left_paths:
        row_labels = []
        for i in range(n_rows):
            if i < len(left_paths) and left_paths[i]:
                p = left_paths[i]
                row_labels.append(p[-1] if len(p) == 1 else " :: ".join(p))
            else:
                row_labels.append(f"row_{i}")
        df.insert(0, "row_header", row_labels)

    md = df.to_markdown(index=False) if hasattr(df, "to_markdown") else df.to_string(index=False)
    note = "" if n_rows == rec.df.shape[0] else f"\n(...{rec.df.shape[0] - n_rows} more rows truncated)"

    # Compact header summary for the model to scan first.
    header_block_lines = []
    for i, p in enumerate(top_paths[:30]):
        header_block_lines.append(f"  col[{i}]: " + " > ".join(p) if p else f"  col[{i}]: (empty)")
    headers = "\n".join(header_block_lines)

    return (
        f"Title: {rec.title}\n"
        f"Top header paths:\n{headers}\n\n"
        f"Data (row_header column is the left-side row label, then numeric cells):\n{md}{note}"
    )


@dataclass
class AnswerResult:
    answer: str
    raw_output: str
    table_id: str


class LocalLLMAnswerer:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        max_new_tokens: int = 96,
        quantization: Optional[str] = "4bit",  # None | "4bit" :: "8bit"
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        load_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", "low_cpu_mem_usage": True}
        if quantization in {"4bit", "8bit"} and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()

    def answer(self, query: str, rec: TableRecord) -> AnswerResult:
        table_block = _format_table_for_llm(rec)
        system = (
            "You are a precise table QA assistant. Answer ONLY from the table below. "
            "Many questions require selecting a subset of rows or columns and computing a "
            "value (sum, difference, ratio, max/min, argmax, sign change). "
            "Think step by step using `Reasoning:` then give the final answer after "
            "`Final answer:` on its own line. "
            "If the final answer is a number, output just the number (no units, no commas, "
            "no '%'); for fractions/percentages match the form used in the question's data. "
            "If multiple numbers, separate with ', '. "
            "If not answerable, write `Final answer: N/A`."
        )
        user = f"Table:\n{table_block}\n\nQuestion: {query}"
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = out[0, inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Extract content after "Final answer:" if the CoT format is followed; else fall back.
        m = re.search(r"final\s*answer\s*[:\-]\s*(.+?)(?:\n|$)", text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            cleaned = m.group(1).strip()
        else:
            cleaned = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
        cleaned = re.sub(r"^(answer|the answer is|=|:)\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = cleaned.rstrip(".").strip()
        return AnswerResult(answer=cleaned, raw_output=text, table_id=rec.table_id)
