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


def _format_table_for_llm(rec: TableRecord, max_rows: int = 30) -> str:
    """Render the table as a compact markdown block plus header context."""
    df = rec.df
    n_rows = min(df.shape[0], max_rows)
    truncated = df.iloc[:n_rows]
    md = truncated.to_markdown(index=True) if hasattr(truncated, "to_markdown") else truncated.to_string()
    note = "" if n_rows == df.shape[0] else f"\n(...{df.shape[0] - n_rows} more rows truncated)"
    headers = "\n".join(
        f"  col[{i}]: " + " > ".join(p) for i, p in enumerate(rec.top_header_paths[:30])
    )
    return (
        f"Title: {rec.title}\n"
        f"Header paths (top):\n{headers}\n\n"
        f"Data:\n{md}{note}"
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
        quantization: Optional[str] = "4bit",  # None | "4bit" | "8bit"
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
            "If the answer is a number, output just the number (no units, no commas). "
            "If multiple numbers, separate with ', '. "
            "If not answerable from the table, output 'N/A'."
        )
        user = f"Table:\n{table_block}\n\nQuestion: {query}\n\nAnswer:"
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

        # Clean common LLM filler: take first non-empty line, strip leading "Answer:" etc.
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), text)
        first_line = re.sub(r"^(answer|the answer is|=|:)\s*", "", first_line, flags=re.IGNORECASE)
        return AnswerResult(answer=first_line, raw_output=text, table_id=rec.table_id)
