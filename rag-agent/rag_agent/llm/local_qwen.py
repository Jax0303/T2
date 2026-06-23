"""Local HF model backend (default: Qwen2.5-7B-Instruct, 4-bit on CUDA).

Self-contained loader (no external package imports).
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import BaseLLM

logger = logging.getLogger(__name__)


class LocalQwenLLM(BaseLLM):
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        dtype: str = "bfloat16",
        device: Optional[str] = None,
        quantization: Optional[str] = "4bit",   # None | "4bit" | "8bit"
        default_max_tokens: int = 256,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name = f"local:{model_name}"
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        load_kwargs = {"torch_dtype": torch_dtype, "device_map": "auto", "low_cpu_mem_usage": True}
        if quantization in {"4bit", "8bit"} and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        self.model.eval()
        self._torch = torch
        logger.info("LocalQwenLLM loaded %s on %s (quant=%s)", model_name, self.device, quantization)

    def complete(self, system: str, user: str, max_tokens: int = 256,
                 temperature: float = 0.0, top_p: float = 0.95) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=max_tokens or self.default_max_tokens,
                          pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
        else:
            gen_kwargs.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()
