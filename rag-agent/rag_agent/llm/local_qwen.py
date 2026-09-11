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
        revision: Optional[str] = None,
        retry_on_429: int = 0,  # inert: no rate limit locally. Accepted so a
                                # caller can hand the same kwargs to any backend.
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # the quantization is part of the reader identity, not a loading detail:
        # 4-bit and 8-bit of the same weights answer differently, and
        # guard_resume has to refuse to join their records
        self.name = f"local:{model_name}" + (
            f"?quantization={quantization}" if quantization else "")
        self.model_name = model_name
        self.default_max_tokens = default_max_tokens
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        torch_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype

        # a typo used to mean "full precision" silently, which changes the
        # experiment rather than failing it
        if quantization not in {None, "none", "4bit", "8bit"}:
            raise ValueError(f"quantization must be none/4bit/8bit, got {quantization!r}")
        if quantization in {"4bit", "8bit"} and not self.device.startswith("cuda"):
            raise ValueError("requested quantization requires CUDA; use quantization=none explicitly on CPU")
        self.quantization = quantization if quantization != "none" else None
        self.revision = revision

        load_kwargs = {"torch_dtype": torch_dtype, "device_map": {"": self.device}, "low_cpu_mem_usage": True}
        if quantization in {"4bit", "8bit"}:
            from transformers import BitsAndBytesConfig
            if quantization == "4bit":
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True, bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4",
                )
            else:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, revision=revision, **load_kwargs)
        self.model.eval()
        self._torch = torch
        logger.info("LocalQwenLLM loaded %s on %s (quant=%s)", model_name, self.device, quantization)

    def metadata(self) -> dict:
        return {"name": self.name, "revision_requested": self.revision,
                "revision_resolved": getattr(self.model.config, "_commit_hash", None),
                "quantization": self.quantization, "device": self.device,
                "dtype": str(self.model.dtype), "context_limit": self.context_limit,
                "chat_template": self.tokenizer.chat_template, "enable_thinking": False}

    def _prompt(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Qwen3 templates default to thinking mode: the model emits
        # "<think>...</think>" before the answer, which both eats the
        # max_new_tokens budget and lands inside the string the EM scorer reads.
        # enable_thinking=False makes the template pre-close the block. Verified
        # BYTE-IDENTICAL on Qwen2.5-7B-Instruct (its template ignores the flag),
        # so every result already on disk stays reproducible from this code.
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )

    def n_prompt_tokens(self, system: str, user: str) -> int:
        """생성에 들어가는 프롬프트의 토큰 수 — 채팅 템플릿까지 포함한 실측값.

        입력은 어디서도 잘리지 않는다(`complete` 가 truncation 을 걸지 않는다).
        그래서 한계를 넘는 프롬프트는 조용히 짧아지는 대신 실패하거나 OOM 이
        되고, 조건별로 그 건수를 세려면 이 값이 필요하다.
        """
        return len(self.tokenizer(self._prompt(system, user))["input_ids"])

    @property
    def context_limit(self) -> int:
        return int(getattr(self.model.config, "max_position_embeddings", 0)) or 0

    def complete(self, system: str, user: str, max_tokens: int = 256,
                 temperature: float = 0.0, top_p: float = 0.95) -> str:
        prompt = self._prompt(system, user)
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
