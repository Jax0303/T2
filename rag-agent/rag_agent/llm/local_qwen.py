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

    def _prompt(self, system: str, user: str, thinking: bool = False) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Qwen3 templates default to thinking mode: the model emits
        # "<think>...</think>" before the answer, which both eats the
        # max_new_tokens budget and lands inside the string the EM scorer reads.
        # enable_thinking=False makes the template pre-close the block. Verified
        # BYTE-IDENTICAL on Qwen2.5-7B-Instruct (its template ignores the flag),
        # so every result already on disk stays reproducible from this code.
        # thinking=True is Qwen3's own recommendation for math (model card, Best Practices).
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=thinking,
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
                 temperature: float = 0.0, top_p: float = 0.95, top_k: Optional[int] = None,
                 min_p: Optional[float] = None, thinking: bool = False) -> str:
        """thinking=True 이면 ``</think>`` 뒤의 답만 돌려준다(모델 카드의 파싱 그대로, 토큰 151668).
        생성 토큰 수와 생각 블록이 닫혔는지는 ``last_generation`` 에 남긴다."""
        prompt = self._prompt(system, user, thinking)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs = dict(max_new_tokens=max_tokens or self.default_max_tokens,
                          pad_token_id=self.tokenizer.eos_token_id)
        if temperature and temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=temperature, top_p=top_p)
            if top_k is not None:
                gen_kwargs["top_k"] = top_k
            if min_p is not None:
                gen_kwargs["min_p"] = min_p
        else:
            gen_kwargs.update(do_sample=False)
        with self._torch.inference_mode():
            out = self.model.generate(**inputs, **gen_kwargs)
        gen = out[0, inputs["input_ids"].shape[1]:].tolist()
        cut = 0
        if thinking:
            try:
                cut = len(gen) - gen[::-1].index(151668)          # </think>
            except ValueError:
                cut = 0                                          # 생각이 한도 안에 안 닫힘
        self.last_generation = {"new_tokens": len(gen), "think_tokens": cut,
                                "think_closed": (not thinking) or cut > 0}
        return self.tokenizer.decode(gen[cut:], skip_special_tokens=True).strip()

    def complete_batch(self, system: str, users: list, max_tokens: int = 256) -> list:
        """비생각·greedy 전용 continuous batching(`generate_batch`) — 끝난 문항 자리에 다음 문항을
        바로 넣는다. 8GB 에서 고정 묶음은 긴 입력이 메모리를 넘겨 쓰지 않았다. 배치 1 과 연산 경로가
        달라 ``complete`` 와 출력이 바이트 단위로 같다는 보장은 없다 — 쓰기 전에 대조한다."""
        from transformers import ContinuousBatchingConfig, GenerationConfig
        tok = self.tokenizer
        ids = [tok(self._prompt(system, u))["input_ids"] for u in users]
        cfg = GenerationConfig(max_new_tokens=max_tokens, do_sample=False,
                               eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
        out = self.model.generate_batch(inputs=ids, generation_config=cfg, progress_bar=False,
                                        continuous_batching_config=ContinuousBatchingConfig(
                                            max_memory_percent=0.9))
        # 요청 id 는 'req_<입력 순번>' — 순번으로 되돌린다
        keys = sorted(out, key=lambda k: int(k.rsplit("_", 1)[1]))
        if len(keys) != len(ids):
            raise RuntimeError(f"generate_batch 가 {len(ids)}건 중 {len(keys)}건만 돌려줬다")
        return [tok.decode(out[k].generated_tokens, skip_special_tokens=True).strip() for k in keys]
