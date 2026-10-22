"""로컬 백엔드 — google-research `table_rag` 를 외부 API 없이 돌리기 위한 같은 호출면의 대체 클래스.
(2026-10-23 재작성: 집 PC 의 원본 vendor/ 가 이 PC 에 없어 `results/mh_only_20260928/baselines/tablerag_chen/README.md` 의
"원설정과 다른 것" 1·2·6 행과 `scripts/baselines/tablerag_chen.py` 가 쓰는 호출면대로 다시 만들었다.)

- LocalQwen: OpenAI 클라이언트 모양(`chat.completions.create` → `.choices[0].message.content`)으로 `rag_agent.llm.LocalQwenLLM` 을 감싼다.
  사용자 메시지 하나(시스템 없음), greedy(temperature 0), 정지열(`stop`)은 생성 중 멈추고 출력에서 잘라낸다(OpenAI 의미와 같음).
  모델 사양 `local:<hf id>?quantization=4bit|none`(기본 4bit — 2026-10-06 패치: `_split_spec`).
- HuggingFaceEmbeddings: sentence-transformers 인코딩(모델 자체 풀링+정규화, 질의 접두어 없음). langchain_community 의 같은 이름 클래스 대체.
- FAISS / Document / VectorRetriever: langchain_community FAISS(IndexFlatL2 전수 L2) 대체. 캐시 = 디렉터리에 `vecs.npy` + `docs.json`.
  `as_retriever(search_kwargs={'k'})` → `.invoke(query)`(문서 k개), `.search(query)`([(문서, L2 거리)]), `.store.index.ntotal`.
"""
from __future__ import annotations

import json
import os
from types import SimpleNamespace

import numpy as np


class Document:
    def __init__(self, page_content: str, metadata: dict | None = None):
        self.page_content = page_content
        self.metadata = metadata or {}

    def __repr__(self):
        return f"Document(page_content={self.page_content!r})"


class HuggingFaceEmbeddings:
    def __init__(self, model_name: str, device: str | None = None, batch_size: int = 256):
        from sentence_transformers import SentenceTransformer
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed_documents(self, texts):
        if not texts:
            return []
        v = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                              normalize_embeddings=True, show_progress_bar=False)
        return v.astype(np.float32).tolist()

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class VectorRetriever:
    def __init__(self, store: "FAISS", k: int):
        self.store, self.k = store, k

    def search(self, query):
        return self.store.similarity_search_with_score(query, k=self.k)

    def invoke(self, query):
        return [d for d, _ in self.search(query)]

    get_relevant_documents = invoke


class FAISS:
    def __init__(self, index, docs, embedder):
        self.index, self.docs, self.embedder = index, docs, embedder

    @classmethod
    def from_documents(cls, docs, embedder):
        import faiss
        vecs = np.asarray(embedder.embed_documents([d.page_content for d in docs]), dtype=np.float32)
        index = faiss.IndexFlatL2(vecs.shape[1] if len(vecs) else 1)
        if len(vecs):
            index.add(vecs)
        return cls(index, list(docs), embedder)

    def save_local(self, folder):
        import faiss
        os.makedirs(folder, exist_ok=True)
        n = self.index.ntotal
        vecs = faiss.rev_swig_ptr(self.index.get_xb(), n * self.index.d).reshape(n, self.index.d).copy() if n else np.zeros((0, self.index.d), np.float32)
        np.save(os.path.join(folder, "vecs.npy"), vecs)
        with open(os.path.join(folder, "docs.json"), "w", encoding="utf-8") as f:
            json.dump([{"page_content": d.page_content, "metadata": d.metadata} for d in self.docs], f, ensure_ascii=False)

    @classmethod
    def load_local(cls, folder, embedder, allow_dangerous_deserialization=True):
        import faiss
        vecs = np.load(os.path.join(folder, "vecs.npy")).astype(np.float32)
        docs = [Document(d["page_content"], d.get("metadata")) for d in json.load(open(os.path.join(folder, "docs.json"), encoding="utf-8"))]
        index = faiss.IndexFlatL2(vecs.shape[1] if vecs.ndim == 2 and vecs.shape[1] else 1)
        if len(vecs):
            index.add(vecs)
        return cls(index, docs, embedder)

    def similarity_search_with_score(self, query, k=5):
        if self.index.ntotal == 0:
            return []
        q = np.asarray([self.embedder.embed_query(query)], dtype=np.float32)
        dist, idx = self.index.search(q, min(k, self.index.ntotal))
        return [(self.docs[i], float(d)) for d, i in zip(dist[0], idx[0]) if i >= 0]

    def as_retriever(self, search_kwargs=None):
        return VectorRetriever(self, int((search_kwargs or {}).get("k", 5)))


class _Completions:
    def __init__(self, owner):
        self._o = owner

    def create(self, model=None, messages=None, temperature=0.0, top_p=0.95, max_tokens=256, stop=None, n=1, **kwargs):
        system = ""
        user = ""
        for m in messages or []:
            if m.get("role") == "system":
                system = m.get("content") or ""
            elif m.get("role") == "user":
                user = m.get("content") or ""
        stops = list(stop) if isinstance(stop, (list, tuple)) else ([stop] if stop else [])
        self._o.n_calls += 1
        text = self._o.llm.complete(system, user, max_tokens=max_tokens, temperature=temperature or 0.0, top_p=top_p, stop=stops or None)
        for s in stops:
            i = text.find(s)
            if i >= 0:
                text = text[:i]
        text = text.rstrip()
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))], model=self._o.name)


class LocalQwen:
    """OpenAI 클라이언트 모양의 로컬 리더. `spec` = 'local:<hf id>[?quantization=4bit|none]'."""

    def __init__(self, spec: str, llm=None):
        from rag_agent.llm.factory import _split_spec, build_llm
        from rag_agent.llm.local_qwen import LocalQwenLLM
        if spec.startswith("vllm:"):                                   # 2026-10-23: 로컬 vLLM 서버 경로(같은 프롬프트 렌더링, 정지열 서버 처리)
            self.llm = llm or build_llm(spec)
            model_name = spec.split(":", 1)[1]
        else:
            model_name = spec.split(":", 1)[1] if spec.startswith("local:") else spec
            name, kw = _split_spec(model_name)
            self.llm = llm or LocalQwenLLM(model_name=name, **{"quantization": "4bit", **kw})
        self.name = getattr(self.llm, "name", f"local:{model_name}")
        self.tokenizer = self.llm.tokenizer
        self.n_calls = 0
        self.chat = SimpleNamespace(completions=_Completions(self))

    @property
    def context_limit(self):
        return getattr(self.llm, "context_limit", 0)
