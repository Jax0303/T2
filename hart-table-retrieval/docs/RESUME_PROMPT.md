# 연구실 PC에서 작업 이어할 때 쓰는 프롬프트

아래 텍스트를 통째로 복사해서 Claude Code에 첫 메시지로 붙여넣으면 됩니다.

---

```
HART 본 프로젝트는 마무리됐고, 이제 사이드 프로젝트인 "Sidecar Verifier Agent"를 이어 작업한다.

[리포지토리]
- github.com/Jax0303/T2, 브랜치 main, 최신 커밋까지 push 완료.
- 작업 디렉토리는 hart-table-retrieval/.
- 아키텍처 다이어그램: docs/sidecar_architecture.png
- 주요 결과는 hart-table-retrieval/README.md + sidecar_verifier/README.md 에 정리돼 있음.

[코드 상태]
- HART 파이프라인 (scripts/run_indexing → run_retrieval → run_evaluation → run_ablation → token_length_control) 완료.
  주요 결과: plain_markdown이 R@1=0.609로 HART-full(0.460)을 압도. HART scorer는 negative result.
- 사이드 프로젝트 (sidecar_verifier/) v2 완성. query→table keyword/numeric overlap을 rerank 신호로 사용.
  - HiTab dev 300q: 모든 시리얼라이저에서 vector 대비 일관 lift. plain_markdown +12.3pp R@1.
  - TARGET 벤치마크: FeTaQA +1.5pp R@10, TabFact -1.2pp R@10 (negative — 균일 구조 한계).
  - end-to-end (Qwen2.5-3B-Instruct 4-bit): retrieval은 +6.7pp R@1 개선되지만 answer accuracy는 LLM 자체 한계로 7%만 도달.

[환경 재셋업 (연구실 PC, 처음이라면)]
1. 리포 clone:
   git clone https://github.com/Jax0303/T2.git
   cd T2/hart-table-retrieval

2. 데이터 받기:
   mkdir -p data && cd data
   git clone --depth=1 https://github.com/microsoft/HiTab.git hitab
   python3 -c "import zipfile; zipfile.ZipFile('hitab/data/tables.zip').extractall('hitab/data')"
   cd ..

3. (TARGET 벤치마크 평가용)
   git clone --depth=1 https://github.com/target-benchmark/target.git target_bench
   # target_bench/target_benchmark/retrievers/__init__.py 를 슬림하게 수정:
   # from .AbsRetrieverBase, AbsCustomEmbeddingRetriever, AbsStandardEmbeddingRetriever 만 import.

4. venv + 패키지 (GPU 있는 환경 가정):
   python3 -m venv .venv  (만약 ensurepip 에러면: python3 -m venv --without-pip .venv && curl -sS https://bootstrap.pypa.io/get-pip.py | .venv/bin/python3)
   source .venv/bin/activate
   pip install torch --index-url https://download.pytorch.org/whl/cu126
   pip install chromadb sentence-transformers numpy pandas pyyaml tiktoken matplotlib scipy openai \
       accelerate bitsandbytes tabulate \
       datasets pydantic python-dotenv func-timeout qdrant-client evaluate rouge-score sacrebleu \
       langchain langchain-community langchain-core langchain-openai langchain-text-splitters

5. configs/experiment.yaml의 paths를 본인 환경에 맞게 수정 (data_dir, chroma_dir, hf_cache).

[해야 할 일 (남은 작업, 우선순위)]
1. **OTTQA TARGET 평가** (아직 못 돌림 — 코퍼스가 큼).
     python sidecar_verifier/eval/target_run.py --dataset ottqa --top-k 10
   가설: BM25가 OTTQA에서 R@10=0.967 찍는 강한 keyword 신호 데이터셋 → 우리 verifier가 가장 큰 lift를 줄 가능성.
   결과를 README의 "TARGET benchmark — generalization check" 표에 추가.

2. **TabFact regression 원인 분석 + filter 개선** (sidecar_verifier/agent/verifier.py).
   - 가설: 균일한 column name + 잦은 수치 중복으로 false positive.
   - 시도: (a) column-name TF-IDF 가중치, (b) numeric overlap을 confidence가 아닌 *cutoff*로만 사용,
           (c) header keyword 매칭에 leaf-only restrict.
   - target_run.py를 모드 인자 받게 확장 (filter, rerank, filter+rerank).

3. **답변 정확도 끌어올리기** (sidecar_verifier/agent/answerer.py).
   - 현재 Qwen2.5-3B 4-bit으로 7%. Oracle 10%.
   - 옵션 (a) Qwen 2.5-7B-Instruct 4-bit (~5GB VRAM. 연구실 GPU면 여유로움).
   - 옵션 (b) Groq 무료 API (Llama-3.3-70B): answerer.py에 별도 GroqAnswerer 클래스 추가.
   - 옵션 (c) TabSQLify-style cell-pre-extraction: LLM에 전체 markdown 대신 verifier가 찾은 cell 후보들만 전달.

4. **Tracer 버그 점검**.
   - sidecar_verifier/agent/tracer.py에서 LLM이 출력한 숫자가 cell에 있어도 grounded=False로 나오는 케이스 있음.
   - find_value()의 int/float coerce 또는 NaN 처리 의심. 작은 unit test 추가.

[참고]
- 연구실 PC는 데이터/모델 캐시가 D 드라이브에 있을 것이니, configs/experiment.yaml의 paths를 거기로 맞추기.
- ~/.claude/projects/-home-user-T2/memory/ 에 작업 기록 있음 (hart_project, side_project_verifier_agent, env_quirks).
- 메모리에 기록되지 않은 직전 상태: TabFact 결과 분석 직후 OTTQA 시작 직전.

먼저 위 4가지 중 어디부터 손볼지 같이 정하고 진행하자.
```

---

## 토큰/보안 메모

- 이 리포지토리에 push할 때 쓴 GitHub PAT가 이전 대화 컨텍스트에 노출된 적 있습니다.
  **반드시 [GitHub Settings → Tokens](https://github.com/settings/tokens) 에서 해당 토큰을 revoke 하고 새로 발급**하세요.
- 연구실에서 다시 push 할 때는 새 토큰을 환경변수 (`export GITHUB_TOKEN=...`) 또는 `gh auth login` 으로 관리.
- 토큰을 `.git/config`, `.env`, 코드, 문서 어디에도 평문으로 두지 마세요.
