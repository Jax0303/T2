# 연구실 재개용 메모 (RESUME)

작성: 2026-06-22 · 브랜치: `claude/wizardly-ride-4AsIo`

> Claude Code에 붙여넣어 작업 재개할 때 참고. 아래 "재개 프롬프트"를 복사해 쓰면 됨.

---

## 0. 한 줄 상태
교수님 "표 구조 메타데이터로 binding 해결" 가설을 **검증 완료**. 메타는 binding 실패(NO_MATCH)를 박멸하지만 정답률 천장은 LLM 계산. 검색 지표는 R@1 대신 **full-set recall@k + MRR**로 봐야 함. 다음 과제 = **reranking**.

---

## 1. 무엇을 했나 (검증된 사실)

1. **메타데이터 저장 구현** — `scripts/build_table_meta.py` → `data/table_meta/{bench}/{tid}.json`
   (n_rows/cols, **row_depth/col_depth**, row_paths, col_paths, **cell_paths**=셀별 풀경로+값). cap 없음.
   규모: hitab 540(depth≤4, 계층 472) / finqa 883(flat) / wikisql 2630(행헤더 없음).

2. **binding 가설 검증** — `scripts/binding_eval.py --meta-store data/table_meta`
   (naive=메타X vs grounded=메타O, 검색=gold 고정, Qwen7B 동일모델)
   - NO_MATCH: hitab 0.80→0.14, finqa 0.84→0.18, wikisql 0.86→0.30 (전부 큰 폭↓) ✅
   - 정답률: hitab만 0.06→0.40. finqa·wikisql ~0 (계산이 천장) ❌
   - 결과: `results/binding_eval_stored/summary.json`

3. **셀 단위 검색 측정** — `scripts/cell_path_retrieval.py`
   (저장 cell_paths를 인덱스로 BM25/dense/hybrid, gold=bench.gold_operands, n=50)
   - R@1 천장: 멀티operand 때문에 finqa 0.33 / wikisql 0.46 / hitab 0.90 → **R@1 95% 불가**
   - full-set recall@k(dense): hitab @50 0.86 · finqa **@50 1.00** · wikisql @50 0.78
   - MRR: finqa 0.81 / hitab 0.65 / wikisql 0.40
   - bge-large ≈ bge-small (큰 모델이 이득 없음)
   - 결과: `results/cell_retrieval/cell_path_metrics.json`, `cell_path_bge_large.json`

## 2. 핵심 결론
- **메타데이터 = 검색 가능하게 + binding 실패 박멸.** 단 정답률 천장 아님.
- **R@1은 멀티operand라 잘못된 지표.** full-set recall@k(k=operand수) + MRR로 봐야.
- **recall 올릴 레버 = reranking + (wikisql) 행라벨 보강.** 메타 추가/임베더 키우기 ❌.
- 최종 정답률 천장 = LLM **계산식 설계**(operand 분해), 검색 아님.

## 3. 다음 할 일 (우선순위)
1. **reranking 붙이기** — cross-encoder로 top-k 재정렬 → full-set recall@k↑ 목표(≈95%, k=operand수).
   대상: hitab/finqa. wikisql는 2번 먼저.
2. **wikisql 행라벨 보강** — 셀 직렬화에 행 식별자 추가(현재 행헤더 없어 한 열 셀이 같은 텍스트=구분 불가).
3. (선택) **번호선택 binding** — LLM이 헤더 문자열 타이핑 대신 저장경로 ID 선택 → NO_MATCH 구조적 0.
4. 표본 키우기(n=50→200), seed 다중화로 신뢰구간 좁히기.

## 4. 환경/실행 메모
- venv: `/home/user/T2/hart-table-retrieval/.venv` (torch+cu126, groq, rank_bm25 OK)
- 실행 시 `HF_HOME=/home/user/.cache/huggingface` (Qwen2.5-7B-Instruct, bge-small/large 캐시됨)
- LLM 백엔드: **로컬 Qwen(레이트리밋 없음, GPU ~수십분)** 권장. Groq 70b는 무료티어 TPM 한도로 큰표에서 죽음.
- 데이터셋 레지스트리: `rag_agent/bench/registry.py` (hitab/finqa/wikisql)

### 자주 쓰는 커맨드
```bash
VENV=/home/user/T2/hart-table-retrieval/.venv
HF=/home/user/.cache/huggingface

# 메타 재생성
HF_HOME=$HF $VENV/bin/python scripts/build_table_meta.py --benches hitab,finqa,wikisql

# binding 검증 (저장 메타 사용, 로컬 Qwen)
HF_HOME=$HF $VENV/bin/python scripts/binding_eval.py --benches hitab,finqa,wikisql \
  --n 50 --modes naive,grounded --backend local --meta-store data/table_meta \
  --out-dir results/binding_eval_stored

# 셀 단위 검색 지표 (full-set recall@k + MRR)
HF_HOME=$HF $VENV/bin/python scripts/cell_path_retrieval.py --benches hitab,finqa,wikisql \
  --n 50 --device cuda
```

## 5. 재개 프롬프트 (복사해서 Claude Code에 붙여넣기)
```
docs/RESUME_PROMPT.md 읽고 현재 상태 파악해줘. 교수님 메타데이터 binding 가설은 검증 끝났고
(메모리 binding-metadata-verification, cell-retrieval-metrics 참고), 다음 과제는 reranking으로
full-set recall@k를 95%까지 올리는 거야. reranking부터 붙여서 hitab/finqa로 검증해줘.
```
