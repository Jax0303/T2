# 비교군 `raptor` — RAPTOR (Sarthi et al., ICLR 2024) 원본 코드, MultiHiertt 같은 조건

- 원본 코드: <https://github.com/parthsarthi03/raptor> → `vendor/raptor`, 커밋 `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767`. **vendor 코드 패치 0건**(모델은 확장점으로만 갈아 끼움).
- 구현: `scripts/baselines/raptor.py`. 실행: `run.sh` (GPU 잠금 `flock .gpu.lock` 안에서 돈다).
- 새로 설치한 패키지: `umap-learn 0.5.12`, `tiktoken 0.14.0`(원본 requirements; venv 에 없었다).

## 같은 조건

| 항목 | 값 |
|---|---|
| 검색 범위 | 질문이 속한 MultiHiertt train 문서의 **표 전부**(글 문단 제외). 표마다 청킹 arm 과 같은 markdown 줄(`chunks.markdown_source`, 표 전체 입력 조건과 같은 렌더링, 빈 `Table name:` 줄 제거), 표 사이 빈 줄. 머리글 규칙 v3.3u(`mh_arms.build_tables`) — 데이터 셀 판정에만 쓰인다 |
| 트리 | 문서마다 1개, 렌더링 글의 sha256 앞 16자리로 캐시(`trees/<key>.pkl`). 2,885 문항 → **문서 1,096개**(같은 문서를 여러 질문이 공유) |
| 검색 평가 | `results/mh_only_20260928/retrieval/mh_train_s3c_records.jsonl` 의 2,885 문항. gold 좌표 `"표-행-열"` = 펼친 격자 좌표(`mh_arms.cell_id`) — 2,885 문항 gold 셀 전부가 렌더링한 데이터 셀에 있음을 dry-run 으로 확인(제외 0) |
| 답변 평가 | `results/mh_arms/cap300_20260924/cell_uniq.jsonl` 의 882 문항 × 두 설정 |
| 채점 | EM = `rag_agent/eval/multihiertt_em.mh_exact_match`(공식). F1 = `rag_agent/eval/multihiertt_f1.mh_f1`(공식 `psunlpgroup/MultiHiertt@45bd9cc` 포팅; span 문항 DROP F1, 프로그램 문항은 예측이 숫자면 EM 과 같고 아니면 DROP F1). 이 파일에 먼저 두었던 같은 포팅과 연기 시험 출력에서 값이 같은지 대조함(아래). 정답은 공식 파일(`official_answers`) |
| 리더 | Qwen3-8B 4bit 비생각 greedy, seed 42, `--batch-size 1` 이면 `LocalQwenLLM.complete` |

## 두 검색 설정 (같은 트리, collapsed tree)

| 설정 | 문맥 상한 | 뜻 |
|---|---|---|
| (a) `2000` | 2,000 토큰 | 논문 collapsed tree 의 값 |
| (b) `575` | 575 토큰 | 본 방법 20셀 문맥의 평균 토큰(882건 `ranked_tokens[:20]` 합 평균 574.49) |

문서가 짧다(표 글 평균 **1,398 Qwen3 토큰**, 잎 청크 평균 12.9개). 그래서 (a)는 잎+요약 노드로 문서 대부분을 돌려주는 조건이다.
선택은 원본 `retrieve_information_collapse_tree` 그대로: 모든 노드를 질문과의 코사인 거리로 정렬해 앞에서부터 담고, 상한을 처음 넘기는 노드에서 멈춘다(건너뛰지 않음).

## 셀 배달(채점) 규칙

- 잎 청크: 셀 값 글 전체가 청크의 원문 조각 하나 안에 온전히 들어 있는 셀만 배달한다. 원본 `split_text` 를 한 줄씩 따라가며 조각의 원문 위치를 남기는 `split_text_spans` 로 판정하고, 문서마다 원본 `split_text` 출력과 글이 같은지 확인한다(다르면 멈춤).
- 요약 노드: 자식 노드가 배달하는 셀의 합집합.
- 검색 정확도 = gold 셀 전부가 문맥의 배달 셀에 든 문항 비율(본 방법과 같은 정의).

**원본 청커의 성질(이탈 아님, 보고 사항):** `split_text` 는 `. ! ? \n` 마다 문장을 끊고 100 토큰을 넘는 문장은 `, ; :` 로 다시 끊는다. 표의 소수 값(`12.5`)은 `.` 에서 갈라져 `12 5` 로 이어 붙고, 긴 행의 `1,234` 는 `,` 에서 갈린다. 그래서 잎 어딘가에 온전히 들어 있는 gold 셀 전부를 가진 문항은 2,885 중 **2,335건(80.94%)**, 나머지 550건은 gold 셀 중 하나 이상이 어느 잎에도 온전히 없어 — 이 값이 이 비교군 검색 정확도의 상한이다(dry-run `gold_all_deliverable_by_leaves`).

## 이탈 목록 (왼쪽 = 이번 실행, 오른쪽 = 논문/원본 코드)

| # | 항목 | 이번 실행 | 논문·원본 코드 | 이유 |
|---|---|---|---|---|
| 1 | 임베딩 | `BAAI/bge-base-en-v1.5`(`BgeEmbedding`, `BaseEmbeddingModel`), 노드 접두어 없음, **질문에는 bge 질의 접두어** | SBERT `multi-qa-mpnet-base-cos-v1`, 질문·노드 같은 인코딩 | 같은 조건 규칙(본 방법 인코더·질의 접두어) |
| 2 | 요약 LLM | Qwen3-8B 4bit 비생각 greedy(`QwenSummarizer`, `BaseSummarizationModel`), 프롬프트·`max_tokens=100` 원본 그대로 | gpt-3.5-turbo, 요약 100 토큰 | 같은 조건 규칙(폐쇄 환경, 무API) |
| 3 | QA LLM | Qwen3-8B 4bit 비생각 greedy(`QwenQA`, `BaseQAModel`), 시스템·사용자 프롬프트 글자 그대로 | gpt-3.5-turbo(`GPT3TurboQAModel`) | 같은 조건 규칙 |
| 4 | QA 프롬프트 뒤에 붙인 문장 | `" On the last line write 'Final answer: ' followed by only the final value, with no units or explanation."` — `answer_accuracy_mh.extract` 로 마지막 `Final answer:` 뒤만 채점, 표시가 없으면 출력 전체 | 없음(자유 문장 답) | 숫자 EM 채점에 답을 뽑아야 함 |
| 5 | QA 생성 상한 | 384 새 토큰 | 150 | 본 방법 리더와 같은 값; 표시 문장이 잘리지 않게 |
| 6 | collapsed tree `top_k` | 노드 전부(토큰 상한만 작동) | 코드 기본 `top_k=10`, `max_tokens=3500`(논문 본문은 2,000 토큰 상한만 서술) | 노드 10개(≈1,000 토큰)에서 먼저 잘리면 2,000 설정이 성립하지 않음 |
| 7 | 문맥 토큰 계산 | Qwen3-8B 토크나이저(리더 토크나이저) | tiktoken cl100k | 본 방법 575 가 Qwen3 토큰 기준이라 같은 단위로 셈. 잎 청크 100 토큰은 원본대로 cl100k |
| 8 | 잎 임베딩 스레드 | `use_multithreading=False` | 기본 True | 한 GPU 인코더에 동시 호출 방지(결과 동일) |
| 9 | 표 사이 구분·표 제목 줄 | 표 사이 빈 줄, 청킹 arm 의 빈 `# ` 제목 줄 없음 | 해당 없음(원본은 평문 문서) | 표 전체 입력 조건과 같은 렌더링 |

원본 코드 사용상 주의(이탈 아님): `ClusterTreeBuilder.construct_tree` 는 층을 멈출 때 `self.num_layers` 를 그 층 번호로 덮어쓴다. 빌더 하나를 여러 문서에 재사용하면 잎이 11개 이하인 문서를 한 번 만난 뒤로는 `num_layers=0` 이 되어 다음 문서부터 군집·요약이 사라진다(연기 시험 1차 `smoke3_attempt1_builder_reuse.log` 에서 13·20잎 문서가 요약 0회로 나와 확인). 그래서 문서마다 `ClusterTreeBuilder(config)` 를 새로 만든다(vendor 패치 없음).

**군집 실패 → 잎만:** 원본 군집(UMAP+GMM, `cluster_utils.get_optimal_clusters`)이 작은 문서에서 sklearn `GaussianMixture` "ill-defined empirical covariance" 로 죽는다(전체 실행 1차 `full.log`, 385번째 트리 뒤). 그 문서는 요약 없는 잎만의 트리(층 0)로 두고, `trees/stats.jsonl` 의 `cluster_failed_leaves_only`(오류 문구)와 `raptor_records.json` 의 `cluster_failed_leaves_only.{n, doc_keys}` 에 센다(vendor 패치 없음). 1차 실행의 부분 레코드는 `raptor_records.partial_crashed_385trees.jsonl` 로 옮기고 다시 만든다(트리 385개는 `trees/` 캐시 그대로 재사용).

바꾸지 않은 것: 잎 청크 100 토큰(cl100k)·문장 경계 청커, UMAP+GMM(BIC) 소프트 군집(`threshold=0.1`, `reduction_dimension=10`, 재군집 3,500 토큰), 요약 프롬프트, 트리 최대 5층, collapsed-tree 검색.
UMAP 은 원본이 `random_state` 를 두지 않아 트리 만들기는 실행마다 달라질 수 있다 — 트리는 캐시되므로 그 뒤 검색·답변은 캐시에서 재현된다.

## 결과 파일 (`--out-dir`)

- `raptor_records.jsonl` — 문항마다 `query_id, gold_ids, doc_key, ranked_units`(거리 순 상위 30 노드: `cells, text, n_tok(Qwen3), layer`), `context_2000`/`context_575`(리더에 준 문자열 그대로), `correct_*`, `n_units_*`, `n_tok_*`, `cells_*`. 요약 `raptor_records.json`.
- `raptor_answer_2000.jsonl`, `raptor_answer_575.jsonl` — `query_id, pred, answer, answer_correct(EM), f1, raw, marker_found, n_tok(프롬프트 토큰), n_llm_calls(=1), seconds`. 요약 `*.json`. 트리 만들기 LLM 호출은 답변에 넣지 않고 `trees/stats.jsonl` 에 문서별로 따로 적는다(`n_summary_calls, seconds, n_leaves, n_nodes, n_layers`).
- 있는 결과 파일은 덮어쓰지 않는다(`raptor_records.jsonl` 이 있으면 멈춤). 답변 파일은 있으면 남은 문항만 이어 쓴다.

## 연기 시험 (`--limit 3`, `smoke3/`, 2026-09-29)

882 답변 집합의 앞 3 문항 = 문서 3개. 1차 실행(`smoke3_attempt1_builder_reuse.log`)은 위 빌더 재사용 문제로 요약이 0회라 버리고, 문서마다 새 빌더로 다시 돌린 결과다.

트리 (`smoke3/trees/stats.jsonl`; 요약 1회 = Qwen3-8B 100토큰 생성):

| 문서 | 표 글(cl100k 토큰) | 잎 | 노드 | 층 | 층별 노드 | 요약 호출 | 트리 초 | 셀 / 잎이 온전히 배달 가능한 셀 |
|---|---|---|---|---|---|---|---|---|
| bc002b36 | 812 | 10 | 10 | 0 | {0: 10} | 0 | 0.4 | 100 / 100 |
| d0d73e9c | 1,163 | 13 | 17 | 1 | {0: 13, 1: 4} | 4 | 37.2 (9.3/호출) | 156 / 90 |
| b6c86679 | 1,687 | 20 | 25 | 1 | {0: 20, 1: 5} | 5 | 34.1 (6.8/호출) | 253 / 229 |

잎이 11개 이하면 원본 규칙(`len(nodes) <= reduction_dimension + 1`)대로 층이 생기지 않는다.

검색 (`raptor_records.jsonl`; 셀 = 배달 셀 수, 정답 = gold 전부 포함):

| 문항 | 유형 | gold | (a) 2000: 노드/토큰/셀/정답 | (b) 575: 노드/토큰/셀/정답 |
|---|---|---|---|---|
| 007a44f1 | lookup_m2+ | 1-5-1, 1-5-3 | 10(전부)/1,039/100/1 | 5/560/45/1 |
| 00fa4ad4 | lookup_m2+ | 2-4-4, 2-7-4, 2-15-4 | 17(전부)/1,807/90/1 | 5/542/20/0 |
| 01208501 | arith_m2+ | 0-1-2, 1-20-5 | 17/1,876/229/1 | 5/529/97/1 |

(a)는 문서 두 개에서 트리 노드 전부(잎+요약)를 돌려줬다 — "문서 대부분" 조건임을 확인.

답변 (`raptor_answer_*.jsonl`, 표시 누락 0/6, `--batch-size 1`):

| 문항 | 정답 | (a) 2000 pred / EM / F1 / 초 | (b) 575 pred / EM / F1 / 초 |
|---|---|---|---|
| 007a44f1 | no | `No` / 1 / 1.0 / 2.3 | `No` / 1 / 1.0 / 3.8 |
| 00fa4ad4 | Fixed maturities, Commercial mortgage and other loans | `2578` / 0 / 0.0 / 4.4 | `Fixed maturities` / 0 / 0.44 / 21.2 |
| 01208501 | 429192.5 | `79300065385` / 0 / 0.0 / 5.5 | `79300065385` / 0 / 0.0 / 4.6 |

- 답변 1건당 평균 (a) 4.0초, (b) 9.9초(21초 1건 포함; 나머지 5건 평균 4.1초). 프롬프트 토큰 (a) 1,120·1,893·1,968, (b) 641·620·623.
- F1 대조: 이 파일에 먼저 두었던 포팅과 `rag_agent/eval/multihiertt_f1.mh_f1` 이 6행 모두 같은 값(불일치 0).

## 전체 실행 (아직 돌리지 않음)

    setsid nohup bash results/mh_only_20260928/baselines/raptor/run.sh > results/mh_only_20260928/baselines/raptor/full.log 2>&1 &

(검색 2,885 + 답변 882×2, 기본 `--out-dir`. 연기 시험 트리를 재사용하려면 `--tree-dir results/mh_only_20260928/baselines/raptor/smoke3/trees`.)

추정(연기 시험 실측 기준, `--batch-size 1`):

| 단계 | 개수 | 단가(실측) | 추정 |
|---|---|---|---|
| 트리: 잎 ≤11 문서 | 540 | 0.4초 | 4분 |
| 트리: 잎 >11 문서 | 556 (평균 16.6잎, 최대 45) | 요약 4~5회/문서(잎 13→4, 20→5) × 6.8~9.3초 | 4.9~7.6시간 (중앙 약 6시간) |
| 답변 | 1,764 (882×2) | 4.1초(21초 이상치 제외) ~ 7.0초(포함) | 2.0~3.4시간 |
| 합계 | | | **약 7~11시간** |

잎이 많은 문서는 1층 노드가 11개를 넘으면 2층까지 생겨 호출이 늘 수 있다(연기 시험 3건에는 없었음). GPU 를 다른 실행과 나눠 쓰면 더 걸린다.
