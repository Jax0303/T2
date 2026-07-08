# 다음 세션 시작 프롬프트 (붙여넣기용)

아래 블록을 그대로 복사해서 새 세션 첫 메시지로 붙여넣으면 됩니다.

---

2026-07-07~08 세션에서 큰 작업 세 갈래(전체 코드 버그 감사, 지표 정합성, 논문 프레이밍 재조정)를
끝냈다. 이번엔 그 결과로 갱신이 밀린 문서 동기화 + H6 재실험 표본 확대부터 이어간다.

## 환경 (매 세션 확인)
- 작업 디렉토리: `/home/user/T2-1/rag-agent`. 작동 인터프리터 =
  `/home/user/T2/hart-table-retrieval/.venv/bin/python` (시스템 python3엔 numpy 없음).
- 실행: `PYTHONPATH=. /home/user/T2/hart-table-retrieval/.venv/bin/python scripts/...`
- GROQ_API_KEY는 env에 있음. 모델별 일일쿼터 분리(leaky-bucket, 시간당 조금씩 회복): 8B=500k TPD,
  70B=100k TPD, gpt-oss-120b=200k TPD.
- 모집단: HiTab dev, arithmetic m≥2, n=161 (gold-offset 버그 수정 후에도 n은 동일, gold cell만
  일부 바뀜).

## 지난 세션에서 끝난 것 (커밋됨, 순서대로)
1. **전체 코드 버그 감사 6개 병렬 에이전트** → 8개 버그 수정·검증(`e4390c1`, `394ba9b`):
   - m=1 쿼리 gold operand offset 오염(테이블 단위로 풀링해서 수정, dev 39개 쿼리 gold 변경)
   - 행축 내부노드 헤더 소실(walk() leaf만 line_idx 기록하던 버그, 509→12테이블)
   - col/row_select_bench.py lexical 후보군 불공정(cross p=.007 결과는 안전 확인됨)
   - e4/e7 bucket() 동어반복(numeric_match(x,x)는 항상 True)
   - answerer.py codegen 예외처리 협소(IndexError 등 못잡아 쿼터소진으로 오분류)
   - answer_accuracy_injection.py flip-subset 라벨링 모호성
   - format_context 조용한 truncation(이제 감지·카운트됨)
   - 상세: `[[codebase-audit-bugfixes-2026-07-07]]` 메모리
2. **지표 정합성(교수님 피드백)**: HiTab 공식 채점기(`hmt_score`) 그대로 포팅 →
   `rag_agent.eval.metrics.hitab_exact_match`(`e6a3a54`). **다른 논문과 비교할 땐 반드시 이걸
   써야 함**, `numeric_match`는 우리 자체 관대한 진단용(±2%, %/비율 자동변환 있음).
3. **코드젠 프롬프트에 비율=소수 규칙 추가**(`adce626`) — HiTab은 "몇 %냐"고 물어도 gold가 항상
   소수 형태(0.34, ×100 안 함)라서 생긴 스케일 혼동 제거.
4. **H6 재실험**(`results/h6_rerun_20260707/`, 프롬프트 수정 후 공식지표로 재측정):
   - 8B: n=134/161, 공식EM 0.112→0.119(Δ+.008, p=1.0, 무의미) — 여전히 병목.
   - 70B: n=23/161(쿼터컷), 공식EM 0.304→0.391(Δ+.087, 2:0 flip, p=.5, n부족).
   - **gpt-oss-120b: n=37/161(쿼터컷), 공식EM 0.405→0.649(Δ+.243, 10:1 flip, p=.012 유의)** —
     H6 첫 유의 신호. 쿼터 회복 후 `--resume`으로 표본 늘리면 더 단단해질 가능성.
5. **논문 프레이밍 재조정**(`38169f5`): 총합-행 주입을 헤드라인(기여4)→사례연구(기여5)로 격하,
   천장진단(유사도가 왜 76% 실패하는지)을 독립 기여(2번)로 승격. §5.10을
   서브그룹표(필요 37%만 유의 상승, 나머지 63%는 정확히 Δ=0)로 재작성.
6. **구조적(산술) 합계행 탐지 추가**(`1d9b024`): `is_total_row_structural`/`total_like_rows_hybrid`
   (학습 없이, 형제/자식 행 합과 값이 맞는지 산술 검증 — 언어독립). 기존 키워드판의 조상경로
   과매칭 버그도 발견·수정. hybrid(keyword∪structural)가 이제 기본값, §5.10 최신 수치:
   bm25 0.770→0.814(p=.016), dense 0.832→0.863(**p=.063 유의성 상실**), hybrid 0.839→0.888(p=.008).
   **⚠ 이 수치는 MiniLM resolver로 측정된 것(브랜치 분기 사고) — 정본은 위 1번 완료 노트의
   bge-reranker 재측정 수치(.876/.863/.907).**

## 이번에 할 것 (우선순위)
1. ~~**`docs/RESULTS.md`, `docs/LAB_MEETING_BRIEF.md`를 PAPER_DRAFT.md와 동기화**~~ ✅ **완료
   (2026-07-08)**: 두 문서 모두 상단에 post-audit 최신 스냅샷 섹션 추가(§5.10 하이브리드 탐지
   수치, H6 공식EM 3솔버, E9, §5.11 스코프, 지표 정책), 옛 섹션엔 "감사 이전 측정치" 경고 배너.
   E2 열거-단독과 07-03 test split 수치는 재측정 전이라 "재검증 필요"로만 표시(날조 안 함).
   **추가로 같은 날 오후: §5.10/E9에 per-cell recall 병기(교수님 레퍼런스-지표 요구) + 재실행.**
   이 과정에서 발견: 아래 6번의 0.814/0.863/0.888은 브랜치 분기 사고로 **MiniLM resolver**로 측정된
   값이었음(이 세션 브랜치가 06-29에서 분기해 07-03의 bge-reranker 기본값 변경이 없었음). bge-reranker
   정본 재측정: **BM25 .770→.876(p=1.5e-5), dense .832→.863(p=.0625 n.s.), hybrid .839→.907(p=.001)**,
   주입 4.4셀, needs-total 서브그룹 gap 폐쇄 65%/31%/58%. 단 strict cell-matched는 **BM25만 유의**
   (dense/hybrid +0.019 n.s.) — 06-29 "strict 셋 다 승리"는 pre-audit이므로 폐기, test split 재실행 필요.
   E9도 post-audit 재측정(250–8k 전 구간 p≤.004, @8k 1.000 vs 0.870) + recall_at_budget 추가.
   PAPER_DRAFT §4/§5.9b/§5.10/기여5, RESULTS.md, LAB_MEETING_BRIEF, e9 summary 전부 반영됨.
2. **H6 표본 확대** — 🔶 진행 중 (07-08 오후 1회차 resume 완료): gpt-oss-120b **37→86/161**,
   공식EM 0.395→0.500(Δ+.105, 11:2, **p=.022 유의 유지**), OSC-flip 11개에선 0.00→0.73;
   70B 23→52/161(0.346→0.404, 3:0, p=.25 여전히 n부족). Δ가 n=37 때보다 준 건 flips-first
   희석(예상된 것). **잔여: 120b 75개(~2일), 70B 109개(~4일)** — 쿼터 회복(일일)마다 같은 명령
   + `--resume` 반복. treat truncation 4건 감지됨(주입이 솔버에 안 닿은 쿼리 — 해석 시 주의).
   ```
   PYTHONPATH=. /home/user/T2/hart-table-retrieval/.venv/bin/python scripts/answer_accuracy_injection.py \
     --solver-model openai/gpt-oss-120b --codegen-max-tokens 1024 --flips-first --resume \
     --out results/h6_rerun_20260707/gptoss120b.json \
     --records results/h6_rerun_20260707/gptoss120b_records.jsonl
   ```
3. **E2 enumeration-alone raw OSC 재검증**: 버그 수정 후 재측정 안 함(PAPER_DRAFT.md에 "재검증 필요"로
   비워둠). `scripts/e2_osc_enum.py --split dev`로 재실행해서 "0.79 > 0.65" 류 비교 문장 복원.
4. (낮은 우선순위) 남은 감사 항목: 테스트 커버리지 공백(decomposition ceiling/cross-encoder
   row-col_mode dispatch 미검증), `--resume` 설정 일치 검증 없음, `download_benchmarks.py` 죽은 import.

## 정직성 가드 (잊지 말 것)
- **총합-행 주입은 이제 "사례연구"이지 헤드라인이 아님** — PAPER_DRAFT.md 기여 목록 순서(진단 2번 >
  주입 5번)를 유지할 것, 다시 승격시키지 말 것.
- 다른 논문과 숫자 비교할 땐 무조건 `hitab_exact_match`(공식) — `numeric_match`/`evaluate_answer`로
  나온 수치를 "정확도"라고 논문에 인용 금지.
- H6은 여전히 쿼터 제약으로 부분표본(n=23~37)임을 항상 병기. gpt-oss-120b의 p=.012도 "첫 유의 신호"
  이지 확정된 결론 아님 — 표본 늘리기 전엔 과장 금지.
- structural 탐지는 "일반성 부분 해소"이지 완전 해소 아님 — 비영어 계층표 벤치마크로 검증 전까지는
  그렇게 표현.

각 단계 결과 보여주고 유의성까지 보고해줘.
