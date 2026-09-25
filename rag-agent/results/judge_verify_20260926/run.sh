#!/usr/bin/env bash
# 2026-09-26 판정자 A 자동 검증: (a) 사전등록 규칙(근거 구절 단위) (b) 사후 추가 분석(단어 단위, 불용어 제외). PREREG §5·§9.
# 실행: cd rag-agent && bash results/judge_verify_20260926/run.sh
set -euo pipefail
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
D=results/judge_verify_20260926
MH=results/doc_identify_20260926/sample100_rater_A.csv
HT=results/hitab_e_miss_20260926/judge40_rater_A.csv
.venv/bin/python scripts/judge_verify.py mh "$MH" $D/mh_A_phrase
.venv/bin/python scripts/judge_verify.py mh "$MH" $D/mh_A_words --words
.venv/bin/python scripts/judge_verify.py hitab "$HT" $D/hitab_A_phrase
.venv/bin/python scripts/judge_verify.py hitab "$HT" $D/hitab_A_words --words
