#!/usr/bin/env bash
# 진행 상황 한눈에: 완료·실패 목록, 결과 파일 줄 수(문항 단위 저장이라 줄 수 = 끝난 문항 수), 마지막 세션 로그 끝.
cd "$(dirname "$0")/../.." || exit 1
S=results/post_defense_20261006; B=results/mh_only_20260928/baselines; touch $S/done.txt $S/failed.txt
echo "done:   $(tr '\n' ' ' < $S/done.txt 2>/dev/null)"; echo "failed: $(tr '\n' ' ' < $S/failed.txt 2>/dev/null)"
rows() { [ -f "$1" ] && wc -l < "$1" || echo 0; }
printf '%-22s %s\n' dev_k1 "$(rows $S/dev_k/dev_s3c_k1_answer.jsonl)/332" dev_k3 "$(rows $S/dev_k/dev_s3c_k3_answer.jsonl)/332" \
  yu882 "$(rows $B/tablerag_yu/tablerag_yu_answer.jsonl)/882" \
  l2_ours "$(rows $S/llm_qwen17/ours_answer.jsonl)/200" l2_langchain "$(rows $S/llm_qwen17/langchain_chunk/langchain_chunk_answer.jsonl)/200" \
  l2_yu "$(rows $S/llm_qwen17/tablerag_yu/tablerag_yu_answer.jsonl)/200" l2_raptor "$(rows $S/llm_qwen17/raptor/raptor_records.jsonl)/200 기록, $(rows $S/llm_qwen17/raptor/raptor_answer_2000.jsonl)/200 답변" \
  l2_chen "$(rows $S/llm_qwen17/tablerag_chen/tablerag_chen_answer.jsonl)/200" \
  e2_ours_ret "$(rows $S/enc_bgem3/retrieval/mh_train_s3c_records.jsonl)/2885" e2_ours "$(rows $S/enc_bgem3/ours_answer.jsonl)/200" \
  e2_langchain "$(rows $S/enc_bgem3/langchain_chunk/langchain_chunk_records.jsonl)/2885 기록, $(rows $S/enc_bgem3/langchain_chunk/langchain_chunk_answer.jsonl)/200 답변" \
  e2_yu "$(rows $S/enc_bgem3/tablerag_yu/tablerag_yu_records.jsonl)/2885 기록, $(rows $S/enc_bgem3/tablerag_yu/tablerag_yu_answer.jsonl)/200 답변" \
  e2_raptor "$(rows $S/enc_bgem3/raptor/raptor_records.jsonl)/200 기록, $(rows $S/enc_bgem3/raptor/raptor_answer_2000.jsonl)/200 답변" \
  e2_chen "$(rows $S/enc_bgem3/tablerag_chen/tablerag_chen_answer.jsonl)/200"
last=$(ls -t $S/session_*.log 2>/dev/null | head -1); [ -n "$last" ] && { echo "--- $last"; tail -3 "$last"; }
pgrep -f "post_defense_20261006/session.sh" > /dev/null && echo "세션 실행 중" || echo "세션 없음"
