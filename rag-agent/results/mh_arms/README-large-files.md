# 커밋하지 않은 큰 결과 파일 (2026-09-27)

GitHub 파일 한도(100MB)를 넘어 저장소 기록에서 뺐다. 파일은 로컬 디스크와 저장소 밖 백업에 있다.

| 파일 | 크기 | sha256 |
|---|---|---|
| `rag-agent/results/mh_arms/mh_train_tablerag_leaf_hv3.3_none_doc_records.jsonl` | 119,604,544 바이트 (114.06 MB) | `321b50bbc74141683f3845d4d07584192dee77037579a9ab1a96e1546e76119d` |

- 요약 파일 `mh_train_tablerag_leaf_hv3.3_none_doc.json` 은 커밋돼 있다. 그 `records_sha256` 이 위 sha256 과 같다.
- 저장소 밖 백업: `/home/user/T2-backup/large-files-2026-09-27/rag-agent/results/mh_arms/mh_train_tablerag_leaf_hv3.3_none_doc_records.jsonl`
  (복사 뒤 sha256 일치 확인).
- `rag-agent/.gitignore` 에 이 경로를 넣어 다시 추가되지 않게 했다.
- 원 실행: 요약 파일 `provenance.git_commit` = `d0f96f5`, `git_dirty` = true. 미커밋 변경이 있던 상태라 다시 만들어도
  바이트가 같다는 것은 확인하지 않았다.
- 재생성 명령(요약 파일 `arguments` 그대로, 기존 출력 보호 때문에 `--out-dir` 만 새 경로):

```
cd rag-agent
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH=. .venv/bin/python scripts/mh_arms.py --split train --unit tablerag \
  --template s3c --row-text sentence --tablerag-colmode leaf --tablerag-dtype infer --header-rule v3.3 --label-rule none \
  --chunk-chars 1000 --chunk-overlap 200 --embed-model BAAI/bge-base-en-v1.5 --embed-overflow error --alpha 0.7 --budget 20 \
  --rowcol-max-pairs 200 --shard 50000 --dump-context 1 --cache-dir .cache/mh_arms \
  --out-dir results/mh_arms_regen --tag mh_train_tablerag_leaf_hv3.3_none_doc
```

## 커밋 다시 쓰기 — 옛 해시 → 새 해시

이 파일을 넣었던 커밋(`bd5add5`)과 그 뒤 미푸시 커밋을 `git filter-branch --index-filter`(이 경로만 `git rm --cached`)로 다시 썼다.
작성자·작성 시각·커밋 시각·메시지는 그대로다. 트리 차이는 이 파일 하나뿐이다. 다시 쓰기 전 상태는 로컬 브랜치
`backup/pre-rewrite-2026-09-27` 에 남겼다(푸시하지 않음).

| 옛 해시 | 새 해시 | 작성 시각 | 비고 |
|---|---|---|---|
| `410c3dc` | `410c3dc` (410c3dc0a7270f4356566b96146e62a9893e484e) | 2026-09-26T23:06:58+09:00 | 이 파일 이전 커밋, 바뀌지 않음 |
| `bd5add5` | `6998304` (6998304dd23767124c610da90195186d433a1075) | 2026-09-26T23:32:28+09:00 | 이 파일 제거 |
| `378db41` | `726dc13` (726dc131fa4641def2a2458df209102b50d81690) | 2026-09-26T23:32:28+09:00 | |
| `5f7ac77` | `ff75bd4` (ff75bd48a4a2f2f1e4fcfaedf337b5cb5c05ba10) | 2026-09-27T03:20:51+09:00 | |
| `4a46e20` | `4db5ec1` (4db5ec1db94e8b05c6ebb4d11b4b287f1fee372b) | 2026-09-27T03:44:28+09:00 | |
| `7be1579` | `3b8b31a` (3b8b31afd0737edeee2ebfc0621b0dfec736aa20) | 2026-09-27T04:12:41+09:00 | |

결과 JSON·로그의 `provenance.git_commit`, `commit …` 줄에 남은 옛 해시(`378db41` 등)는 결과 파일이라 고치지 않았다.
옛 해시는 위 표로 새 해시에 대응한다.
