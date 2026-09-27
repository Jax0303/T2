# 커밋하지 않은 큰 결과 파일 (2026-09-28)

GitHub 파일 한도(100MB)를 넘어 저장소에 넣지 않았다. 이 폴더의 나머지 파일은 모두 커밋했다. 같은 형식: `results/rerun_20260926/README-large-files.md`.

| 파일 | 크기 | sha256 |
|---|---|---|
| `rag-agent/results/tablerag_official_20260928/mh/mh_train_tablerag_leaf_official_records.jsonl` | 129,153,176 바이트 | `213cdd65d0802ee19c56173f6e028f742f0d173d083168cd9f8c2724b2f5b20a` |

- 요약 파일 `mh_train_tablerag_leaf_official.json` 은 커밋돼 있고, 그 `records_sha256` 이 위 sha256 과 같다.
- 저장소 밖 백업: `/home/user/T2-backup/large-files-2026-09-28/rag-agent/results/tablerag_official_20260928/mh/mh_train_tablerag_leaf_official_records.jsonl`
  (복사 뒤 sha256 일치 확인).
- `rag-agent/.gitignore` 에 이 경로를 넣었다.
- 재생성: `results/tablerag_official_20260928/run.sh` 의 `mh_train_tablerag_leaf_official` 줄(기존 출력 보호 때문에 `--out-dir` 은 새 경로로).
