# 커밋하지 않은 큰 결과 파일 (2026-09-27)

GitHub 파일 한도(100MB)를 넘어 저장소에 넣지 않았다. 이 폴더의 나머지 파일(100MB 이하)은 모두 커밋했다.
파일은 로컬 디스크와 저장소 밖 백업에 있다. 같은 형식의 기록: `results/mh_arms/README-large-files.md`.

| 파일 | 크기 | sha256 |
|---|---|---|
| `rag-agent/results/rerun_20260926/mh/mh_train_tablerag_leaf_records.jsonl` | 131,639,283 바이트 (125.54 MiB) | `48ce2aa6c8c42fb295721bfce7bb11a871a4c0823ff7091a5c1faad0b4ef0551` |

- 요약 파일 `mh_train_tablerag_leaf.json` 은 커밋돼 있다. 그 `records_sha256` 이 위 sha256 과 같다.
- 저장소 밖 백업: `/home/user/T2-backup/large-files-2026-09-27/rag-agent/results/rerun_20260926/mh/mh_train_tablerag_leaf_records.jsonl`
  (복사 뒤 sha256 일치 확인).
- `rag-agent/.gitignore` 에 이 경로를 넣었다.
- 원 실행: 요약 파일 `provenance.git_commit` = `378db41`(다시 쓰기 뒤 해시 `726dc13`), `git_dirty` = false.
  다시 만들어 바이트가 같은지는 확인하지 않았다.
- 재생성 명령(`run.sh` 의 `mh_train_tablerag_leaf` 줄과 같은 인자, 기존 출력 보호 때문에 `--out-dir` 만 새 경로):

```
cd rag-agent
PYTHONPATH=. HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 .venv/bin/python scripts/mh_arms.py --split train \
  --header-rule v3.3u --alpha 0.7 --budget 20 --embed-model BAAI/bge-base-en-v1.5 --cache-dir .cache/rerun_20260926_mh \
  --unit tablerag --template s3c --tablerag-colmode leaf \
  --out-dir results/rerun_20260926_regen/mh --tag mh_train_tablerag_leaf
```
