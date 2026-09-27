"""MultiHiertt 공개 test 파일에 정답·근거가 있는지 확인한다 (2026-09-28). 논문 4장·6.5절 평가 분할 근거.

실행:  cd rag-agent && .venv/bin/python results/mh_test_check_20260928/check.py  -> result.json
파일은 Hugging Face 캐시(~/.cache/huggingface)에 받는다. 저장소에는 넣지 않는다.
리비전은 공식 정답을 읽는 scripts/mh_arms.py 의 OFFICIAL_REV 와 같다.
"""
import hashlib
import json
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO, REV, FILE = "yilunzhao/MultiHiertt", "f18473da528dede3d9ce2274366d9ee8102ea0fd", "multihiertt_data/test.json"

path = hf_hub_download(REPO, FILE, repo_type="dataset", revision=REV)
raw = Path(path).read_bytes()
data = json.loads(raw)
qa_keys = Counter(k for d in data for k in d["qa"])
out = {
    "repo": REPO, "revision": REV, "file": FILE,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "n_questions": len(data),
    "qa_keys": dict(qa_keys),
    "n_with_key": {k: qa_keys[k] for k in ("question", "answer", "program", "text_evidence", "table_evidence")},
}
Path(__file__).with_name("result.json").write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
