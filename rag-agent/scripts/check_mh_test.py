"""MultiHiertt 공개 test 파일에 정답·근거가 있는지 확인한다 (2026-09-28). 논문 4.1절 평가 집합 근거.

실행:  cd rag-agent && .venv/bin/python scripts/check_mh_test.py  -> results/mh_test_check_20260928/summary.json
test.json 은 저장소 밖 임시 폴더에 받고 끝나면 지운다. 저장소 작업 트리 밖이므로 git(.gitignore 포함)의 대상이 아니다.
커밋 해시는 Hugging Face API 로 얻고, 파일은 그 커밋을 가리키는 URL 에서 받는다.
"""
import hashlib
import json
import subprocess
import tempfile
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO, FILE = "yilunzhao/MultiHiertt", "multihiertt_data/test.json"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/mh_test_check_20260928/summary.json"

run_at = datetime.now().astimezone().isoformat(timespec="seconds")
commit = json.load(urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}"))["sha"]
url = f"https://huggingface.co/datasets/{REPO}/resolve/{commit}/{FILE}"
top = Path(subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"],
                          capture_output=True, text=True, check=True).stdout.strip())
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "test.json"
    urllib.request.urlretrieve(url, path)
    inside_repo = path.resolve().is_relative_to(top)
    raw = path.read_bytes()
assert not inside_repo

data = json.loads(raw)
keys = Counter(k for d in data for k in d["qa"])
out = {
    "run_at": run_at, "url": url, "hf_commit": commit,
    "download_dir_inside_repo": inside_repo,
    "sha256": hashlib.sha256(raw).hexdigest(),
    "n_questions": len(data),
    "qa_keys": sorted(keys), "qa_key_counts": dict(keys),
    "n_with_key": {k: keys[k] for k in ("answer", "program", "text_evidence", "table_evidence")},
}
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
