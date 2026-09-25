"""gate0_2_scan.json 중 표 전체 계열 arm 의 answer EM 이 있는 파일만 markdown 표로. 추가 계산 없음."""
import json, subprocess
from pathlib import Path
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
W = ("dump", "cell2dump", "table_md", "goldtable", "base")
OURS = ("cell", "cell_sent", "treat")
def git(*a): return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout
print("| 읽은 커밋 | 경로(rag-agent/results/) | 데이터셋·split | 모집단 | query count | 인코더 | 리더 | 예산(토큰) | 검색기 | 표 전체 arm EM | 셀 arm EM |")
print("|---|---|---|---|---:|---|---|---:|---|---|---|")
for x in json.load(open(HERE / "gate0_2_scan.json")):
    em = x["answer_em"]
    if not any(k in W for k in em):
        continue
    corpus = json.loads(git("show", f"{x['ref']}:{x['path']}")).get("corpus") or {}
    ds = f"{x['dataset']} {x['split'] or ''}".strip() + (f" ({corpus.get('tables')}표 한 색인)" if corpus.get("tables") else "")
    print(f"| `{x['ref']}` | {x['path'].split('results/')[-1]} | {ds} | {x['population']} | {x['query_count']} | "
          f"{(x['encoder'] or '-').split('/')[-1]} | {x['reader']} | {x['budget_tokens']} | {x['retriever'] or '-'} | "
          + ", ".join(f"{k} {em[k]}" for k in W if k in em) + " | "
          + ", ".join(f"{k} {em[k]}" for k in OURS if k in em) + " |")
