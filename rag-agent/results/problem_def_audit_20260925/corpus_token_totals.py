"""표 전체 직렬화(chunks.markdown_source)의 코퍼스 총 토큰과 32,768 한도 대비. 토크나이저만, 추론 없음.

- 토크나이저: Qwen/Qwen2.5-7B-Instruct a09a3545…, add_special_tokens=False, 채팅 템플릿·시스템 프롬프트 없음.
- HiTab: test 538표, 표마다 markdown_source(tab, tab.table, 섹션 제목+페이지 제목) — scripts/hitab_fulltable_answer.py:47-50 와 같음.
- MultiHiertt: cap300 과 같은 split=train, header_rule=v1, label_rule=none, 문서 = 그 문서의 표 전부를 "\n" 으로 이음
  (scripts/answer_accuracy_mh.py:90-101, 280-283 의 fulltable 문맥과 같음). 본문 문단은 fulltable 문맥에 없으므로 제외.
- 여러 단위(표·문서)를 이을 때 구분자 "\n\n". 채우기 결과는 실제로 이어 붙인 문자열을 다시 토큰화해 확인."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from transformers import AutoTokenizer                                       # noqa: E402
from rag_agent.bench import hitab_grid as hg                                 # noqa: E402
from rag_agent.serialization.caption import with_page_title                  # noqa: E402
from rag_agent.serialization.chunks import markdown_source                   # noqa: E402
from scripts.answer_accuracy import _PAGE_TITLES                             # noqa: E402
from answer_accuracy_mh import full_tables                                   # noqa: E402

LIMIT, SEP = 32768, "\n\n"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", revision="a09a35458c702b33eeacc393d103063234e8bc28")
n_tok = lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

def summary(texts, what):
    counts = [n_tok(t) for t in texts]
    total_sum = sum(counts)
    total_concat = n_tok(SEP.join(texts))
    order = sorted(range(len(texts)), key=lambda k: counts[k])
    k, acc = 0, 0
    for i in order:                                     # 작은 것부터, 개별 토큰 수 합으로 추정
        if acc + counts[i] > LIMIT:
            break
        acc += counts[i]; k += 1
    while k and n_tok(SEP.join(texts[i] for i in order[:k])) > LIMIT:   # 실제 이어 붙인 길이로 확인
        k -= 1
    s = sorted(counts)
    return {"unit": what, "n_units": len(texts),
            "tokens_sum_of_units": total_sum, "tokens_concatenated_with_sep": total_concat,
            "times_limit_concat": round(total_concat / LIMIT, 2),
            "per_unit_min_median_max": [s[0], s[len(s) // 2] if len(s) % 2 else (s[len(s) // 2 - 1] + s[len(s) // 2]) / 2, s[-1]],
            "units_over_limit": sum(c > LIMIT for c in counts),
            "max_units_fitting_smallest_first": k,
            "ratio_of_all": round(k / len(texts), 4),
            "tokens_of_that_fill": n_tok(SEP.join(texts[i] for i in order[:k])) if k else 0}

ids = sorted({json.loads(l)["table_id"] for l in open(ROOT / "data/hitab/data/test_samples.jsonl")})
hitab = []
for tid in ids:
    tab = hg.load_table(tid)
    hitab.append(markdown_source(tab, tab.table, with_page_title(tab.title, _PAGE_TITLES.get(tid)))[0])

mh = full_tables("train", "v1", "none")             # 문서 id -> [표 markdown]
docs = {d: "\n".join(v) for d, v in mh.items()}
cap_ids = [json.loads(l)["query_id"] for l in open(ROOT / "results/mh_arms/cap300_20260924/fulltable.jsonl")]
print(json.dumps({
    "tokenizer": "Qwen/Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28 (add_special_tokens=False, 채팅 템플릿 없음)",
    "limit": LIMIT, "separator_between_units": "\\n\\n",
    "hitab_test_tables": summary(hitab, "HiTab test 표 (markdown_source)"),
    "multihiertt_train_documents": summary(list(docs.values()), "MultiHiertt train 문서 (표 전부 \\n 연결)"),
    "multihiertt_train_tables": summary([t for v in mh.values() for t in v], "MultiHiertt train 표 (markdown_source)"),
    "multihiertt_cap300_documents": summary([docs[q] for q in cap_ids if q in docs], "cap300 882건의 문서"),
    "cap300_ids_found": sum(q in docs for q in cap_ids), "cap300_ids_total": len(cap_ids),
}, indent=1, ensure_ascii=False))
