"""HiTab 300건 본 방법(t_sleaf_gold) 리더 입력 토큰을 구성요소별로 나눈다. 토크나이저만, 추론 없음.

문장(sleaf) = "{행 잎} / {열 잎}: In the table '{제목}', among {행 경로}, the value of {열 경로} is {값}."
(`rag_agent/serialization/templates.py:71-120`). 각 문장을 구성요소 조각으로 다시 만들어 기록된 문장과
바이트 일치를 확인하고, 리더 프롬프트(`scripts/fair_filter_eval.py:278-281`)를 Qwen2.5-7B-Instruct
토크나이저로 오프셋과 함께 토큰화해 토큰마다 글자 다수결로 범주를 붙인다. 총 토큰 수가 기록된
reader_input_tokens 와 같은지 질의마다 확인한다."""
import collections, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from transformers import AutoTokenizer                                       # noqa: E402
from rag_agent.bench import hitab_grid as hg                                 # noqa: E402
from rag_agent.serialization.base import fmt_value, join_path                # noqa: E402
from rag_agent.serialization.caption import with_page_title                  # noqa: E402
from scripts.answer_accuracy import PROMPTS, _PAGE_TITLES                    # noqa: E402
from scripts.fair_filter_eval import split_lines                             # noqa: E402

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct",
                                    revision="a09a35458c702b33eeacc393d103063234e8bc28")
rows = [r for r in map(json.loads, open(ROOT / "results/fair_filter_20260921/rows.jsonl")) if r["arm"] == "ours"]
recs = {r["query_id"]: r for r in map(json.loads, open(ROOT / "results/retrieval_accuracy/t_sleaf_gold_records.jsonl"))}
tabs = {}

def segments(tid, i, j):
    """sleaf 문장을 (글, 범주) 조각으로. templates.render 와 같은 규칙."""
    if tid not in tabs:
        tab = hg.load_table(tid)
        tabs[tid] = (tab.table, with_page_title(tab.title, _PAGE_TITLES.get(tid)))
    t, title = tabs[tid]
    rp, cp, v = t.row_path(i), t.col_path(j), t.data[i][j]
    val, title_s, row_s, col_s = fmt_value(v), fmt_value(title) if title else "", join_path(rp), join_path(cp)
    leaves = [x for x in (fmt_value(rp[-1]) if rp else "", fmt_value(cp[-1]) if cp else "") if x]
    seg = []
    for k, x in enumerate(leaves):
        seg += ([(" / ", "frame")] if k else []) + [(x, "leaf_prefix")]
    if leaves:
        seg.append((": ", "frame"))
    if not title_s:   # structural_compact 제목 없음 → 경로: 값
        path = join_path([*rp, *cp])
        return seg + ([(path, "header_path"), (": ", "frame")] if path else []) + [(val, "value")]
    seg += [("In the table '", "frame"), (title_s, "title"), ("', ", "frame")]
    if row_s:
        seg += [("among ", "frame"), (row_s, "header_path"), (", ", "frame")]
    seg += ([("the value of ", "frame"), (col_s, "header_path")] if col_s else [("the value", "frame")])
    return seg + [(" is ", "frame"), (val, "value"), (".", "frame")]

tot = collections.Counter()
per_q = []
for r in rows:
    rec = recs[r["query_id"]]
    lines = split_lines(rec["context"])
    assert len(lines) == len(rec["context_units"])
    segs = []
    for line, u in zip(lines, rec["context_units"]):
        (tid, i, j), = u["cells"]
        s = segments(str(tid), i, j)
        assert "".join(x for x, _ in s) == line, (r["query_id"], line)
        segs.append(s)
    user = "Context:\n" + "\n".join(lines) + f"\n\nQuestion: {rec['question']}\nAnswer:"
    assert user.count("Context:\n") == 1
    prompt = tok.apply_chat_template([{"role": "system", "content": PROMPTS["neutral"]},
                                      {"role": "user", "content": user}],
                                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
    cat = ["rest"] * len(prompt)
    pos = prompt.index(user) + len("Context:\n")
    for k, s in enumerate(segs):
        for x, c in s:
            cat[pos:pos + len(x)] = [c] * len(x)
            pos += len(x)
        pos += 1  # 줄바꿈 → rest
    q0 = prompt.index(f"Question: {rec['question']}", pos - 1)
    cat[q0:q0 + len("Question: ") + len(rec["question"])] = ["question"] * (len("Question: ") + len(rec["question"]))
    enc = tok(prompt, return_offsets_mapping=True, add_special_tokens=True)
    assert len(enc["input_ids"]) == r["reader_input_tokens"], (r["query_id"], len(enc["input_ids"]), r["reader_input_tokens"])
    c = collections.Counter()
    for a, b in enc["offset_mapping"]:
        span = cat[a:b] or ["rest"]
        top = collections.Counter(span).most_common()
        c[top[0][0] if len(top) == 1 or top[0][1] > top[1][1] else span[0]] += 1
    assert sum(c.values()) == r["reader_input_tokens"]
    per_q.append(c)
    tot += c

n = len(per_q)
order = ["title", "header_path", "leaf_prefix", "value", "frame", "question", "rest"]
mean = {k: round(tot[k] / n, 1) for k in order}
print(json.dumps({
    "query_count": n,
    "source": {"rows": "results/fair_filter_20260921/rows.jsonl (arm=ours, reader_input_tokens)",
               "records": "results/retrieval_accuracy/t_sleaf_gold_records.jsonl",
               "tokenizer": "Qwen/Qwen2.5-7B-Instruct a09a35458c702b33eeacc393d103063234e8bc28, chat template 포함"},
    "checks": {"sentences_byte_identical": True, "token_total_equals_reader_input_tokens": True},
    "mean_tokens_per_query": mean,
    "mean_total": round(sum(tot.values()) / n, 1),
    "mean_lines_per_query": round(sum(len(split_lines(recs[r['query_id']]['context'])) for r in rows) / n, 2),
    "categories": {"title": "표 제목(섹션 제목 + ToTTo 페이지 제목)", "header_path": "행 경로 + 열 경로(' > ' 포함)",
                   "leaf_prefix": "문장 앞 행·열 잎 라벨", "value": "셀 값",
                   "frame": "문장 틀 글자(In the table ', among, the value of, is, ., : , / )",
                   "question": "Question: + 질문", "rest": "시스템 프롬프트·채팅 템플릿·Context:·줄바꿈·Answer:"},
}, indent=1, ensure_ascii=False))
