#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 안 실패에서 정답 셀과 1등 셀의 **색인 문장**이 얼마나 겹치는가.

`results/colsig/VERDICT.md` 가 "질문이 어느 쪽을 지목하는가"를 쟀다면 이쪽은
"두 후보가 인코더에게 얼마나 같아 보이는가"를 잰다. 둘은 다른 질문이다.

⚠️ **경로+값만으로 재면 안 된다** -- 자카드 중앙 .414 가 나오는데, 색인되는 실제
문장에는 **표 제목이 들어가고 그 제목은 같은 표의 두 셀이 공유**한다. 실제 색인
문장으로 재면 중앙 .825 다. 겹침을 과소평가하면 "bi-encoder 가 구별 정보를 평균에
묻는다"는 기전 자체를 놓친다.

  PYTHONPATH=. .venv/bin/python analysis/failure_overlap.py
"""
import json, sys, statistics as st
from pathlib import Path
ROOT = Path("/home/user/T2-1/rag-agent")
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"analysis")]
from header_path_coverage import load_corpus
from cell_rank_dump import cell_texts

class A: pass
a = A()
a.dataset="hitab"; a.split="dev"; a.population="hitab_dev_lookup_all"
a.data_dir="data/hitab"; a.cell_scheme="S3c"; a.seed=42; a.max_queries=0
a.rhb_question_types=[]; a.rhb_em_only=False; a.mh_queries=400
C = load_corpus(a)
txt = cell_texts(C, "S3c", "page")
pos = {c:n for n,c in enumerate(C.cell_owner)}
by = {}
for n,(t,i,j) in enumerate(C.cell_owner): by[(t,i,j)] = n

rows=[json.loads(l) for l in open(ROOT/"results/colsig/hitab_dev_lookup_all_S3c_page_hybrid0.8_above.jsonl")]
intab=[r for r in rows if r['top_above'] and r['top_above'][0]['table_id']==r['gold_table']]
ov=[]; ex=[]
for r in intab:
    t=r['top_above'][0]
    gk=(r['gold_table'], r['gold_row'], r['gold_col'])
    wk=(t['table_id'], t['row'], t['col'])
    if gk not in by or wk not in by: continue
    g, w = txt[by[gk]], txt[by[wk]]
    gt, wt = g.lower().split(), w.lower().split()
    inter=len(set(gt)&set(wt)); uni=len(set(gt)|set(wt))
    ov.append(inter/uni)
    if len(ex)<2: ex.append((g,w,inter/uni))
print(f"표 안 실패 {len(ov)}건 — 실제 색인 문장(S3c, page 제목 포함) 토큰 자카드")
print(f"  중앙값 {st.median(ov):.3f}  평균 {st.mean(ov):.3f}")
for th in (0.5,0.7,0.8,0.9):
    k=sum(1 for x in ov if x>=th); print(f"  {th:.0%}+ 겹침: {k}/{len(ov)} = {k/len(ov):.3f}")
print()
for g,w,j in ex:
    print(f"  자카드 {j:.3f}")
    print(f"    정답: {g}")
    print(f"    1등 : {w}")
