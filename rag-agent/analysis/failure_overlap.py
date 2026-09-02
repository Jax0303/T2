#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""표 안 실패에서 정답 셀과 1등 셀이 **인코더에게** 얼마나 같아 보이나.

`results/colsig/VERDICT.md` 가 "질문이 어느 쪽을 지목하는가"를 헤더 경로 차집합으로
쟀다면, 이쪽은 같은 질문을 **벡터 공간에서** 다시 잰다. 두 계측이 일치하는지가
`colsig` 판정의 독립 확인이 된다.

⚠️ **문자열 겹침으로 기전을 주장하지 말 것.** 전체 문장 자카드는 중앙 .825 가 나오는데
그중 상당 부분이 **템플릿과 표 제목**이고 둘 다 같은 표의 두 셀이 **무조건** 공유한다.
내용(행경로+열경로+값)만 보면 **.500** 이다. 2026-09-03 에 전체 문장 .825 를 근거로
"구별 정보가 압축에서 사라진다"고 적었다가 아래 코사인 측정에 반증됐다.

**측정이 말하는 것.** cos(정답,1등) 중앙 .825 대 cos(정답, 같은 표 무작위) 중앙 .454 --
인코더는 두 셀을 잘 구별한다. cos>.99 는 4.2% 뿐이다. 실패의 원인은 셀이 뭉개져서가
아니라 **질의가 오답 셀에 더 가깝기 때문**이다 (질의 코사인 정답 .662 / 1등 .751,
차이 중앙 +.047, 접전(<.005)은 10.8%). `colsig` 의 63.3% 와 같은 이야기다.

  PYTHONPATH=. .venv/bin/python analysis/failure_overlap.py
"""

import json
import random
import statistics as st
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT/"scripts"), str(ROOT/"analysis")]
import numpy as np
import corpus_dump_vs_cell as cdv
from header_path_coverage import load_corpus
from cell_rank_dump import cell_texts
from rag_agent.retrieve.encoders import default_encoder

class A: pass
a=A(); a.dataset="hitab"; a.split="dev"; a.population="hitab_dev_lookup_all"
a.data_dir="data/hitab"; a.cell_scheme="S3c"; a.seed=42; a.max_queries=0
a.rhb_question_types=[]; a.rhb_em_only=False; a.mh_queries=400
C=load_corpus(a); txt=cell_texts(C,"S3c","page")
by={}; 
for n,(t,i,j) in enumerate(C.cell_owner): by[(t,i,j)]=n
bytab={}
for n,(t,i,j) in enumerate(C.cell_owner): bytab.setdefault(t,[]).append(n)

enc=cdv._CachedEncoder(default_encoder(model_name="models/bge-base-cell-ft-p0"),
                       ".cache/corpus_dump_vs_cell","hitab_dev_models/bge-base-cell-ft-p0")
E=enc.encode(txt)                      # L2 정규화됨 -> 내적 = 코사인
print("[emb]", E.shape, "norm", float(np.linalg.norm(E[0])))

rows=[json.loads(l) for l in open(ROOT/"results/colsig/hitab_dev_lookup_all_S3c_page_hybrid0.8_above.jsonl")]
intab=[r for r in rows if r['top_above'] and r['top_above'][0]['table_id']==r['gold_table']]

def content(rp, cp, v):
    # 템플릿·제목을 정규식으로 벗기지 말고 구조화된 필드를 직접 쓴다
    return set(" ".join(str(x) for x in list(rp)+list(cp)+[v]).lower().split())

rnd=random.Random(42)
cos_pair=[]; cos_rand=[]; jac_full=[]; jac_cont=[]; nparse=0
for r in intab:
    t=r['top_above'][0]
    g=by[(r['gold_table'],r['gold_row'],r['gold_col'])]
    w=by[(t['table_id'],t['row'],t['col'])]
    cos_pair.append(float(E[g]@E[w]))
    others=[n for n in bytab[r['gold_table']] if n!=g]
    if others: cos_rand.append(float(E[g]@E[rnd.choice(others)]))
    gs,ws=txt[g].lower().split(),txt[w].lower().split()
    jac_full.append(len(set(gs)&set(ws))/len(set(gs)|set(ws)))
    cg=content(r['gold_row_path'],r['gold_col_path'],r['gold_value'])
    cw=content(t['row_path'],t['col_path'],t['value'])
    jac_cont.append(len(cg&cw)/len(cg|cw)); nparse+=1

f=lambda v:f"중앙 {st.median(v):.3f} 평균 {st.mean(v):.3f} 최소 {min(v):.3f} 최대 {max(v):.3f}"
print(f"\nn={len(intab)}  (템플릿 파싱 성공 {nparse})")
print(f"자카드 전체문장       {f(jac_full)}")
print(f"자카드 내용만(템플릿·제목 제거) {f(jac_cont)}")
print(f"코사인 정답 vs 1등    {f(cos_pair)}")
print(f"코사인 정답 vs 같은표 무작위 {f(cos_rand)}")
print(f"\n코사인 정답-1등 분포:")
for th in (0.90,0.95,0.97,0.99):
    print(f"  >{th}: {sum(1 for x in cos_pair if x>th)}/{len(cos_pair)} = {sum(1 for x in cos_pair if x>th)/len(cos_pair):.3f}")

# 결정적 진단: 질의 벡터가 정답과 1등 중 어느 쪽에 얼마나 가까운가
qs=[r['question'] for r in intab]
Q=enc.encode(qs)
mg=[];mw=[];marg=[]
for k,r in enumerate(intab):
    t=r['top_above'][0]
    g=by[(r['gold_table'],r['gold_row'],r['gold_col'])]
    w=by[(t['table_id'],t['row'],t['col'])]
    a_,b_=float(Q[k]@E[g]),float(Q[k]@E[w])
    mg.append(a_);mw.append(b_);marg.append(b_-a_)
print(f"\n질의 코사인 — 정답 {st.median(mg):.3f} / 1등 {st.median(mw):.3f}")
print(f"차이(1등-정답) 중앙 {st.median(marg):.4f} 평균 {st.mean(marg):.4f} 최대 {max(marg):.4f}")
for th in (0.005,0.01,0.02,0.05):
    k=sum(1 for x in marg if x<th)
    print(f"  차이 < {th}: {k}/{len(marg)} = {k/len(marg):.3f}")
print(f"\n참고: 정답-1등 셀 코사인이 .95 초과인 {sum(1 for x in cos_pair if x>0.95)}건에서")
hi=[marg[k] for k in range(len(intab)) if cos_pair[k]>0.95]
print(f"  질의 차이 중앙 {st.median(hi):.4f}")
