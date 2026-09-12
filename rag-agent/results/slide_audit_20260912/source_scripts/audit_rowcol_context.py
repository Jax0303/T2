import json,sys
from pathlib import Path
from collections import defaultdict,Counter
root=Path('/home/user/T2-1/rag-agent')
sys.path.insert(0,str(root))
from analysis.validated_tables import load_manifest
from scripts.retrieval_accuracy import subtable_context,PAGE_TITLES
from rag_agent.bench import hitab_grid as hg
out=Path(__file__).resolve().parent/'slide_audit_20260912'
spec,base,arms,ids=load_manifest(root/'analysis/comparison_manifest.v2.json')
tabs={}
titles=json.loads(PAGE_TITLES.read_text())
stats=Counter(); groups=defaultdict(Counter); rows=[]
common={q for q in ids if all(r[q]['correct'] for r,a,_,_ in arms.values())}
rr,ra,_,_=arms['rowcol_v2']; sr,sa,_,_=arms['s3c_v2']
for q in sorted(ids):
 r=rr[q]
 # Coordinate list used by the verified renderer; no gold-based selection.
 cells={tuple(c) for unit in r['context_units'] for c in unit['cells']}
 leaf=subtable_context(cells,tabs,'data/hitab',titles,header_mode='leaf')
 assert [u['text'] for u in leaf]==r['context'],q
 path=subtable_context(cells,tabs,'data/hitab',titles,header_mode='path')
 assert {tuple(c) for u in path for c in u['cells']}==cells
 tid,i,j=r['gold_cells'][0]
 if tid not in tabs:tabs[tid]=hg.load_table(tid,'data/hitab')
 t=tabs[tid].table
 rp=list(t.row_path(i)); cp=list(t.col_path(j))
 competing=[]
 if r['correct']:
  labels=defaultdict(list)
  for ct,ri,cj in sorted(cells):
   if ct!=tid:continue
   tab=tabs[ct].table
   key=(tuple(tab.row_path(ri)[-1:]),tuple(tab.col_path(cj)[-1:]))
   labels[key].append((ri,cj,str(tab.data[ri][cj])))
  competing=labels[(tuple(rp[-1:]),tuple(cp[-1:]))]
 collision=len({v for ri,cj,v in competing})>1
 lost=len(rp)>1 or len(cp)>1
 x=dict(query_id=q,question=r['question'],answer=r['answer'],row_path=rp,col_path=cp,
        hit=r['correct'],rowcol_correct=ra[q]['answer_correct'],s3c_correct=sa[q]['answer_correct'],
        rowcol_pred=ra[q]['pred'],s3c_pred=sa[q]['pred'],common_hit=q in common,
        ancestor_omitted=lost,label_collision=collision,competing_cells=competing,
        leaf_context=r['context'],path_context=[u['text'] for u in path])
 rows.append(x)
 for name,include in [('all',True),('common_hit',q in common),('rowcol_hit',r['correct'])]:
  if not include:continue
  g=groups[name];g['n']+=1;g['ancestor_omitted']+=lost;g['label_collision']+=collision
  g['rowcol_wrong']+=not ra[q]['answer_correct']
  g['wrong_with_collision']+=collision and not ra[q]['answer_correct']
  g['s3c_correct_rowcol_wrong']+=sa[q]['answer_correct'] and not ra[q]['answer_correct']
  g['s3c_correct_rowcol_wrong_collision']+=sa[q]['answer_correct'] and not ra[q]['answer_correct'] and collision
with (out/'rowcol_context_audit.jsonl').open('w',encoding='utf-8') as f:
 for x in rows:f.write(json.dumps(x,ensure_ascii=False)+'\n')
(out/'rowcol_context_summary.json').write_text(json.dumps(groups,indent=2),encoding='utf-8')
print(json.dumps(groups,indent=2))
examples=[x for x in rows if x['common_hit'] and x['label_collision'] and x['s3c_correct'] and not x['rowcol_correct']]
for x in examples[:3]:print(json.dumps(x,ensure_ascii=False))
