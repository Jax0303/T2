import json,re,sys
from pathlib import Path
root=Path('/home/user/T2-1/rag-agent');sys.path.insert(0,str(root))
from rag_agent.eval.metrics import hitab_exact_match_text
out=Path(__file__).resolve().parent/'slide_audit_20260912'
items=[json.loads(x) for x in (out/'blind_review.jsonl').read_text().splitlines()]
mapping=[json.loads(x) for x in (out/'review_key.jsonl').read_text().splitlines()]
# A conservative candidate detector, not a replacement score. No extraction
# from arbitrary sentences, no rescaling, rounding, or partial list matches.
pattern=re.compile(r'(?P<number>[+-]?(?:\d+(?:,\d{3})*)(?:\.\d+)?)\s*(?:percentage points?|meters?|metres?|m|%\s+increase)',re.I)
found={}
for x in items:
 m=pattern.fullmatch(x['prediction'].strip())
 if m and hitab_exact_match_text(m.group('number'),x['reference_answer']):
  found[x['review_id']]={**x,'candidate_reason':'Exact number with explicit unit/direction suffix; verify unit and direction in question/table','status':'pending independent review; not human scored'}
counts={}
for row in mapping:
 counts.setdefault(row['run'],0)
 if row['review_id'] in found and not row['automatic_correct']:counts[row['run']]+=1
(out/'format_candidates.json').write_text(json.dumps({'rule':pattern.pattern,'extra_candidates_by_run':counts,'items':list(found.values()),'score_changed':False},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(counts,indent=2))
