import json, hashlib
from pathlib import Path
import sys

root = Path('/home/user/T2-1/rag-agent')
sys.path.insert(0, str(root))
from rag_agent.eval.artifacts import read_records, file_digest
from analysis.validated_tables import validate_answers, paired_counts, load_manifest

out = Path(__file__).resolve().parent / 'slide_audit_20260912'
out.mkdir(exist_ok=True)
old = root / 'results/retrieval_accuracy'
ref = read_records(old / 't_s3c_hybrid_records.jsonl')
ids = {q for q,r in ref.items() if 'correct' in r and r['mode']=='all' and r['m']==1 and (r.get('aggregation') or 'none')=='none'}
historical = {}
loaded = {}
for tag in ['t_s3c_hybrid','t_chunk1000','t_mt2net_hybrid','t_trag_hetero','t_row_values','t_rowcol_values']:
    rp = old / (tag+'_records.jsonl')
    ap = old / (tag+'_answer_retrieved.jsonl')
    r,a = read_records(rp),read_records(ap)
    validate_answers(r,a,tag)
    historical[tag] = dict(paired_counts(r,a,ids), retrieval_file=str(rp), answer_file=str(ap), retrieval_sha256=file_digest(rp), answer_sha256=file_digest(ap), evidence='historical; not verified v2')
    loaded[tag] = (r,a)
_,_,new,newids = load_manifest(root / 'analysis/comparison_manifest.v2.json')
assert ids == newids
current = {tag: paired_counts(r,a,ids) for tag,(r,a,_,_) in new.items()}
common = {q for q in ids if all(r[q]['correct'] for r,a,_,_ in new.values())}
common_results = {tag: paired_counts(r,a,common) for tag,(r,a,_,_) in new.items()}
payload = dict(n=len(ids),historical=historical,current=current,common_hit_n=len(common),common_hit=common_results)
(out/'audit.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')

# Review every distinct question/prediction pair, including automatically correct
# predictions. The review sheet hides method and automatic correctness.
items, mappings = {}, []
all_runs = dict(loaded)
all_runs.update({tag:(r,a) for tag,(r,a,_,_) in new.items()})
for tag,(r,a) in all_runs.items():
    for q in sorted(ids):
        x=a[q]
        key=hashlib.sha256((q+'\0'+x['pred']).encode()).hexdigest()[:20]
        items[key]=dict(review_id=key,question=x['question'] if 'question' in x else r[q]['question'],prediction=x['pred'],reference_answer=x['answer'],table_id=r[q]['table_id'],judgment='',reason='')
        mappings.append(dict(review_id=key,run=tag,query_id=q,automatic_correct=x['answer_correct']))
with (out/'blind_review.jsonl').open('w',encoding='utf-8') as f:
    for key in sorted(items): f.write(json.dumps(items[key],ensure_ascii=False)+'\n')
with (out/'review_key.jsonl').open('w',encoding='utf-8') as f:
    for row in mappings:f.write(json.dumps(row,ensure_ascii=False)+'\n')
print(json.dumps(payload,ensure_ascii=False,indent=2))
print('Distinct blind review items:',len(items))
