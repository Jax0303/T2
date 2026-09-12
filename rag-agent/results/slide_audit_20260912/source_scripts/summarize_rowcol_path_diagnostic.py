import json,sys
from pathlib import Path
from scipy.stats import binomtest
root=Path('/home/user/T2-1/rag-agent');sys.path.insert(0,str(root))
from analysis.validated_tables import load_manifest,read_leg,validate_answers
from rag_agent.eval.artifacts import digest,file_digest,read_records,validate_retrieval
out=Path(__file__).resolve().parent/'slide_audit_20260912'
_,_,arms,ids=load_manifest(root/'analysis/comparison_manifest.v2.json')
rr,leaf,_,lm=arms['rowcol_v2']
pr=read_records(out/'rowcol_path_records.jsonl')
rm=json.loads((out/'rowcol_path.json').read_text())
assert rm['records_sha256']==file_digest(out/'rowcol_path_records.jsonl')
assert set(pr)==set(rr) and len(pr)==1584
assert rm['derivation']['source_records_sha256']==file_digest(root/'results/evaluation_v2/rowcol_v2_records.jsonl')
assert sum('correct' in r for r in pr.values())==1581
for q,r in pr.items():
 validate_retrieval(r)
 for key in ('question','answer','mode','m','correct','excluded','table_id','gold_cells','context_cells','cells_in_context'):
  assert r.get(key)==rr[q].get(key),(q,key)
path,pm=read_leg(out/'rowcol_path_answer_primary.jsonl',verified=True)
assert set(path)==ids
validate_answers(pr,path,'path',ids=ids)
for key in ('reader_details','prompt_sha256','seed','max_new_tokens','scorer'):
 assert pm[key]==lm[key],key
assert pm['retrieval_records_sha256']==file_digest(out/'rowcol_path_records.jsonl')
assert pm['query_ids_sha256']==digest(sorted(ids))
for q in ids:
 validate_retrieval(pr[q])
 assert pr[q]['context_cells']==rr[q]['context_cells'] and pr[q]['correct']==rr[q]['correct']
 assert path[q]['context_sha256']==pr[q]['context_sha256']
 assert path[q]['source_context_sha256']==pr[q]['context_sha256']
 assert path[q]['cells_in_context']==pr[q]['cells_in_context']
 assert path[q]['context_condition']=='retrieved'
 assert path[q]['n_tok']+pm['max_new_tokens']<=pm['context_limit']
assert pm['n']==991 and pm['primary_only'] is True and pm['condition']=='retrieved'
assert abs(pm['answer_accuracy']-sum(a['answer_correct'] for a in path.values())/991)<.00005
audit={x['query_id']:x for x in map(json.loads,(out/'rowcol_context_audit.jsonl').read_text().splitlines())}
common={q for q in ids if all(r[q]['correct'] for r,a,_,_ in arms.values())}
groups={'primary':ids,'common_hit':common,'rowcol_hit':{q for q in ids if rr[q]['correct']},'common_hit_collision':{q for q in common if audit[q]['label_collision']},'common_hit_no_collision':{q for q in common if not audit[q]['label_collision']}}
result={}
for name,qs in groups.items():
 gains=sum(not leaf[q]['answer_correct'] and path[q]['answer_correct'] for q in qs)
 losses=sum(leaf[q]['answer_correct'] and not path[q]['answer_correct'] for q in qs)
 lc=sum(leaf[q]['answer_correct'] for q in qs);pc=sum(path[q]['answer_correct'] for q in qs)
 result[name]=dict(n=len(qs),leaf_correct=lc,path_correct=pc,leaf_em=lc/len(qs),path_em=pc/len(qs),gained=gains,lost=losses,delta=(pc-lc)/len(qs),mcnemar_exact_p=binomtest(gains,gains+losses,.5).pvalue if gains+losses else 1)
result['scope']='posthoc diagnostic, frozen selected cells; not independent method validation'
result['input_tokens']={'path':pm['input_tokens'],'leaf_primary_mean':sum(leaf[q]['n_tok'] for q in ids)/len(ids)}
(out/'path_diagnostic_result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
