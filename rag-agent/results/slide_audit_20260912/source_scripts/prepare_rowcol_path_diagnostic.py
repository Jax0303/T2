import copy,json,sys
from pathlib import Path
root=Path('/home/user/T2-1/rag-agent')
sys.path.insert(0,str(root))
from scripts.answer_accuracy import load_evidence
from scripts.retrieval_accuracy import subtable_context,PAGE_TITLES
from rag_agent.eval.artifacts import Selection,evidence_fields,validate_retrieval,write_pair,file_digest,provenance
source=root/'results/evaluation_v2/rowcol_v2_records.jsonl'
records,meta=load_evidence(source)
tabs={}; titles=json.loads(PAGE_TITLES.read_text()); rows=[]
for q,original in records.items():
 r=copy.deepcopy(original)
 if 'correct' in r:
  cells=set(map(tuple,r['context_cells']))
  old=subtable_context(cells,tabs,'data/hitab',titles,header_mode='leaf')
  assert [u['text'] for u in old]==r['context'],q
  units=subtable_context(cells,tabs,'data/hitab',titles,header_mode='path')
  r.update(evidence_fields(Selection(cells,units,True),set(map(tuple,r['gold_cells']))))
  validate_retrieval(r)
  assert r['context_cells']==sorted(cells)
  assert r['correct']==original['correct'] and r['cells_in_context']==original['cells_in_context']
 rows.append(r)
meta=copy.deepcopy(meta)
meta['source_retrieval_provenance']=meta['provenance']
meta['provenance']=provenance(root)
meta['derivation']={'operation':'frozen selected cells; leaf labels replaced by full paths','source_records_sha256':file_digest(source),'script_sha256':file_digest(__file__),'posthoc_diagnostic':True}
meta['source_arguments']=meta.pop('arguments')
meta['arguments']={'script':str(Path(__file__).resolve()),'header_mode':'path','source':str(source)}
meta['rowcol_header']='path'
out=Path(__file__).resolve().parent/'slide_audit_20260912'
write_pair(out/'rowcol_path_records.jsonl',rows,meta,summary_path=out/'rowcol_path.json')
print('Validated unchanged selected cells and hit flags on all',len(rows),'records')
