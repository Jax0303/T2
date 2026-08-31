import json, sys
from pathlib import Path
sys.path[:0] = ['.', 'scripts']
from rag_agent.bench.hitab import load_queries
from point3_reconstruction_cost import build_table_paths
from tree_reconstruct_hitab_raw import align, tree_lines

_, tables = load_queries('data/hitab', 'dev')
raw_dir = Path('data/hitab/data/tables/raw')
n_tab = n_raw = n_noraw = 0
fail = {'no_texts':0,'no_tree':0,'bad_nh':0,'align_none':0,'ok':0}
ch=ct=rh=rt=0
per=[]
for tid, bt in tables.items():
    n_tab += 1
    f = raw_dir / f'{tid}.json'
    if not f.exists():
        n_noraw += 1; continue
    n_raw += 1
    raw = json.load(open(f))
    texts = raw.get('texts') or []
    if not texts: fail['no_texts'] += 1; continue
    cols_c,_ = tree_lines(raw.get('top_root') or {}, 'top')
    rows_c,_ = tree_lines(raw.get('left_root') or {}, 'left')
    if not cols_c or not rows_c: fail['no_tree'] += 1; continue
    nhr, nhc = min(rows_c), min(cols_c)
    if nhr <= 0 or nhc <= 0: fail['bad_nh'] += 1; continue
    if align(texts, rows_c, cols_c, nhr, nhc, bt) is None:
        fail['align_none'] += 1; continue
    fail['ok'] += 1
    pt = build_table_paths(raw, bt)
    if pt: 
        a,b,c,d = pt['recon']; ch+=a; ct+=b; rh+=c; rt+=d
print(json.dumps({'n_dev_tables': n_tab, 'n_with_raw': n_raw, 'n_without_raw': n_noraw,
  'stages': fail,
  'align_fail_rate_of_raw': round((n_raw-fail['ok'])/n_raw,4),
  'align_none_rate_of_raw': round(fail['align_none']/n_raw,4),
  'col_recon_acc': round(ch/ct,4) if ct else None, 'col_recon_fail': round(1-ch/ct,4) if ct else None,
  'row_recon_acc': round(rh/rt,4) if rt else None, 'row_recon_fail': round(1-rh/rt,4) if rt else None,
  'col_n': ct, 'row_n': rt}, indent=2))
