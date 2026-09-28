"""MultiHiertt 882건 답변 표본의 조회·셀 1개 211문항: 근거 셀 하나로 답이 정해지는가 (2026-09-28, 새 실행 없음).

실행:  .venv/bin/python results/mh_lookup1_evidence_20260928/analyze.py   (rag-agent/ 에서)
입력:  공식 train.json(리비전 고정), 최종 버전 답변 cap300_20260924/cell_uniq.jsonl,
       표 전체 cap300_20260924/fulltable.jsonl, 1,000자 청크(최종 규칙) reader_format_20260927/test/mh_chunk_final.jsonl,
       keywords.txt(실행 전 고정), labels.csv(문항을 읽고 붙인 판정).
출력:  summary.json (이 폴더).

1단계  정답(공식 answer)과 근거 셀 값(table_description 의 마지막 " is " 뒤)을 비교한다(숫자는 $ , % 공백 제거 후 비교).
2단계  정답이 셀 값과 다른 문항을 keywords.txt 로 (가) 비교·최상급 / (나) 나머지로 나눈다.
3단계  labels.csv: 211문항 전부를 읽고 "근거 셀이 아닌 셀의 값이 바뀌면 정답이 바뀔 수 있는가"로 판정한다.
       S 최상급·비교, R 순위, C 개수·증가 여부·합계(조건부 합계·평균 포함), N 근거 셀 하나로 정해짐.
       질문이 요구하는 것으로 판정한다(정답 주석이 질문과 맞지 않는 문항도 질문 기준).
"""
import csv, json, os, re
from collections import Counter
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
P = lambda *a: os.path.join(ROOT, *a)
path = hf_hub_download('yilunzhao/MultiHiertt', 'multihiertt_data/train.json', repo_type='dataset',
                       revision='f18473da528dede3d9ce2274366d9ee8102ea0fd')
D = {x['uid']: x for x in json.load(open(path))}
jl = lambda r: {json.loads(l)['query_id']: json.loads(l) for l in open(P(r)) if l.strip()}
OURS = jl('results/mh_arms/cap300_20260924/cell_uniq.jsonl')
FULL = jl('results/mh_arms/cap300_20260924/fulltable.jsonl')
CHUNK = jl('results/reader_format_20260927/test/mh_chunk_final.jsonl')
KWS = [l.strip() for l in open(os.path.join(HERE, 'keywords.txt')) if l.strip() and not l.startswith('#')]
KW = re.compile(r'\b(' + '|'.join(re.escape(k) for k in KWS) + r')\b')


def num(s):
    s = re.sub(r'[\$,%\s]', '', str(s)).strip('()')
    try:
        return float(s)
    except ValueError:
        return None


def cell_value(e):
    desc = e['table_description'].get(e['qa']['table_evidence'][0], '').strip().rstrip('.').strip()
    return desc.rsplit(' is ', 1)[-1].strip()


def same(v, a):
    return (num(v) is not None and num(a) is not None and abs(num(v) - num(a)) <= 1e-6 * max(1, abs(num(a)))) or v == a


ids = sorted(q for q, r in OURS.items() if r['layer'] == 'lookup_m1')
diff = [q for q in ids if not same(cell_value(D[q]), str(D[q]['qa']['answer']).strip())]
A = [q for q in diff if KW.search(OURS[q]['question'].lower())]
B = [q for q in diff if q not in A]
lab = {r['query_id']: r for r in csv.DictReader(open(os.path.join(HERE, 'labels.csv'), encoding='utf-8'))}
assert set(lab) == set(ids), 'labels.csv 가 211문항과 다르다'
assert all(lab[q]['answer_ne_cell'] == str(int(q in diff)) for q in ids)
assert all(lab[q]['keyword'] == (m.group(1) if (m := KW.search(OURS[q]['question'].lower())) else '') for q in ids)


def acc(qs):
    out = {'n': len(qs)}
    for name, src in (('ours', OURS), ('chunk_final', CHUNK), ('fulltable', FULL)):
        out[name] = sum(src[q]['answer_correct'] for q in qs)
    out['ours_retrieval'] = sum(OURS[q]['retrieval_correct'] for q in qs)
    return out


multi = [q for q in ids if lab[q]['label'] != 'N']
rest = [q for q in ids if lab[q]['label'] == 'N']
summary = {
    'keywords_sha256': __import__('hashlib').sha256(open(os.path.join(HERE, 'keywords.txt'), 'rb').read()).hexdigest(),
    'n_lookup_m1': len(ids),
    'answer_ne_cell': {'n': len(diff), 'A_keyword': acc(A), 'B_rest': acc(B)},
    'answer_eq_cell': {'n': len(ids) - len(diff), 'keyword_hits': sum(1 for q in ids if q not in diff and lab[q]['keyword'])},
    'A_keyword_labels': dict(Counter(lab[q]['label'] for q in A)),
    'B_rest_labels': dict(Counter(lab[q]['label'] for q in B)),
    'answer_eq_cell_labels': dict(Counter(lab[q]['label'] for q in ids if q not in diff)),
    'not_determined_by_one_cell': {'n': len(multi), 'share': round(len(multi) / len(ids), 4),
                                   'S_superlative_compare': sum(lab[q]['label'] == 'S' for q in ids),
                                   'R_rank': sum(lab[q]['label'] == 'R' for q in ids),
                                   'C_count_trend_sum': sum(lab[q]['label'] == 'C' for q in ids)},
    'table_I1': {g: dict(Counter(lab[q]['label'] for q in qs), n=len(qs))
                 for g, qs in (('ne_keyword', A), ('ne_rest', B), ('eq_cell', [q for q in ids if q not in diff]))},
    'eq_cell_not_determined': sum(1 for q in ids if q not in diff and lab[q]['label'] != 'N'),
    'note_counts': {k: sum(1 for q in ids if lab[q]['note'].startswith(p) or p in lab[q]['note'])
                    for k, p in (('percent_same_value', '퍼센트 표기'), ('name_exclusion_one_cell', '이름으로 뺀 합계'),
                                 ('threshold_on_self', '기준값이 근거 셀 자신'), ('keyword_in_header', '키워드가 머리글 이름 안'))},
    'example_percent_same_value': next([cell_value(D[q]), str(D[q]['qa']['answer'])] for q in ids if '퍼센트 표기' in lab[q]['note']),
    'exploratory_answer': {'not_determined_by_one_cell': acc(multi), 'determined_by_one_cell': acc(rest)},
}
json.dump(summary, open(os.path.join(HERE, 'summary.json'), 'w'), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))
