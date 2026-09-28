"""논문 본문(thesis/src/*.md)의 모든 수치를 뽑아 결과 파일과 대조한다 (2026-09-26).

실행:  python3 results/thesis_numbers_20260926/extract.py        (rag-agent/ 어디서든)
입력:  thesis/src/*.md, results/ 아래 기존 결과 파일, PREREG-*.md·THESIS-INTERIM 본문(결과 절)
       .jsonl 은 한 줄씩 읽는다. .npy·데이터셋·모델은 읽지 않는다.
출력:  numbers.csv (UTF-8 BOM), summary.json  — 이 파일과 같은 폴더.
       text_only 열: 결과 파일(json·jsonl·csv)이 아니라 글 기록(md 등)에서만 찾은 수치. 같은 줄의 결과 파일에서도
       같은 값이 나오면 결과 파일을 출처로 적는다(2026-09-28).

구성
  1. 토큰화: 수치 토큰(sha256·커밋 해시 16진 64·40자, p값, 불일치 쌍 a:b, 날짜, 식별자 v3.3u·Qwen3-8B, 표 번호, 일반 수)
  2. 종류(kind) 자동 규칙 + KIND_OVERRIDE(수동)
  3. 원천 후보 집합(SETS): 결과 파일에서 읽거나 레코드에서 다시 센 값. 후보마다 tag
     ('sleaf' = HiTab sleaf 결과, 'ours' = sleaf 가 아닌 본 방법 결과)
  4. 수동 지정(MANUAL) — '# ===== MANUAL' 표시 아래:
       LS        논문 (파일, 줄) → 대조할 후보 집합 이름들 (L(...) 호출로 채움)
       PREFER    같은 값이 여러 조건에 있을 때 줄마다 먼저 볼 필드
       COLPREF   표 행의 열 번호마다 먼저 볼 필드
       EXPECT    (파일, 줄, 토큰[, 몇 번째]) → 가리켜야 할 원천 필드. 값이 안 맞으면 불일치(원천 값)
       NOTFOUND  원천 결과 파일이 없는 수치와 그 사유 (대조하지 않고 출처 못 찾음)
       KIND_OVERRIDE (KO(...) 호출) 종류 수동 지정
  5. 대조: 쓰인 자릿수로 반올림해 같으면 일치(p값은 쓰인 유효숫자/부등호로 판정, a:b 쌍은 정확히 같거나 순서 반대).
"""
import csv, glob, json, math, os, re
from collections import Counter, namedtuple
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))       # rag-agent/
P = lambda *a: os.path.join(ROOT, *a)

# =============================================================== 1. tokenizer
SUP = '⁻⁰¹²³⁴⁵⁶⁷⁸⁹'
INT = r'(?:\d{1,3}(?:,\d{3})+|\d+)'
NUM = rf'(?:{INT}(?:\.\d+)?|\.\d+)'
SCI = rf'(?:\d+(?:\.\d+)?×)?10[{SUP}]+'
TOKEN = re.compile(rf'''
 (?P<sha>(?<![0-9a-f])(?:[0-9a-f]{{64}}|[0-9a-f]{{40}})(?![0-9a-f]))
|(?P<fn>\[\^\d+\])
|(?P<p>(?<![A-Za-z])p\s?[=<>≥≤]\s?(?:{SCI}|{NUM}))
|(?P<cmp>[<>≤≥](?:{SCI}))
|(?P<ratio>(?<![\w.]){INT}:{INT}(?!\d))
|(?P<date>\d{{4}}-\d{{2}}-\d{{2}})
|(?P<arxiv>arXiv:\d{{4}}\.\d{{4,5}})
|(?P<ident>(?<![A-Za-z0-9])[A-Za-z][A-Za-z\-_@]*\d(?:[A-Za-z0-9.\-_]*[A-Za-z0-9])?)
|(?P<sci>{SCI})
|(?P<ver>\d+\.\d+\.\d+)
|(?P<tab>(?<![\w.])\d+-\d+(?!\d))
|(?P<num>[+−]?{NUM}(?:%p|%)?)
''', re.X)


def tokens(line):
    for m in TOKEN.finditer(line):
        typ, s = m.lastgroup, m.group()
        if typ == 'ident' and re.fullmatch(r'[A-G]-\d+', s):
            typ = 'tab'
        yield typ, s, m.start(), m.end()


def context(line, a, b, width=60):
    extra = max(0, width - (b - a))
    lo = max(0, a - extra // 2)
    hi = min(len(line), lo + width)
    lo = max(0, hi - width)
    return line[lo:hi].strip()


SUPD = {c: str(i) for i, c in enumerate('⁰¹²³⁴⁵⁶⁷⁸⁹')}


def parse(s):
    """쓰인 토큰 -> dict(k=num|pct|pp|ratio|p, v=값, dec=소수 자릿수, sig=유효숫자(과학 표기), op)"""
    t = s.replace('−', '-').replace(',', '')
    if re.fullmatch(r'\d+:\d+', t):
        a, b = t.split(':')
        return dict(k='ratio', v=(int(a), int(b)))
    op = None
    m = re.fullmatch(r'p\s?([=<>≥≤])\s?(.*)', t)
    if m:
        op, t = m.group(1), m.group(2)
    if t[:1] in '<>≤≥':
        op, t = t[0], t[1:]
    if '10' in t and any(c in t for c in SUP):
        mant, ex = t.split('×') if '×' in t else ('1', t)
        e = -int(''.join(SUPD[c] for c in ex[2:] if c in SUPD))
        sig = len(mant.replace('.', '')) if '×' in t else 0
        return dict(k='p', v=float(mant) * 10 ** e, op=op or '=', sig=sig, dec=None)
    k = 'num'
    if t.endswith('%p'):
        k, t = 'pp', t[:-2]
    elif t.endswith('%'):
        k, t = 'pct', t[:-1]
    dec = len(t.split('.')[1]) if '.' in t else 0
    if op:
        return dict(k='p', v=float(t), op=op, sig=None, dec=dec)
    return dict(k=k, v=float(t), dec=dec)


# =============================================================== helpers
Cand = namedtuple('Cand', 'typ field value src tag')     # typ: v(값) r(쌍) p(p값) h(해시 문자열)


def V(field, value, src, tag=None):
    return Cand('v', field, value, src, tag)


def mcnemar_p(b, c):
    n = b + c
    if n == 0:
        return 1.0
    s = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * s / 2 ** n)


def MC(field, b, c, src, tag=None):
    return [Cand('r', field, (b, c), src, tag), Cand('p', field + ' p', mcnemar_p(b, c), src, tag)]


def pair_counts(a, b):
    """a,b: dict id->0/1 on same ids -> (a만, b만)"""
    ids = a.keys() & b.keys()
    return sum(1 for i in ids if a[i] and not b[i]), sum(1 for i in ids if b[i] and not a[i])


def jload(r):
    with open(P(r), encoding='utf-8') as fh:
        return json.load(fh)


def jl(r):
    with open(P(r), encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def R(r):
    return 'rag-agent/' + r


def text_cands(doc, lines, tag=None):
    """다른 문서(PREREG 결과 절 등)의 지정 줄에 쓰인 수치를 후보로."""
    out = []
    all_lines = open(P(doc), encoding='utf-8').read().split('\n')
    for ln in lines:
        line = all_lines[ln - 1]
        for typ, s, a, b in tokens(line):
            if typ in ('ident', 'date', 'arxiv', 'tab', 'ver'):
                continue
            rng = re.match(r'[~–][+−]?[\d.,]+(%p|%)', line[b:]) if typ == 'num' else None
            q = parse(s + rng.group(1) if rng else s)
            f = f'{doc}:{ln} 본문 "{s}"'
            if q['k'] == 'ratio':
                out.append(Cand('r', f, q['v'], R(doc), tag))
            elif q['k'] == 'p':
                out.append(Cand('p', f, q['v'], R(doc), tag))
            else:
                out.append(Cand('v', f, q['v'] / 100 if q['k'] in ('pct', 'pp') else q['v'], R(doc), tag))
    return out


# =============================================================== 3. source sets
RA = 'results/retrieval_accuracy'
SETS = {}


def src(fn):
    SETS[fn.__name__] = lru_cache(None)(fn)
    return fn


@src
def hitab_sleaf():
    r = f'{RA}/t_sleaf_gold.json'
    d, S = jload(r), R(r)
    ta = d['type_accuracy']
    out = []
    for t in ('single_cell', 'multi_cell', 'arithmetic'):
        out += [V(f'type_accuracy.{t}.accuracy', ta[t]['accuracy'], S, 'sleaf'),
                V(f'type_accuracy.{t}.n', ta[t]['n'], S),
                V(f'type_accuracy.{t}.success', ta[t]['success'], S, 'sleaf')]
    ex = d['excluded_by_reason']
    out += [V('accuracy_all_mode', d['accuracy_all_mode'], S, 'sleaf'),
            V('n_queries_in_split', d['n_queries_in_split'], S), V('n_scored', d['n_scored'], S),
            V('n_excluded', d['n_excluded'], S), V('n_all_mode', d['n_all_mode'], S),
            V('n_any_mode', d['n_any_mode'], S), V('n_tables', d['n_tables'], S),
            V('n_units', d['n_units'], S),
            V('embedding_input_audit.documents.max_tokens', d['embedding_input_audit']['documents']['max_tokens'], S, 'sleaf'),
            V('encoder_details.max_seq_length', d['encoder_details']['max_seq_length'], S)]
    out += [V(f'excluded_by_reason.{k[:22]}', v, S) for k, v in ex.items()]
    return out


@src
def hitab_kladder():
    """부록 E: sleaf 단일 셀 조회 recall@k — 레코드 gold_rank 로 다시 셈 + SUMMARY_TABLES 표."""
    single = {r['query_id'] for r in jl(f'{RA}/t_sleaf_gold_type_accuracy.jsonl') if r['query_type'] == 'single_cell'}
    ranks = [r['gold_rank'] for r in jl(f'{RA}/t_sleaf_gold_records.jsonl') if r['query_id'] in single]
    S = R(f'{RA}/t_sleaf_gold_records.jsonl')
    out = [V(f'gold_rank<={k} 비율 (단일 셀 {len(ranks)}건)', sum(1 for x in ranks if x <= k) / len(ranks), S, 'sleaf')
           for k in (1, 5, 10, 20)]
    return out + text_cands('results/SUMMARY_TABLES-2026-09-18.md', [56], 'sleaf')


@src
def hitab_type_mcnemar():
    r = f'{RA}/hitab_type_mcnemar_gold.txt'
    S, out, maxp, diffs = R(r), [], [], []
    for line in open(P(r), encoding='utf-8'):
        arm, rest = line.split(' ', 1)
        for seg in rest.split('|'):
            m = re.search(r'(\w+): ([\d.]+) vs ([\d.]+) (\d+):(\d+) p=([\d.e+-]+) \(n=(\d+)\)', seg)
            t, a, o, b, c, p, n = m.groups()
            a, o, b, c, p = float(a), float(o), int(b), int(c), float(p)
            out += [V(f'{arm}.{t}.본방법', a, S, 'sleaf'), V(f'{arm}.{t}.상대', o, S),
                    Cand('r', f'{arm}.{t} 본방법만:상대만', (b, c), S, 'sleaf'),
                    Cand('p', f'{arm}.{t} p', p, S, 'sleaf')]
            diffs += [V(f'{arm}.{t} 본방법−상대', a - o, S, 'sleaf'), V(f'{arm}.{t} 상대−본방법', o - a, S, 'sleaf')]
            if arm in ('randrow', 'tablerag_path', 'tablerag_leaf'):
                maxp.append((p, f'{arm}.{t}'))
    out += diffs
    p, where = max(maxp)
    out.append(Cand('p', f'RandRow·path·leaf 9칸 중 최대 p ({where})', p, S, 'sleaf'))
    return out


def _succ(r, qtype='single_cell'):
    return {x['query_id']: x['retrieval_success'] for x in jl(r) if x['query_type'] == qtype}


@src
def hitab_labelabl():
    out = []
    acc = {}
    for scope in ('gold', 'split'):
        for t in ('s3c', 's3frame', 's2'):
            r = f'{RA}/t_{t}_{scope}_labelabl.json'
            a = jload(r)['type_accuracy']['single_cell']['accuracy']
            acc[t, scope] = a
            out.append(V(f't_{t}_{scope}_labelabl.json type_accuracy.single_cell.accuracy', a, R(r), 'ours' if t == 's3c' else None))
        s = {t: _succ(f'{RA}/t_{t}_{scope}_labelabl_type_accuracy.jsonl') for t in ('s3c', 's3frame', 's2')}
        S = R(f'{RA}/t_{{s3c,s3frame,s2}}_{scope}_labelabl_type_accuracy.jsonl')
        out += MC(f'{scope} 라벨 효과 s3c만:s3frame만', *pair_counts(s['s3c'], s['s3frame']), S, 'ours')
        out += MC(f'{scope} 문장 틀 효과 s3frame만:s2만', *pair_counts(s['s3frame'], s['s2']), S)
        out.append(V(f'{scope} s3c−s3frame', acc['s3c', scope] - acc['s3frame', scope], S, 'ours'))
    return out


@src
def hitab_sleaf_vs_s3c():
    a = _succ(f'{RA}/t_sleaf_gold_type_accuracy.jsonl')
    b = _succ(f'{RA}/t_s3c_gold_labelabl_type_accuracy.jsonl')
    S = R(f'{RA}/t_sleaf_gold_type_accuracy.jsonl + t_s3c_gold_labelabl_type_accuracy.jsonl')
    return MC('sleaf만:s3c만 (단일 셀 991)', *pair_counts(a, b), S, 'sleaf') + [V('n', len(a), S)]


FF = 'results/fair_filter_20260921'
ARMS = ['ours', 'chunk', 'trag_hetero', 'rowcol', 'randrow', 'tablerag_path', 'tablerag_leaf']


@src
def hitab_answer300():
    summ = jload(f'{FF}/summary.json')['arms']
    rows = list(jl(f'{FF}/rows.jsonl'))
    full = {r['query_id']: r for r in jl('results/fulltable_20260924/hitab_rows.jsonl')}
    SS, SR, SF = R(f'{FF}/summary.json'), R(f'{FF}/rows.jsonl'), R('results/fulltable_20260924/hitab_rows.jsonl')
    by = {a: {r['query_id']: r for r in rows if r['arm'] == a} for a in ARMS}
    out = []
    for a in ARMS:
        s, tag = summ[a], ('sleaf' if a == 'ours' else None)
        rr = by[a].values()
        hits = sum(r['retrieval_correct'] for r in rr)
        miss = [r for r in rr if not r['retrieval_correct']]
        drop = [r for r in rr if r['correct_base'] and not r['correct_filtered']]
        hdr = sum(1 for r in drop if 1 not in r['filter_kept_idx'])
        out += [V(f'{a}.retrieval_accuracy', s['retrieval_accuracy'], SS, tag),
                V(f'{a}.answer_base', s['answer_base'], SS, tag),
                V(f'{a}.answer_filtered', s['answer_filtered'], SS, tag),
                V(f'{a}.delta', s['delta'], SS, tag),
                Cand('p', f'{a}.mcnemar_base_vs_filtered.p_value', s['mcnemar_base_vs_filtered']['p_value'], SS, tag),
                V(f'{a}.answer_given_retrieval_hit_base', s['answer_given_retrieval_hit_base'], SS, tag),
                V(f'{a}.filter_input_tokens_mean', s['filter_input_tokens_mean'], SS, tag),
                V(f'{a}.lines_mean', s['lines_mean'], SS, tag), V(f'{a}.lines_kept_mean', s['lines_kept_mean'], SS, tag),
                V(f'{a} 검색 성공 수(retrieval_correct 합)', hits, SR, tag),
                V(f'{a} 검색 실패 수', len(miss), SR, tag),
                V(f'{a} 1−검색 정확도', 1 - hits / len(rr), SR, tag),
                V(f'{a} 검색 실패 시 답변 정확도', sum(r['correct_base'] for r in miss) / max(1, len(miss)), SR, tag),
                V(f'{a} reader_input_tokens 평균', sum(r['reader_input_tokens'] for r in rr) / len(rr), SR, tag),
                V(f'{a} 무필터만 맞힘 수(a_only)', len(drop), SR, tag),
                V(f'{a} 그중 1번 줄(첫 줄)을 버린 수', hdr, SR, tag),
                V(f'{a} 첫 줄 버린 비율', hdr / len(drop), SR, tag)]
        if a != 'ours':
            ca = {q: r['correct_base'] for q, r in by['ours'].items()}
            cb = {q: r['correct_base'] for q, r in by[a].items()}
            out += MC(f'본방법만:{a}만 (correct_base)', *pair_counts(ca, cb), SR, 'sleaf')
            out.append(V(f'본방법−{a} answer_base', summ['ours']['answer_base'] - s['answer_base'], SS, 'sleaf'))
    sl = jload(f'{RA}/t_sleaf_gold.json')['type_accuracy']['single_cell']['accuracy']
    out.append(V('단일 셀 991건 전수(t_sleaf_gold.json) − 300건 표본 검색 정확도', sl - summ['ours']['retrieval_accuracy'], SS, 'sleaf'))
    ft = {q: r['correct_base'] for q, r in full.items()}
    ours = {q: r['correct_base'] for q, r in by['ours'].items()}
    chunk = {q: r['correct_base'] for q, r in by['chunk'].items()}
    fa = sum(ft.values()) / len(ft)
    out += [V('fulltable 답변 정확도', fa, SF),      # 2026-09-28: 'fulltable retrieval(정의상) 1.0' 상수 후보 제거 — 결과 파일에 없음
            V('fulltable reader_input_tokens 평균', sum(r['reader_input_tokens'] for r in full.values()) / len(full), SF),
            V('fulltable−본방법', fa - summ['ours']['answer_base'], SF, 'sleaf'),
            V('n', len(ft), SF)]
    out += MC('본방법만:fulltable만', *pair_counts(ours, ft), SF, 'sleaf')
    out += MC('fulltable만:chunk만', *pair_counts(ft, chunk), SF)
    return out


@src
def hitab_oracle():
    """정답 셀만 넣은 조건(부록 B) — s3c v2 틀(sleaf 아님)."""
    S1, S2 = R('results/evaluation_v2/s3c_v2_answer_gold_primary.jsonl'), R('results/evaluation_v2/s3c_v2_answer_gold_modeall.jsonl')
    r = list(jl('results/evaluation_v2/s3c_v2_answer_gold_primary.jsonl'))
    out = [V('단일 셀 991 answer_correct 비율 (s3c_v2 틀)', sum(x['answer_correct'] for x in r) / len(r), S1), V('n', len(r), S1)]
    grp = Counter(); ok = Counter()
    for x in jl('results/evaluation_v2/s3c_v2_answer_gold_modeall.jsonl'):
        g = ('집계없음' if x['aggregation'] in ('none', None, '') else '산술') + ('_셀1' if x['n_ctx'] == 1 else '_셀2+')
        grp[g] += 1; ok[g] += x['answer_correct']
    for g in grp:
        out += [V(f'{g} 정확도 (s3c_v2 틀)', ok[g] / grp[g], S2), V(f'{g} n', grp[g], S2)]
    return out


# ------------------------------------------------------------------ MultiHiertt
MA = 'results/mh_arms'
GROUPS = ['lookup_m1', 'lookup_m2+', 'arith_m1', 'arith_m2+']
MH_REC = {
    'v33u': f'{MA}/mh_train_cell_hv3.3u_none_doc_records.jsonl',
    'v1': f'{MA}/mh_train_cell_hv1_none_doc_records.jsonl',
    'v33': 'results/mh_interim200_v33/mh_cell_hv33_records.jsonl',
    'chunk': f'{MA}/mh_train_chunk_hv1_none_doc_records.jsonl',
    'trag_hetero': f'{MA}/mh_train_trag_hetero_hv1_none_doc_records.jsonl',
    'rowcol_values': f'{MA}/mh_train_rowcol_values_hv1_none_doc_records.jsonl',
    'tablerag_path': f'{MA}/mh_train_tablerag_path_hv1_none_doc_records.jsonl',
    'tablerag_leaf': f'{MA}/mh_train_tablerag_leaf_hv1_none_doc_records.jsonl',
    'randrow_values': f'{MA}/mh_train_randrow_values_hv1_none_doc_records.jsonl',
    'rowcol_sentence': f'{MA}/mh_train_rowcol_hv1_none_doc_records.jsonl',
    'randrow_sentence': f'{MA}/mh_train_randrow_hv1_none_doc_records.jsonl',
    'rowcol_hv33': f'{MA}/mh_train_rowcol_hv3.3_none_doc_records.jsonl',
    'tablerag_path_hv33': f'{MA}/mh_train_tablerag_path_hv3.3_none_doc_records.jsonl',
    'tablerag_leaf_hv33': f'{MA}/mh_train_tablerag_leaf_hv3.3_none_doc_records.jsonl',
}
OURS_MH = {'v1', 'v33u', 'v33'}


@lru_cache(None)
def mh_doc(arm):
    """qid -> (layer, doc correct), v1 모집단(2,871)으로 제한."""
    d = {}
    for x in jl(MH_REC[arm]):
        if 'excluded' not in x:
            d[x['query_id']] = (x['layer'], x['doc']['correct'])
    if arm != 'v1':
        v1 = mh_doc('v1')
        d = {q: v for q, v in d.items() if q in v1}
    return d


@src
def mh_ret():
    v1 = mh_doc('v1')
    assert len(v1) == 2871, len(v1)
    out = []
    acc = {}
    for arm, r in MH_REC.items():
        d = mh_doc(arm)
        S = R(r) + ('' if 'hv1' in r else ' (v1 모집단 2,871건으로 제한해 다시 셈)')
        tag = 'ours' if arm in OURS_MH else None
        assert len(d) == 2871, (arm, len(d))
        for g in GROUPS + ['ALL']:
            ids = [q for q, (l, _) in d.items() if g == 'ALL' or l == g]
            a = sum(d[q][1] for q in ids) / len(ids)
            acc[arm, g] = a
            out.append(V(f'{arm} {g} doc 정확도', a, S, tag))
    for g in GROUPS + ['ALL']:
        out.append(V(f'v1 {g} n', sum(1 for l, _ in v1.values() if g == 'ALL' or l == g), R(MH_REC['v1'])))

    def cmp(a, b, tag, g):
        da, db = mh_doc(a), mh_doc(b)
        ids = [q for q in da if g == 'ALL' or da[q][0] == g]
        x = {q: da[q][1] for q in ids}; y = {q: db[q][1] for q in ids}
        S = R(MH_REC[a]) + ' vs ' + R(MH_REC[b])
        return MC(f'{a}만:{b}만 {g}', *pair_counts(x, y), S, tag) + [V(f'{a}−{b} {g}', acc[a, g] - acc[b, g], S, tag)]

    base6 = ['chunk', 'trag_hetero', 'rowcol_values', 'tablerag_path', 'tablerag_leaf', 'randrow_values']
    ps24 = []
    for b in base6:
        for g in GROUPS + ['ALL']:
            c = cmp('v33u', b, 'ours', g)
            out += c
            if g != 'ALL':
                ps24.append((c[1].value, f'{b} {g}'))
    S = R(MH_REC['v33u']) + ' vs 비교군 6개'
    out += [Cand('p', f'v3.3u 대 비교군 24칸 중 최대 p ({max(ps24)[1]})', max(ps24)[0], S, 'ours'),
            V('24칸 중 p<.05 칸 수', sum(1 for p, _ in ps24 if p < .05), S, 'ours'),
            V('비교 칸 수', len(ps24), S)]
    ps8 = []
    for b in ('tablerag_path_hv33', 'tablerag_leaf_hv33'):
        for g in GROUPS + ['ALL']:
            c = cmp('v33u', b, 'ours', g)
            out += c
            if g != 'ALL':
                ps8.append(c[1].value)
    out += [V('hv3.3 비교군 8칸 중 p<.05 칸 수', sum(1 for p in ps8 if p < .05), S, 'ours')]
    for g in GROUPS + ['ALL']:
        out += cmp('v33', 'v1', 'ours', g) + cmp('v33u', 'v33', 'ours', g)
        out += cmp('rowcol_sentence', 'rowcol_values', None, g)
        for b in ('tablerag_path', 'tablerag_leaf'):
            out.append(V(f'{b}_hv33−{b} {g}', acc[b + '_hv33', g] - acc[b, g], R(MH_REC[b + '_hv33']), None))
    return out


@src
def mh_meta():
    a, b = f'{MA}/mh_train_cell_hv1_none_doc.json', f'{MA}/mh_train_cell_hv3.3u_none_doc.json'
    d1, d3 = jload(a), jload(b)
    skip = d1['population_skipped_upstream']['needs_text_evidence']
    return [V('hv1 n_units', d1['n_units'], R(a)), V('v3.3u n_units', d3['n_units'], R(b)),
            V('n_tables', d1['n_tables'], R(a)), V('n_queries(표 근거만)', d1['n_queries'], R(a)),
            V('n_docs', d1['n_docs'], R(a)), V('hv1 n_scored', d1['n_scored'], R(a)),
            V('population_skipped_upstream.needs_text_evidence', skip, R(a)),
            V('n_queries + needs_text_evidence (train 전체)', d1['n_queries'] + skip, R(a)),
            V('n_tables / n_docs', d1['n_tables'] / d1['n_docs'], R(a)),
            V('v3.3u n_scored − hv1 n_scored', d3['n_scored'] - d1['n_scored'], R(b)),
            V('v3.3u n_scored', d3['n_scored'], R(b))]


@src
def mh_sample():
    r = f'{MA}/sample_cap300_seed20260913.json'
    d = jload(r)
    out = [V(f'{g} 표본 수', len(v), R(r)) for g, v in d['by_layer'].items()]
    return out + [V('표본 합계', sum(len(v) for v in d['by_layer'].values()), R(r)),
                  V('sample_seed', d['sample_seed'], R(r)), V('stratum_cap', d['stratum_cap'], R(r))]


CAP = f'{MA}/cap300_20260924'
OURS_ANS = {'cell_uniq', 'cell', 'cell_hv33r'}


@lru_cache(None)
def cap_rows(name):
    return {x['query_id']: x for x in jl(f'{CAP}/{name}.jsonl')}


@src
def mh_ans():
    rep = jload(f'{FX}/mh_answer_2885/report.json')      # 가중치 = 최종 머리글 규칙의 그룹 크기(2,885), 2026-09-27
    S = R(f'{FX}/mh_answer_2885/report.json')
    out = []
    for cond, d in rep.items():
        tag = 'ours' if cond in OURS_ANS else None
        for g in GROUPS:
            out.append(V(f'{cond}.groups.{g}.em', d['groups'][g]['em'], S, tag))
            out.append(V(f'{cond}.groups.{g}.n', d['groups'][g]['n'], S))
        out += [V(f'{cond}.weighted_em', d['weighted_em'], S, tag),
                V(f'{cond}.ci95[0]', d['ci95'][0], S, tag), V(f'{cond}.ci95[1]', d['ci95'][1], S, tag)]
        if 'vs_cell' in d:
            v = d['vs_cell']
            out += [V(f'{cond}.vs_cell.weighted_diff', v['weighted_diff_cell_minus_this'], S, 'ours'),
                    V(f'{cond}.vs_cell.ci95[0]', v['ci95'][0], S, 'ours'), V(f'{cond}.vs_cell.ci95[1]', v['ci95'][1], S, 'ours')]
            for g in GROUPS:
                m = v['mcnemar'][g]
                out += [Cand('r', f'{cond}.vs_cell.mcnemar.{g} (본방법만:이 조건만)', (m['cell_only'], m['this_only']), S, 'ours'),
                        Cand('p', f'{cond}.vs_cell.mcnemar.{g}.p', m['p'], S, 'ours')]
    mx = max((rep[c]['vs_cell']['mcnemar'][g]['p'], f'{c} {g}') for c in ('randrow_values', 'tablerag_leaf', 'tablerag_path') for g in GROUPS)
    out.append(Cand('p', f'RandRow·leaf·path 12칸 중 최대 p ({mx[1]})', mx[0], S, 'ours'))
    cu = jload(f'{CAP}/cell_uniq.json')
    SU = R(f'{CAP}/cell_uniq.json')
    tag = 'ours'
    for g, b in cu['by_layer'].items():
        out += [V(f'cell_uniq.by_layer.{g}.retrieval_accuracy_here', b['retrieval_accuracy_here'], SU, tag),
                V(f'cell_uniq.by_layer.{g}.answer_given_retrieval_hit', b['answer_given_retrieval_hit'], SU, tag),
                V(f'cell_uniq.by_layer.{g}.input_tokens_mean', b['input_tokens_mean'], SU, tag)]
    out.append(V('cell_uniq.scorer_self_check.em (채점 상한)', cu['scorer_self_check']['em'], SU))
    ft = jload(f'{CAP}/fulltable.json')
    out.append(V('fulltable.by_layer.ALL.input_tokens_mean', ft['by_layer']['ALL']['input_tokens_mean'], R(f'{CAP}/fulltable.json')))
    th = jload(f'{CAP}/trag_hetero.json')
    out.append(V('trag_hetero.by_layer.ALL.input_tokens_mean', th['by_layer']['ALL']['input_tokens_mean'], R(f'{CAP}/trag_hetero.json')))
    # 짝지은 정오(합친 882건 또는 조회 두 그룹) — jsonl 에서 다시 셈
    def ac(name, layers=None):
        return {q: x['answer_correct'] for q, x in cap_rows(name).items() if layers is None or x['layer'] in layers}
    L = {'lookup_m1', 'lookup_m2+'}
    for a, b, lay, tg in [('rowcol', 'rowcol_values', None, None), ('randrow', 'randrow_values', None, None),
                          ('cell', 'chunk', L, 'ours'), ('cell_uniq', 'chunk', L, 'ours')]:
        out += MC(f'{a}만:{b}만 ({"조회 두 그룹" if lay else "882건"})', *pair_counts(ac(a, lay), ac(b, lay)),
                  R(f'{CAP}/{a}.jsonl') + ' vs ' + R(f'{CAP}/{b}.jsonl'), tg)
    v1l = [x for x in cap_rows('cell').values() if x['layer'] in L]
    out.append(V('v1(cell) 조회 두 그룹 검색 성공 수', sum(x['retrieval_correct'] for x in v1l), R(f'{CAP}/cell.jsonl'), 'ours'))
    for a, b in (('rowcol', 'rowcol_values'), ('randrow', 'randrow_values')):
        out.append(V(f'{a}−{b} weighted_em', rep[a]['weighted_em'] - rep[b]['weighted_em'], S))
    return out


@src
def mh_label():
    out = []
    for name, r in (('라벨 없음(v2)', f'{MA}/mh_cell_hv2.json'), ('L1', f'{MA}/mh_cell_hv2_L1.json')):
        b = jload(r)['by_layer']['ALL']
        out += [V(f'{name} by_layer.ALL.doc.accuracy_all', b['doc']['accuracy_all'], R(r), 'ours'),
                V(f'{name} by_layer.ALL.corpus.accuracy_all', b['corpus']['accuracy_all'], R(r), 'ours')]
    r = 'results/components_20260925/mh_layers.json'
    for x in jload(r):
        if x['layer'] == 'ALL':
            out += [Cand('r', f'{x["scope"]} ALL c_only:e_only(라벨없음만:L1만)', (x['c_only'], x['e_only']), R(r), 'ours'),
                    Cand('p', f'{x["scope"]} ALL p_exact', x['p_exact'], R(r), 'ours')]
    return out + text_cands('PREREG-2026-09-13-table-label.md', [72], 'ours')


@src
def mh_b1():
    r = f'{MA}/mh_train_cell_hv1_none_doc_b1.json'
    bl = jload(r)['by_layer']
    return [V(f'b1 by_layer.{g}.doc.accuracy_all (1위 적중률)', bl[g]['doc']['accuracy_all'], R(r), 'ours') for g in GROUPS]


@src
def batchcheck():
    base = {x['query_id']: x for x in jl(f'{MA}/mh_train_cell_hv1_none_doc_answer_doc_retrieved_fulln_cot384.jsonl')}
    out = []
    for name in ('b1_20', 'cb_120'):
        r = f'{MA}/batchcheck_20260924/{name}.jsonl'
        rows = list(jl(r))
        S = R(r) + ' vs ' + R(f'{MA}/mh_train_cell_hv1_none_doc_answer_doc_retrieved_fulln_cot384.jsonl')
        n = len(rows)
        same_raw = sum(1 for x in rows if x['raw'] == base[x['query_id']]['raw'])
        same_pred = sum(1 for x in rows if x['pred'] == base[x['query_id']]['pred'])
        a = {x['query_id']: base[x['query_id']]['answer_correct'] for x in rows}
        b = {x['query_id']: x['answer_correct'] for x in rows}
        out += [V(f'{name} n', n, S), V(f'{name} 출력 문자열 일치 수', same_raw, S), V(f'{name} 추출 답 일치 수', same_pred, S),
                V(f'{name} 출력 문자열 불일치 비율', 1 - same_raw / n, S), V(f'{name} 추출 답 일치 비율', same_pred / n, S),
                V(f'{name} 정답 수(이번)', sum(b.values()), S), V(f'{name} 정답 수(기존 한 건씩)', sum(a.values()), S)]
        out += MC(f'{name} 정오 불일치(기존만:이번만)', *pair_counts(a, b), S)
        j = jload(f'{MA}/batchcheck_20260924/{name}.json')
        out.append(V(f'{name} generation_seconds/n (초/건)', j['generation_seconds'] / n, R(f'{MA}/batchcheck_20260924/{name}.json')))
    return out + text_cands('PREREG-2026-09-24-mh-answer-cap300.md', [76, 77, 78, 79])


@src
def prereg_uniq():
    return text_cands('PREREG-2026-09-24-mh-answer-cap300.md', [116, 117, 125, 126])


@src
def reader_pilot():
    out = text_cands('PREREG-2026-09-13-reader-qwen3.md', [12, 64, 72]) + text_cands('PREREG-2026-09-23-reader-thinking-pilot.md', [77])
    r = f'{MA}/reader_pilot_cot_rerun_20260923_validation.jsonl'
    rows = list(jl(r))
    ok = sum(1 for x in rows if x.get('answer_correct', x.get('em', 0)))
    return out + [V('cot 재실행 정답 수', ok, R(r)), V('cot 재실행 n', len(rows), R(r))]


@src
def interim_meta():
    return (text_cands('THESIS-INTERIM-2026-09-23.md', [84, 64]) + text_cands('DATA-USE-2026-09-14.md', [12])
            + text_cands('PREREG-2026-09-23-reader-thinking-pilot.md', [15]))


# ------------------------------------------------------------------ 2026-09-27 3단계: 본 방법 = s3c, MultiHiertt 모집단 2,885
FX = 'results/thesis_fix_20260927'
DJ = f'{FX}/derive.json'


@lru_cache(None)
def dj():
    return jload(DJ)


def _p_of(field, v, S, tag):
    return [Cand('r', field, (v['b'], v['c']), S, tag), Cand('p', field + ' p', v['p'], S, tag)]


@src
def hitab_meta():
    """HiTab test 모집단·색인 크기(재실행 s3c 결과 JSON). 정확도는 본 방법(s3c)."""
    r = 'results/rerun_20260926/hitab/hitab_test_gold_s3c.json'
    d, S = jload(r), R(r)
    out = [V('n_queries_in_split', d['n_queries_in_split'], S), V('n_scored', d['n_scored'], S),
           V('n_excluded', d['n_excluded'], S), V('n_all_mode', d['n_all_mode'], S), V('n_any_mode', d['n_any_mode'], S),
           V('n_tables', d['n_tables'], S), V('n_units', d['n_units'], S),
           V('encoder_details.max_seq_length', d['encoder_details']['max_seq_length'], S)]
    out += [V(f'excluded_by_reason.{k[:22]}', v, S) for k, v in d['excluded_by_reason'].items()]
    for t in ('single_cell', 'multi_cell', 'arithmetic'):
        out += [V(f'type_accuracy.{t}.n', d['type_accuracy'][t]['n'], S),
                V(f's3c type_accuracy.{t}.accuracy', d['type_accuracy'][t]['accuracy'], S, 'ours')]
    return out


@src
def hitab_rerun():
    """HiTab 질문의 표 안, 재실행 유형별 정확도와 s3c 대비 McNemar (derive.json <- rerun type_accuracy.jsonl)."""
    h, S = dj()['hitab_retrieval_gold'], R(DJ)
    out = []
    for arm, accs in h['accuracy'].items():
        tag = {'s3c': 'ours', 'sleaf': 'sleaf'}.get(arm)
        for t, a in accs.items():
            out.append(V(f'{arm}.{t}.accuracy', a, S, tag))
    for arm, byt in h['vs_s3c'].items():
        for t, v in byt.items():
            out += _p_of(f'{arm}.{t} s3c만:상대만', v, S, 'ours')
            if 'diff_s3c_minus_this' in v:
                out += [V(f'{arm}.{t} s3c−상대', v['diff_s3c_minus_this'], S, 'ours'),
                        V(f'{arm}.{t} 상대−s3c', -v['diff_s3c_minus_this'], S, 'ours')]
    p, where = h['max_p_randrow_path_leaf_9']
    out += [Cand('p', f'RandRow·path·leaf 9칸 중 최대 p ({where})', p, S, 'ours'),
            V('s3c 셀 문장 최대 토큰', h['s3c_max_doc_tokens'], S, 'ours'), V('s3c 셀 문장 수', h['s3c_n_units'], S),
            V('RandRow·path·leaf 비교 칸 수(3방법 × 3유형)', sum(len(h['vs_s3c'][a]) for a in ('randrow', 'tablerag_path', 'tablerag_leaf')), S)]
    out += [V(f's3c 단일 셀 recall@{k}', v, S, 'ours') for k, v in h['s3c_recall_at_k_single_cell'].items()]
    out += [V(f'{t}.n', n, S) for t, n in h['n'].items()]
    return out


@src
def hitab_s3c_answer():
    """HiTab 300건 답변: s3c(results/s3c_answer_hitab300_20260926) 대 비교군(fair_filter 무필터)·표 전체."""
    a, S = dj()['hitab_answer300'], R(DJ)
    s = a['s3c']
    out = [V('s3c.answer', s['answer'], S, 'ours'), V('s3c.answer_correct', s['answer_correct'], S, 'ours'),
           V('s3c.retrieval', s['retrieval'], S, 'ours'), V('s3c.retrieval_hits', s['retrieval_hits'], S, 'ours'),
           V('s3c.retrieval_misses', s['retrieval_misses'], S, 'ours'),
           V('s3c.1−retrieval', round(1 - s['retrieval'], 4), S, 'ours'),
           V('s3c.answer_given_hit', s['answer_given_hit'], S, 'ours'),
           V('s3c.answer_given_miss', s['answer_given_miss'], S, 'ours'),
           V('s3c.reader_input_tokens_mean', s['reader_input_tokens_mean'], S, 'ours'),
           V('991건 전수 − 300건 표본 검색 정확도(s3c)', a['retrieval_991_minus_300'], S, 'ours'), V('n', a['n'], S)]
    out += [V(f'{arm} 검색 정확도(재실행, 300건)', v, S, 'sleaf' if arm == 'sleaf' else None)
            for arm, v in a['retrieval_rerun_300'].items()]
    for arm, v in a['vs_s3c'].items():
        out += _p_of(f's3c만:{arm}만 (답변)', v, S, 'ours')
        if 'diff_s3c_minus_this' in v:
            out.append(V(f's3c−{arm} 답변', v['diff_s3c_minus_this'], S, 'ours'))
    ft = a['vs_s3c']['fulltable']
    out += [V('fulltable 답변 정확도', ft['answer'], S), V('fulltable 정답 수', ft['answer_correct'], S)]
    ck = next(c for c in hitab_answer300() if c.field == 'chunk reader_input_tokens 평균')
    out.append(V('s3c 리더 입력 토큰 / chunk 리더 입력 토큰', s['reader_input_tokens_mean'] / ck.value, S + ' + ' + ck.src, 'ours'))
    return out


@src
def hitab_answer300_others():
    """fair_filter 300건 중 본 방법이 아닌 비교군 값만(이전 본 방법 sleaf 행 제외)."""
    return [c for c in hitab_answer300() if c.tag != 'sleaf'
            and not c.field.startswith(('ours', '본방법', '단일 셀 991건'))]


@src
def hitab_dup():
    d, S = dj()['hitab_same_sentence_s3c'], R(DJ)
    return [V('s3c 같은 표 안 값 뺀 문장이 같은 셀 수', d['cells_with_same_sentence_in_table'], S, 'ours'),
            V('s3c 같은 표 안 값 뺀 문장이 같은 셀 비율', d['ratio'], S, 'ours'), V('셀 문장 수', d['n_cells'], S)]


@src
def mh_pop():
    """MultiHiertt 모집단: 최종 규칙(재실행 s3c) 2,885 와 처음 규칙(v1) 2,871."""
    a, b = 'results/rerun_20260926/mh/mh_train_s3c.json', f'{MA}/mh_train_cell_hv1_none_doc.json'
    d, v, Sa, Sb = jload(a), jload(b), R(a), R(b)
    skip = d['population_skipped_upstream']['needs_text_evidence']
    ex = d['excluded_by_reason']
    out = [V('n_queries(표 근거만)', d['n_queries'], Sa), V('needs_text_evidence', skip, Sa),
           V('train 전체 = n_queries + needs_text_evidence', d['n_queries'] + skip, Sa), V('n_scored', d['n_scored'], Sa),
           V('n_excluded 합', sum(ex.values()), Sa), V('n_tables / n_docs', d['n_tables'] / d['n_docs'], Sa),
           V('v1 n_scored', v['n_scored'], Sb), V('v1 excluded gold_in_header', v['excluded_by_reason']['gold_in_header'], Sb),
           V('최종 n_scored − v1 n_scored', d['n_scored'] - v['n_scored'], Sa)]
    out += [V(f'excluded_by_reason.{k}', n, Sa) for k, n in ex.items()]
    out += [V(f'{g} n', b_['n'], Sa) for g, b_ in d['by_layer'].items() if g != 'ALL']
    out += [V(f'v1 {g} n', b_['n'], Sb) for g, b_ in v['by_layer'].items() if g != 'ALL']
    return out + text_cands('PREREG-2026-09-24-mh-answer-cap300.md', [11])


@src
def mh_rerun():
    m, S = dj()['mh_retrieval_doc'], R(DJ)
    out = []
    for arm, accs in m['accuracy'].items():
        tag = {'s3c': 'ours', 'sleaf': 'sleaf'}.get(arm)
        out += [V(f'{arm} {g} 정확도', a, S, tag) for g, a in accs.items()]
    for arm, byg in m['vs_s3c'].items():
        for g, v in byg.items():
            out += _p_of(f's3c만:{arm}만 {g}', v, S, 'ours') + [V(f's3c−{arm} {g}', v['diff_s3c_minus_this'], S, 'ours')]
    x = m['six_baselines_24_cells']
    out += [Cand('p', f's3c 대 비교군 24칸 중 최대 p ({x["max_p"][1]})', x['max_p'][0], S, 'ours'),
            V('24칸 중 p<.05 칸 수', x['n_p_lt_05'], S, 'ours'),
            V('비교 칸 수', sum(1 for a, byg in m['vs_s3c'].items() if a != 'sleaf' for g in byg if g != 'ALL'), S)]
    out += [V(f'{g} n', n, S) for g, n in m['n'].items()]
    return out


@src
def mh_rules():
    h, g57, S = dj()['mh_header_rules'], dj()['mh_v1_m1_groups'], R(DJ)
    out = [V('v1 머리글 칸 14건(실패로 셈)', h['v1_gold_in_header_counted_as_failure']['n'], S)]
    for r, accs in h['accuracy'].items():
        out += [V(f'{r} {g} 정확도', a, S, 'ours') for g, a in accs.items()]
    for k in ('v33_vs_v1', 'v33u_vs_v33', 'v33u_vs_v1'):
        a_, b_ = k.split('_vs_')
        for g, v in h[k].items():
            out += _p_of(f'{a_}만:{b_}만 {g}', v, S, 'ours')
    for k in ('diff_v33_minus_v1', 'diff_v33u_minus_v33', 'diff_v33u_minus_v1'):
        out += [V(f'{k} {g}', d, S, 'ours') for g, d in h[k].items()]
    c = h['common_2871']
    out += [V('공통 2,871 n', c['n'], S), V('공통 2,871 v1 맞힘', c['v1_correct'], S, 'ours'),
            V('공통 2,871 v33u 맞힘', c['v33u_correct'], S, 'ours')] + _p_of('공통 2,871 v33u만:v1만', c['v33u_vs_v1'], S, 'ours')
    for name in ('top1', 'budget20'):
        x = g57[name]
        for g in ('lookup_m1', 'arith_m1'):
            out += [V(f'v1 {name} {g} 정확도', x[g]['accuracy'], S, 'ours'), V(f'v1 {name} {g} 맞힘', x[g]['correct'], S, 'ours'),
                    V(f'v1 {name} {g} n', x[g]['n'], S)]
        out.append(Cand('p', f'v1 {name} 조회1 대 산술1 Fisher p', x['fisher_exact_two_sided_p'], S, 'ours'))
    return out


@src
def mh_dev_pop():
    """α 선택에 쓴 MultiHiertt dev(validation) 911건과 표 근거만 필요한 332건(2026-09-27)."""
    a, b = 'results/dev_alpha_20260926/mh_dev/mh_dev_a1.0.json', 'results/reader_format_20260927/mh_dev_pop332.jsonl'
    n911, n332 = jload(a)['n_scored'], sum(1 for _ in jl(b))
    return [V('α 선택 MH dev 채점 문항', n911, R(a)), V('표 근거만 필요한 문항', n332, R(b)),
            V('본문 근거 필요 문항 = 911 − 332', n911 - n332, R(a) + ' − ' + R(b))]


@src
def mh_label_pop():
    r = f'{MA}/mh_cell_hv2.json'
    return [V('라벨 실험(v2) n_scored', jload(r)['n_scored'], R(r))]


# ------------------------------------------------------------------ 2026-09-27 4단계: 새 원천
def _sl(arm):
    return {'s3c': 'ours', 'cell_uniq': 'ours', 'sleaf': 'sleaf'}.get(arm)


@src
def rerun_cmp():
    """재실행 비교(compare.json): 범위별 정확도·맞힘, s3c 대비 b:c·p·Holm p."""
    r = 'results/rerun_20260926/compare.json'
    d, S = jload(r)['item1'], R(r)
    out = []
    for scope, rows in d.items():
        for x in rows:
            out += [V(f'{scope} {x["arm"]} accuracy', x['accuracy'], S, _sl(x['arm'])), V(f'{scope} {x["arm"]} correct', x['correct'], S, _sl(x['arm']))]
            v = x.get('vs_s3c')
            if v:
                out += [Cand('r', f'{scope} s3c만:{x["arm"]}만', (v['b'], v['c']), S, 'ours'),
                        V(f'{scope} s3c만 맞힘 b (대 {x["arm"]})', v['b'], S, 'ours'), V(f'{scope} {x["arm"]}만 맞힘 c', v['c'], S, 'ours'),
                        Cand('p', f'{scope} s3c만:{x["arm"]}만 p', v['p'], S, 'ours'),
                        Cand('p', f'{scope} s3c만:{x["arm"]}만 Holm p', v['p_holm'], S, 'ours')]
    return out


@src
def step4():
    r = f'{FX}/step4_values.json'
    d, S = jload(r), R(r)
    out = [V(f'MH {a} 전달 셀 평균(2,885)', v, S, _sl(a)) for a, v in d['mh_cells_delivered_2885'].items()]
    out += [V(f'HiTab 표 단위 {t}', v, S) for t, v in d['hitab_table_unit_gold']['type_accuracy'].items()]
    out += [V(f'{a} 전달 셀 평균(단일 셀 991)', v, S, _sl(a)) for a, v in d['hitab_cells_delivered_single991'].items()]
    w = d['mh882_weighted']
    out += [V('chunk_final weighted_em', w['chunk_final']['weighted_em'], S), V('chunk_final ci95[0]', w['chunk_final']['ci95'][0], S),
            V('chunk_final ci95[1]', w['chunk_final']['ci95'][1], S),
            V('cell−chunk_final weighted_diff', w['cell_minus_chunk_final']['weighted_diff'], S, 'ours'),
            V('cell−chunk_final ci95[0]', w['cell_minus_chunk_final']['ci95'][0], S, 'ours'),
            V('cell−chunk_final ci95[1]', w['cell_minus_chunk_final']['ci95'][1], S, 'ours')]
    out += [V(f'chunk_final group {g} em', v, S) for g, v in w['groups_em_chunk_final'].items()]
    for k, sp in d['diff_split'].items():
        for part, v in sp.items():
            if isinstance(v, dict):
                out += [V(f'{k} {part} n', v['n'], S), V(f'{k} {part} ours_correct', v['ours_correct'], S, 'ours'),
                        V(f'{k} {part} other_correct', v['other_correct'], S), V(f'{k} {part} diff', v['diff'], S, 'ours')]
            else:
                out.append(V(f'{k} {part}', v, S, 'ours'))
    c = d['mh882_chunk_v1_vs_final']
    m = c['mcnemar_final_vs_v1']
    out += [V('chunk_v1 맞힘', c['chunk_v1'], S), V('chunk_final 맞힘', c['chunk_final'], S),
            Cand('r', 'chunk 최종만:처음만', (m['b'], m['c']), S, None), Cand('p', 'chunk 최종만:처음만 p', m['p'], S, None)]
    return out


@src
def stats3b():
    r = 'results/stats_20260926/stats.json'
    s, S = jload(r), R(r)
    out = []
    for fam in ('item3b_hitab300', 'item3b_mh882_pooled'):
        for k, v in s[fam].items():
            out += [V(f'{fam} {k} correct', v['correct'], S, _sl(k)), Cand('r', f'{fam} 기준만:{k}만', (v['b'], v['c']), S, 'ours'),
                    Cand('p', f'{fam} {k} p', v['p'], S, 'ours'), Cand('p', f'{fam} {k} Holm p', v['p_holm'], S, 'ours')]
    c = s['item3b_mh882_pooled']['cell']            # 기준 = 최종 규칙 셀 문장, cell = 처음 규칙 셀 문장
    out += [V('item3b_mh882_pooled 처음 규칙만 맞힘(c)', c['c'], S, 'ours'), V('item3b_mh882_pooled 최종 규칙만 맞힘(b)', c['b'], S, 'ours')]
    return out


@src
def reader_test():
    r = 'results/reader_format_20260927/test.json'
    d, S = jload(r), R(r)
    b = d['multihiertt_test']
    out = []
    for c, v in b['conditions'].items():
        out += [V(f'test {c} correct', v['correct'], S, _sl(c) or ('ours' if c == 'cell' else None)), V(f'test {c} n', v['n'], S),
                V(f'test {c} cells', v['cells_delivered_mean'], S), V(f'test {c} tokens', v['input_tokens_mean'], S),
                V(f'test {c} answer rate', v['correct'] / v['n'], S, _sl(c) or ('ours' if c == 'cell' else None))]
    for p, v in b['pairs'].items():
        out += [Cand('r', f'test {p}', (v['b'], v['c']), S, 'ours'), Cand('p', f'test {p} p', v['p'], S, 'ours')]
    for g, x in b['groups_exploratory'].items():
        for p, v in x['pairs'].items():
            out += [Cand('r', f'test {g} {p}', (v['b'], v['c']), S, 'ours'), Cand('p', f'test {g} {p} p', v['p'], S, 'ours')]
    for g, x in d['decomposition'].items():
        for k in ('cell', 'chunk'):
            v = x[k]
            for f in ('retrieval_success', 'answer_given_hit', 'answer_given_miss'):
                out += [V(f'decomp {g} {k} {f} rate', v[f]['rate'], S), V(f'decomp {g} {k} {f} k', v[f]['k'], S), V(f'decomp {g} {k} {f} n', v[f]['n'], S)]
        i = x['intersection_both_retrieved']
        out += [V(f'inter {g} n', i['n'], S), V(f'inter {g} cell rate', i['cell']['rate'], S, 'ours'), V(f'inter {g} cell k', i['cell']['k'], S, 'ours'),
                V(f'inter {g} chunk rate', i['chunk']['rate'], S), V(f'inter {g} chunk k', i['chunk']['k'], S),
                Cand('r', f'inter {g} cell만:chunk만', (i['mcnemar_cell_vs_chunk']['b'], i['mcnemar_cell_vs_chunk']['c']), S, 'ours'),
                Cand('p', f'inter {g} cell만:chunk만 p', i['mcnemar_cell_vs_chunk']['p'], S, 'ours')]
    return out


@src
def reader_dev():
    r = 'results/reader_format_20260927/dev.json'
    d, S = jload(r), R(r)
    out = []
    for blk, b in d.items():
        for c, v in b['conditions'].items():
            out += [V(f'dev {blk} {c} correct', v['correct'], S), V(f'dev {blk} {c} n', v['n'], S),
                    V(f'dev {blk} {c} cells', v['cells_delivered_mean'], S), V(f'dev {blk} {c} tokens', v['input_tokens_mean'], S)]
        for p, v in b['pairs'].items():
            out += [Cand('r', f'dev {blk} {p}', (v['b'], v['c']), S, None), Cand('p', f'dev {blk} {p} p', v['p'], S, None)]
        for g, x in b.get('groups_exploratory', {}).items():
            out.append(V(f'dev {blk} {g} n', x['conditions']['cell']['n'], S))
    g = d['multihiertt_dev_primary']['groups_exploratory']
    out.append(V('dev 정답 셀 1개 문항(조회 1 + 산술 1)', g['lookup_m1']['conditions']['cell']['n'] + g['arith_m1']['conditions']['cell']['n'], S))
    return out


@src
def decomp():
    ra_, ri = f'results/answer_decomp_20260927/answer_decomp.json', 'results/answer_decomp_20260927/intersection.json'
    a, i, SA, SI = jload(ra_), jload(ri), R(ra_), R(ri)
    out = []
    for blk, conds in (('hitab', a['hitab_300']), ('mh', {k: v['ALL'] for k, v in a['mh_882'].items()})):
        for c, v in conds.items():
            tag = _sl(c)
            if 'cells_delivered_mean' in v:
                out.append(V(f'decomp {blk} {c} cells', v['cells_delivered_mean'], SA, tag))
                for f in ('retrieval_success', 'answer_given_hit', 'answer_given_miss'):
                    out += [V(f'decomp {blk} {c} {f} rate', v[f]['rate'], SA, tag), V(f'decomp {blk} {c} {f} k', v[f]['k'], SA, tag),
                            V(f'decomp {blk} {c} {f} n', v[f]['n'], SA, tag)]
            out += [V(f'decomp {blk} {c} answer rate', v['answer']['rate'], SA, tag), V(f'decomp {blk} {c} answer k', v['answer']['k'], SA, tag)]
    pairs = {'hitab s3c_vs_chunk': i['hitab_300']['s3c_vs_chunk']} | {f'mh {k}': v['ALL'] for k, v in i['mh_882'].items() if isinstance(v, dict) and 'ALL' in v}
    for k, x in pairs.items():
        m = x['mcnemar_on_both']
        out += [V(f'inter {k} both n', x['both_retrieved'], SI), V(f'inter {k} ours rate', x['ours_answer_on_both']['rate'], SI, 'ours'),
                V(f'inter {k} ours k', x['ours_answer_on_both']['k'], SI, 'ours'), V(f'inter {k} other rate', x['other_answer_on_both']['rate'], SI),
                V(f'inter {k} other k', x['other_answer_on_both']['k'], SI),
                Cand('r', f'inter {k} ours만:other만', (m['b_ours_only'], m['c_other_only']), SI, 'ours'),
                Cand('p', f'inter {k} ours만:other만 p', m['p'], SI, 'ours'),
                V(f'inter {k} only ours', x['only_ours_retrieved'], SI), V(f'inter {k} only other', x['only_other_retrieved'], SI)]
    return out


@src
def v1cmp():
    r = f'{FX}/mh_answer_v1_compare.json'
    c, S = jload(r), R(r)
    out = [V('v1 본 방법 맞힘', c['reference']['correct'], S, 'ours')]
    for n, v in c['comparisons'].items():
        p = v['pooled']
        out += [V(f'v1cmp {n} correct', v['correct'], S), Cand('r', f'v1cmp v1만:{n}만', (p['b'], p['c']), S, 'ours'),
                Cand('p', f'v1cmp {n} p', p['p'], S, 'ours'), Cand('p', f'v1cmp {n} Holm p', p['p_holm'], S, 'ours')]
    mx = max((v['pooled']['p_holm'], n) for n, v in c['comparisons'].items() if n not in ('fulltable', 'chunk', 'trag_hetero'))
    out.append(Cand('p', f'v1cmp 셀 표현 비교군 9개 중 최대 Holm p ({mx[1]})', mx[0], S, 'ours'))
    return out


@src
def bottleneck():
    rd, rt = 'results/bottleneck_20260927/diag/summary.json', 'results/bottleneck_20260927/test/summary.json'
    d, t, SD, ST = jload(rd), jload(rt), R(rd), R(rt)
    out = []
    for scope in ('hitab_intable', 'hitab_538', 'mh_indoc'):
        out.append(V(f'{scope} fail', d[scope]['fail'], SD))
        out += [V(f'{scope} {b}', n, SD) for b, n in d[scope]['fail_buckets'].items()]
        m = t[scope]['mcnemar']
        out += [V(f'rerank {scope} pre', m['pre_correct'], ST, 'ours'), V(f'rerank {scope} post', m['post_correct'], ST),
                Cand('r', f'rerank {scope} 전만:후만', (m['b_pre_only'], m['c_post_only']), ST, None), Cand('p', f'rerank {scope} Holm p', m['p_holm_3scopes'], ST, None)]
    out.append(V('MH 누락 셀 전부 50위 이내 실패', d['mh_indoc']['fail_all_missing_rank_le50'], SD))
    out += [V(f'{scope} D(정답 셀 1-20위인데 실패)', d[scope]['fail_rank_bins']['1-20'], SD) for scope in ('hitab_intable', 'hitab_538')]
    j = d['hitab_538']['judge_rater_A']
    out += [V(f'판정 (가) {k}', v, SD) for k, v in j['ga'].items()] + [V(f'판정 (다) {k}', v, SD) for k, v in j['da'].items()]
    out.append(V('판정 문항 수', j['n_with_judge'], SD))
    return out


@src
def fix0928():
    """2026-09-28 원고 점검 반영: values.json(기존 기록에서 다시 셈), 판정 (나) 자동 검증."""
    r = 'results/thesis_fix_20260928/values.json'
    d, S = jload(r), R(r)
    out = [V(f'hitab_split {a} 전달 셀 평균(단일 셀 991)', v, S, _sl(a)) for a, v in d['hitab_split_cells_delivered_991'].items()]
    a = d['arith_m2plus_both_retrieved_final']
    out += [V('산술·셀 2개+ 둘 다 검색 성공(최종 대 최종) n', a['n'], S), V('산술·셀 2개+ 둘 다 검색 성공 cell_acc', a['cell_acc'], S, 'ours'),
            V('산술·셀 2개+ 둘 다 검색 성공 chunk_acc', a['chunk_acc'], S)]
    c = d['chunk_v1_vs_final_context']
    out += [V(f'chunk 처음 대 최종 {k}', v, S) for k, v in c.items()]
    j = 'results/judge_verify_20260926/hitab_A_phrase.json'
    jv, SJ = jload(j), R(j)
    out += [V('판정 (나) 자동 검증 아니오 ((가)=예 전체)', jv['yes_all']['아니오'], SJ), V('판정 (나) 자동 검증 아니오 (기간 있는 예)', jv['yes_with_period']['아니오'], SJ),
            V('판정 (나) 검증 단위 수(HiTab test 표)', jv['units'], SJ)]
    return out


@src
def truncation():
    out = []
    for name, r in (('HiTab', 'results/rerun_20260926/hitab/hitab_test_gold_table.json'), ('MultiHiertt', 'results/rerun_20260926/mh/mh_train_table.json')):
        a = jload(r)['embedding_input_audit']['documents']
        out += [V(f'{name} 표 단위 문서 수', a['n'], R(r)), V(f'{name} 표 단위 512 초과', a['n_overflow'], R(r)), V(f'{name} max_seq_length', a['max_seq_length'], R(r))]
    return out


@src
def alpha_dev():
    r = 'results/dev_alpha_20260926/sensitivity_332.json'
    d, S = jload(r), R(r)
    out = []
    for c, v in d['candidates'].items():
        out += [V(f'α {c} {k}', v[k], S) for k in ('n911_correct', 'n911_accuracy', 'rank911', 'rank911_tied', 'n332_correct', 'n332_accuracy', 'rank332', 'rank332_tied')]
    top5 = [d['candidates'][a]['n332_correct'] for a in ('1.0', '0.9', '0.8', '0.7', '0.6')]
    out.append(V('α 1.0~0.6 n332_correct 최대−최소', max(top5) - min(top5), S))
    return out


@src
def mh_test():
    """2026-09-28 MultiHiertt 공개 test 파일 확인(scripts/check_mh_test.py): 문항 수, qa 키별 문항 수, sha256, HF 커밋."""
    r = 'results/mh_test_check_20260928/summary.json'
    d, S = jload(r), R(r)
    return [V('n_questions', d['n_questions'], S), Cand('h', 'sha256', d['sha256'], S, None),
            Cand('h', 'hf_commit', d['hf_commit'], S, None)] + [V(f'n_with_key.{k}', v, S) for k, v in d['n_with_key'].items()]


@src
def recheck():
    """2026-09-28 글 기록 출처 수치 재계산(scripts/recheck_20260928): 셀 문장 중복(v1·v3.3)·고유화 변화(train 검색 색인), MultiHiertt 사본 대조."""
    r, q = 'results/recheck_20260928/uniq_stats.json', 'results/recheck_20260928/mh_release_compare.json'
    ur, c, S, SQ = jload(r), jload(q), R(r), R(q)
    u, res = ur['train_index'], ur['residual_after_v3.3u']
    ch, KINDS = u['v3.3_to_v3.3u'], ('table_text', 'row_col_number', 'table_number')
    k = c['sig_digits_k_matching_all_differing_numeric']
    return [V('train 색인 v1 셀 수', u['v1']['n_cells'], S), V('train 색인 v1 중복 비율', u['v1']['dup_rate'], S),
            V('train 색인 v3.3 셀 수', u['v3.3']['n_cells'], S), V('train 색인 v3.3 중복 비율', u['v3.3']['dup_rate'], S),
            V('공식·재포장 train 대조 uid 수', c['n_compared_common_uid'], SQ),
            *([V('정답 유효숫자 자릿수(차이 난 수치 정답 전부에서 성립)', k[0], SQ)] if len(k) == 1 else []),
            V('question_type 일치율(채점 2,885)', c['question_type_agreement']['rate'], SQ),
            V('v3.3→v3.3u 바뀐 셀 비율(train 색인)', ch['changed_rate'], S),
            *[V(f'v3.3→v3.3u {a} 비율(train 색인 전체 셀 대비)', ch[f'{a}_rate_of_all_cells'], S) for a in KINDS],
            V('v3.3u 적용 후 같은 문장 셀 수(train 색인 + validation 문서 전체)',
              res['train.table_evidence_only']['dup_cells_novalue'] + res['validation.split_all_docs']['dup_cells_novalue'], S)]


@src
def tdesc():
    """2026-09-28 table_description 셀 문장 대 본 방법 검색(사전등록 b4187a0): compare.json(그룹별 정확도·불일치·Holm), gold_dup.json."""
    r, g = 'results/table_description_20260928/compare.json', 'results/table_description_20260928/gold_dup.json'
    d, u, S, SG = jload(r)['groups'], jload(g), R(r), R(g)
    out = []
    for k, v in d.items():
        out += [V(f'tdesc {k} n', v['n'], S), V(f'tdesc {k} 본 방법', v['ours_acc'], S, 'ours'),
                V(f'tdesc {k} table_description', v['table_description_acc'], S),
                *MC(f'tdesc {k} 본 방법만:table_description만', v['b_ours_only'], v['c_desc_only'], S)]
        if 'p_holm_4groups' in v:
            out.append(Cand('p', f'tdesc {k} Holm p(4그룹)', v['p_holm_4groups'], S, None))
    dup = u['same_sentence_without_value_in_doc']
    return out + [V('tdesc 정답 셀 수', u['gold_cells'], SG), V('tdesc 채점 문항', u['scored_queries'], SG),
                  V('tdesc 같은 문장 비율 table_description', dup['table_description_rate'], SG),
                  V('tdesc 같은 문장 비율 본 방법', dup['ours_rate'], SG, 'ours')]


@src
def tro():
    """2026-09-28 TableRAG(Chen) 재구현을 공식 숫자 열 규칙 결과로 대체(PREREG-2026-09-28-tablerag-official-dtype.md 사후 추가):
    results/tablerag_official_20260928/replace/values.json (검색·Holm 가족 재계산·전달 가능 상한, --answers 면 답변),
    이전 판 값(부록 H)은 derive.json·rerun compare.json 그대로."""
    r = 'results/tablerag_official_20260928/replace/values.json'
    d, S = jload(r), R(r)
    rt, out = d['retrieval'], []
    h = rt['hitab_gold']
    for a in ('tablerag_path', 'tablerag_leaf'):
        for t, v in h['accuracy'][a].items():
            out.append(V(f'TRO {a}.{t}.accuracy', v, S))
        for t, v in h['vs_s3c'][a].items():
            out += _p_of(f'TRO {a}.{t} s3c만:상대만', v, S, 'ours')
        out.append(V(f'TRO {a} 전달 셀 평균(단일 셀 991)', h['cells_delivered_single991'][a], S))
    p, where = h['max_p_randrow_path_leaf_9']
    out.append(Cand('p', f'TRO RandRow·path·leaf 9칸 중 최대 p ({where})', p, S, 'ours'))
    sp = rt['hitab_split_family']
    for a, v in sp['family'].items():
        out += [V(f'TRO split {a} accuracy', v['accuracy'], S, _sl(a)), V(f'TRO split {a} correct', v['correct'], S, _sl(a)),
                Cand('r', f'TRO split s3c만:{a}만', (v['b'], v['c']), S, 'ours'), V(f'TRO split s3c만 맞힘 b (대 {a})', v['b'], S, 'ours'),
                V(f'TRO split {a}만 맞힘 c', v['c'], S, 'ours'), Cand('p', f'TRO split s3c만:{a}만 p', v['p'], S, 'ours'),
                Cand('p', f'TRO split s3c만:{a}만 Holm p', v['p_holm'], S, 'ours')]
    out += [V(f'TRO split {a} 전달 셀 평균', v, S, _sl(a)) for a, v in sp['cells_delivered_single991'].items()]
    m = rt['mh_doc']
    for a in ('tablerag_path', 'tablerag_leaf'):
        out += [V(f'TRO {a} {g} 정확도', v, S) for g, v in m['accuracy'][a].items()]
        for g, v in m['vs_s3c'][a].items():
            out += _p_of(f'TRO s3c만:{a}만 {g}', v, S, 'ours')
        out.append(V(f'TRO MH {a} 전달 셀 평균(2,885)', m['cells_delivered_2885'][a], S))
    out += [Cand('p', f'TRO MH s3c만:{a}만 ALL Holm p', v['p_holm'], S, 'ours') for a, v in m['family_ALL'].items()]
    b = d['bound']
    for ds, x in (('HiTab', b['hitab']), ('MultiHiertt', b['multihiertt'])):
        for a in ('leaf', 'path'):
            out += [V(f'TRO 상한 {ds} {a} 비율(공식 규칙)', x['rate']['official'][a], S), V(f'TRO 상한 {ds} {a} 건수(공식 규칙)', x['deliverable']['official'][a], S)]
        out.append(V(f'TRO 상한 {ds} n', x['n'], S))
    # 부록 H 이전 판 → 새 값: 이전 판은 원고가 인용하던 파생 파일 그대로
    old = dj()
    SD, SC = R(DJ), R('results/rerun_20260926/compare.json')
    cmp_old = {x['arm']: x for x in jload('results/rerun_20260926/compare.json')['item1']['hitab_split']}
    for a in ('tablerag_leaf', 'tablerag_path'):
        out += [V(f'TRO 이전 HiTab 표 안 {a} 단일 셀', old['hitab_retrieval_gold']['accuracy'][a]['single_cell'], SD),
                V(f'TRO 새 HiTab 표 안 {a} 단일 셀', h['accuracy'][a]['single_cell'], S),
                V(f'TRO 이전 HiTab 538표 {a}', cmp_old[a]['accuracy'], SC), V(f'TRO 새 HiTab 538표 {a}', sp['family'][a]['accuracy'], S),
                V(f'TRO 이전 MH {a} ALL', old['mh_retrieval_doc']['accuracy'][a]['ALL'], SD), V(f'TRO 새 MH {a} ALL', m['accuracy'][a]['ALL'], S)]
    an = d.get('answers') or {}
    h3 = an.get('hitab300')
    if h3:
        out.append(V('TRO hitab300 n', h3['family']['sleaf']['n'], S))
        for a, v in h3['family'].items():
            out += _p_of(f'TRO hitab300 s3c만:{a}만', v, S, 'ours') + [Cand('p', f'TRO hitab300 s3c만:{a}만 Holm p', v['p_holm'], S, 'ours')]
        for a, x in h3['tablerag'].items():
            out += [V(f'TRO hitab300 {a} cells 전달 300', x['cells_delivered_300'], S), V(f'TRO hitab300 {a} retrieval 300', x['retrieval_300'], S),
                    V(f'TRO hitab300 {a}.answer', x['answer'], S), V(f'TRO hitab300 {a}.answer_given_hit', x['answer_given_hit'], S),
                    V(f'TRO hitab300 {a} retrieval_hits', x['hits'], S), V(f'TRO hitab300 {a} reader_input_tokens', x['reader_input_tokens_mean'], S),
                    V(f'TRO hitab300 s3c−{a} 답변', x['diff_s3c_minus_this'], S, 'ours')]
            y = x['appendix_A']
            out += [V(f'TRO A {a}.answer_base', y['answer_base'], S), V(f'TRO A {a}.answer_filtered', y['answer_filtered'], S),
                    V(f'TRO A {a}.delta', y['delta'], S), Cand('p', f'TRO A {a}.p_value', y['base_vs_filtered']['p_value'], S, None),
                    V(f'TRO A {a}.lines_mean', y['lines_mean'], S), V(f'TRO A {a}.lines_kept_mean', y['lines_kept_mean'], S)]
    return out


@src
def token_totals():
    """2026-09-25 표 전체 토큰 집계(Qwen2.5-7B-Instruct 토크나이저, results/problem_def_audit_20260925/corpus_token_totals.json)."""
    r = 'results/problem_def_audit_20260925/corpus_token_totals.json'
    d, S = jload(r), R(r)
    h = d['hitab_test_tables']
    return [V('HiTab test 표 수', h['n_units'], S), V('HiTab test 표 토큰 합', h['tokens_concatenated_with_sep'], S),
            V('리더 입력 한도', d['limit'], S), V('HiTab test 표 토큰 합 / 입력 한도', h['times_limit_concat'], S),
            V('HiTab test 표 토큰 합 / 입력 한도 (정수 부분, 원고 "13배")', h['tokens_concatenated_with_sep'] // d['limit'], S),
            # 131,072 = YaRN 적용 최대 문맥(외부 문헌, 5.5 참고 조건의 해석)
            V('HiTab test 표 토큰 합 / 131,072', h['tokens_concatenated_with_sep'] / 131072, S + ' + 외부 문헌 131,072')]


@src
def header_history():
    """2026-09-28 4.1·6.5 머리글 규칙 수정 이력 수치(results/header_history_20260928/values.json)."""
    r = 'results/header_history_20260928/values.json'
    d, S = jload(r), R(r)
    return [V(k, v, S) for k, v in d.items()]


@src
def lookup_arith_posthoc():
    """2026-09-28 5.7절 조회 대 산술 1위 적중 사후 분석(최종 버전, PREREG-2026-09-28-lookup-arith-posthoc.md, T4b·T9 는 그 뒤 추가):
    rank1.json(예산 20), posthoc/posthoc.json(T2·T3·T8), posthoc/posthoc_t4b_t9.json(T4b·T9), analyze.json(path overlap)."""
    d = 'results/lookup_vs_arith_20260928'
    r1 = jload(f'{d}/rank1.json')['multihiertt']['최종_버전_v3.3u']
    ph, x = jload(f'{d}/posthoc/posthoc.json'), jload(f'{d}/posthoc/posthoc_t4b_t9.json')
    an = jload(f'{d}/analyze.json')['multihiertt']['groups']
    S1, SP, SX, SA = R(f'{d}/rank1.json'), R(f'{d}/posthoc/posthoc.json'), R(f'{d}/posthoc/posthoc_t4b_t9.json'), R(f'{d}/analyze.json')
    out = []
    for g in ('조회1', '산술1'):
        b = r1[g]['budget20']
        out += [V(f'최종 {g} 예산20 정확도', b['rate'], S1, 'ours'), V(f'최종 {g} 예산20 맞힘', b['hit'], S1, 'ours'),
                V(f'최종 {g} n', b['n'], S1)]
    i = ph['final']['T2']['i_산술_대_조회']
    for k, g in (('a', '산술1'), ('b', '조회1')):
        out += [V(f'T2 {g} 1위 비율', i[k]['rate'], SP, 'ours'), V(f'T2 {g} 1위 적중', i[k]['hit'], SP, 'ours'), V(f'T2 {g} n', i[k]['n'], SP)]
    out.append(Cand('p', 'T2 1위 산술 대 조회 Fisher p', i['fisher_p'], SP, 'ours'))
    m1 = ph['final']['T3']['m층_1위']
    for tp in ('조회', '산술'):
        out += [V(f'T3 {tp} m=1 1위 비율', m1[tp]['m1']['rate'], SP, 'ours'), V(f'T3 {tp} m=1 1위 적중', m1[tp]['m1']['hit'], SP, 'ours'),
                V(f'T3 {tp} m=1 n', m1[tp]['m1']['n'], SP)]
    out.append(Cand('p', 'T3 m=1 조회 대 산술 Fisher p', m1['fisher_p_m1'], SP, 'ours'))
    out += [V(f'T8 {g} m=1 비율', ph['T8'][g]['전체']['m1_share'], SP, 'ours') for g in ('조회1', '산술1')]
    t = x['final']['T4b']
    out += [V('T4b 조회 비교있음 n', t['조회_비교있음_n'], SX), V('T4b 대상 n', t['대상_n'], SX),
            V('T4b 관측 1위 적중', t['hit'], SX, 'ours'), V('T4b 1/m 합', t['sum_1_over_m'], SX),
            Cand('p', 'T4b 푸아송 이항 p', t['poisson_binomial_p'], SX, 'ours')]
    for k in ('m_le20', 'm_gt20'):
        c = x['final']['T9']['조회'][k]
        out += [V(f'T9 조회 {k} 예산20 비율', c['rate'], SX, 'ours'), V(f'T9 조회 {k} 성공', c['b20_success'], SX, 'ours'),
                V(f'T9 조회 {k} n', c['n'], SX)]
    out += [V(f'{g} path overlap 평균', an[g]['path_overlap']['ratio_mean'], SA, 'ours') for g in ('조회1', '산술1')]
    # 정의 상수: 그룹 '정답 셀 1개', m=1 층, T9 경계 m≤20 / m>20
    return out + [V('정답 셀 1개 그룹·m=1 층', 1, SP), V('T9 m 경계', 20, SX)]


@src
def mh_lookup1_evidence():
    """2026-09-28 6.5절·부록 I: MultiHiertt 답변 표본 조회·셀 1개 211문항의 근거 셀 판정(labels.csv 를 analyze.py 로 집계)."""
    r = 'results/mh_lookup1_evidence_20260928/summary.json'
    d, S = jload(r), R(r)
    nd, ne = d['not_determined_by_one_cell'], d['answer_ne_cell']
    out = [V('n_lookup_m1', d['n_lookup_m1'], S), V('answer_ne_cell.n', ne['n'], S),
           V('answer_ne_cell.A_keyword.n', ne['A_keyword']['n'], S), V('answer_ne_cell.B_rest.n', ne['B_rest']['n'], S),
           V('answer_eq_cell.n', d['answer_eq_cell']['n'], S), V('eq_cell_not_determined', d['eq_cell_not_determined'], S),
           V('not_determined.n', nd['n'], S), V('not_determined.share', nd['share'], S),
           V('not_determined.S', nd['S_superlative_compare'], S), V('not_determined.R', nd['R_rank'], S),
           V('not_determined.C', nd['C_count_trend_sum'], S),
           V('determined_by_one_cell.n', d['exploratory_answer']['determined_by_one_cell']['n'], S)]
    for g, t in d['table_I1'].items():
        out += [V(f'table_I1.{g}.{k}', t.get(k, 0), S) for k in ('n', 'S', 'R', 'C', 'N')]
    out += [V(f'note_counts.{k}', v, S) for k, v in d['note_counts'].items()]
    out += [V(f'example_percent_same_value[{i}]', float(v), S) for i, v in enumerate(d['example_percent_same_value'])]
    return out


# =============================================================== 4. ===== MANUAL: 문장 조각(앵커) → 원천
# 줄 번호 대신 그 줄에만 있는 문장 조각으로 찾는다(원고가 고쳐져도 목록을 다시 쓰지 않게). 조각이 0곳 또는 2곳 이상이면 멈춘다.
@lru_cache(None)
def _lines(f):
    return open(P('thesis', 'src', f), encoding='utf-8').read().split('\n')


def AT(f, anchor):
    hits = [i for i, line in enumerate(_lines(f), 1) if anchor in line]
    if len(hits) != 1:
        raise SystemExit(f'앵커가 {len(hits)}곳: {f} {anchor!r}')
    return hits[0]


LS = {}


def L(f, anchors, *sets):
    for a in ([anchors] if isinstance(anchors, str) else anchors):
        LS.setdefault((f, AT(f, a)), []).extend(sets)


HMETA, HRE, HANS, HOTH = 'hitab_meta', 'hitab_rerun', 'hitab_s3c_answer', 'hitab_answer300_others'
CLAIM = ['rerun_cmp', 'mh_rerun', 'reader_test', 'decomp', 'stats3b', HANS, HMETA]
L('00a_abstract_ko.md', 'HiTab과 MultiHiertt에서', *CLAIM)
L('00b_abstract_en.md', 'On HiTab and MultiHiertt', *CLAIM)
L('01_intro.md', '1. **고유 라벨의 기여', 'hitab_labelabl', HMETA)
L('01_intro.md', '본 연구가 MT2Net과 다른 점은 네 가지다.', 'hitab_labelabl', HMETA)
L('02_related.md', '**MT2Net(Zhao et al., 2022).**', 'hitab_labelabl', HMETA)
L('01_intro.md', '2. **셀 단위 검색을 발표된', 'rerun_cmp', 'mh_rerun', HMETA)
L('01_intro.md', '3. **답변 정확도까지', *CLAIM, 'mh_sample')
L('01_intro.md', '4. **한계를 수치로', 'mh_label')
L('01_intro.md', '모든 비교는 실행 전에 사전등록했다.', HANS)
L('03_method.md', '같은 문서 안에 값을 뺀 문장이 똑같은', 'prereg_uniq', 'mh_meta', 'recheck')
L('03_method.md', '이 규칙을 머리글 고친 규칙(v3.3)의 train 검색 색인', 'prereg_uniq', 'recheck')
L('03_method.md', '**고유 라벨(HiTab).**', 'alpha_dev', 'mh_dev_pop')
L('03_method.md', '잎 라벨 머리말의 유무는', 'hitab_sleaf_vs_s3c', 'hitab_sleaf', 'hitab_labelabl', HANS, 'hitab_answer300')
L('03_method.md', '**임베딩.** 셀 문장과 질문을', HRE, HMETA)
L('03_method.md', 'MultiHiertt 리더 설정은 validation 분할 60건', 'reader_pilot')
L('04_setup.md', ['| 질의 수 |', '| 검색 범위 |', '| 검색 정확도 모집단 |', '| 답변 정확도 표본 |'], HMETA, 'mh_pop', 'mh_sample', 'interim_meta')
L('04_setup.md', '**HiTab.** test 분할 1,584 질의', HMETA)
L('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', 'mh_test', 'mh_pop', 'interim_meta', 'recheck')
L('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', HMETA)                     # 2026-09-28 HiTab 머리글 답 336 대비 문장
L('04_setup.md', '본 연구가 쓴 MultiHiertt 사본은', 'interim_meta', 'recheck')
L('04_setup.md', '[^1]: test.json의 sha256은', 'mh_test')
L('04_setup.md', '**데이터 사용 이력.**', 'mh_pop', 'mh_sample', 'mh_dev_pop')
L('04_setup.md', '정답이 음수인 산술 문항은 숫자 형태의 답으로는', 'mh_ans', 'mh_sample')
L('04_setup.md', '**MultiHiertt 전체 정확도.**', 'mh_pop')
L('04_setup.md', 'MultiHiertt 리더는 두 번의 사전등록 파일럿으로', 'reader_pilot')
L('04_setup.md', '**HiTab 답변 표본.**', HANS, HRE, HMETA)
L('04_setup.md', '**MultiHiertt 답변 표본.**', 'mh_sample', 'mh_pop')
L('04_setup.md', '**검정.**', 'stats3b', HANS, 'mh_sample')
L('04_setup.md', 'MultiHiertt 답변 실행은 transformers의 continuous batching', 'batchcheck')
L('05_results.md', ['HiTab 단일 셀 조회 991건에서 세 가지 문장을', 'test 분할의 538개 표를 한 색인에', '표 5-1. 고유 라벨과'], HMETA)
L('05_results.md', ['| 질문이 속한 표 안 | **.9637**', '| 538개 표를 한 색인에 |', '라벨은 두 범위 모두에서 검색 정확도를'], 'hitab_labelabl', HMETA)
T52 = ['| **본 방법** | .9637', '| sleaf (잎 라벨 머리말 변형) | .9586', '| 고정 청크 | .8789', '| TableRAG(Yu) 청크 | .7881', '| RowCol | .7639',
       '| RandRow | .3744', '| TableRAG 셀 검색 재구현(path) | .2896', '| TableRAG 셀 검색 재구현(leaf) | .4067', '| 표 단위 (상한) |']
L('05_results.md', ['| 방법 | 단일 셀 조회 (991)', *T52, 'RandRow·path·leaf와의 비교는 9칸', '주 모집단인 단일 셀 조회에서', '**산술 216건에서는'],
  HRE, 'step4', HMETA)
T53 = ['| **본 방법** | 19.41', '| sleaf (잎 라벨 머리말 변형) | 19.41', '| 고정 청크 | 69.06', '| TableRAG(Yu) 청크 | 52.09', '| RowCol | 22.52',
       '| RandRow | 22.06', '| TableRAG 셀 검색 재구현(path) | 14.48', '| TableRAG 셀 검색 재구현(leaf) | 17.71', '| 참고: 검색 없이 정답 표 전체 입력 | — | — | .7900']
L('05_results.md', ['표 5-4는 단일 셀 조회 300건에서', '표 5-4. HiTab 답변 정확도', *T53, '괄호 안은 검색에 성공한 질의 수다.',
                    '본 방법의 답변 정확도는 .7900이다.', '본 방법은 .7900 =', '**표 전체와의 비교.** 검색 없이 질문의 표'],
  HANS, 'decomp', 'stats3b', HOTH, 'hitab_answer300')
L('05_results.md', '검색에 성공해도 정답률은', HANS, HOTH, 'hitab_oracle')
T54 = ['| **본 방법** | .9575', '| sleaf (잎 라벨 머리말 변형) | .9528', '| 고정 청크 | .8255', '| TableRAG(Yu) 청크 | .8019', '| RowCol | .6179',
       '| TableRAG 셀 검색 재구현(path) | .6274', '| TableRAG 셀 검색 재구현(leaf) | .5047', '| RandRow | .1698']
L('05_results.md', ['표 5-5는 MultiHiertt train', '표 5-5. MultiHiertt 그룹별', '| 방법 | 조회·셀 1개 (212)', *T54,
                    'sleaf를 뺀 비교군 6개의 24칸은 모두', 'sleaf를 뺀 비교군 6개와의 24개', 'HiTab 산술(표 5-2)과 달리'],
  'mh_rerun', 'step4', 'rerun_cmp', 'mh_pop', 'mh_meta')
L('05_results.md', ['표 5-6은 본 방법의 머리글 규칙만', '| v1 (사전등록 당시) | .9434', '| v3.3 (머리글 수정) | .9623',
                    '| **v3.3u (+ 문장 고유화', '처음 규칙(v1)은 14건의 정답 셀을', 'v1에서 v3.3으로 머리글을 고치면'], 'mh_rules', 'mh_pop')
L('05_results.md', ['MultiHiertt가 함께 제공하는 셀 문장(table_description', '표 5-7. 본 방법과 table_description',
                    '| 조회·셀 1개 | 212 |', '| 조회·셀 2개+ | 367 |', '| 산술·셀 1개 | 71 |', '| 산술·셀 2개+ | 2,235 |', '| 전체 | 2,885 | .8634',
                    'Holm p는 네 그룹을 한 묶음으로', '전체 검색 정확도는 본 방법 86.34%'], 'tdesc', 'mh_meta', 'mh_pop')
T56 = ['| **본 방법 (셀 문장, 최종 규칙)** |', '| 고정 청크 (최종 규칙) |', '| 참고: 검색 없이 문서 표 전체 입력 | .5592', '| TableRAG(Yu) 청크 | .5545']
L('05_results.md', ['표 5-8은 네 그룹 882건에서', '표 5-8. MultiHiertt 답변 정확도', '| 방법 | 조회·셀 1개 (211)', *T56,
                    '그룹별 값은 탐색적 결과다. Holm p는', '**주 비교: 고정 청크(최종 규칙).**', '**표 전체와 TableRAG(Yu) 청크.**'],
  'mh_ans', 'reader_test', 'step4', 'stats3b', 'decomp', 'mh_sample')
L('05_results.md', '**나머지 비교군.**', 'v1cmp')
L('05_results.md', ['**검색 정확도와 답변 정확도의 차이.**', '표 5-9. 본 방법의 머리글 규칙별 답변',
                    '| v1 (사전등록 당시) | .5024', '| v3.3 (머리글 수정) | .5355', '| **v3.3u (주 조건)** |', 'v1에서 v3.3u로 바꾸면 가중 전체가'],
  'mh_ans', 'mh_sample')
L('05_results.md', '**머리글 규칙에 대한 민감도.**', 'stats3b', 'step4', 'reader_test')
L('05_results.md', '결과는 예측과 반대였다.', 'mh_label', 'mh_label_pop', 'mh_pop')
L('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', 'lookup_arith_posthoc')   # 2026-09-28 5.7절 최종 버전 교체
T61 = ['| HiTab (300) | 셀 문장 |', '| HiTab (300) | 고정 청크 |', '| MultiHiertt (882) | 셀 문장 |', '| MultiHiertt (882) | 고정 청크 |']
L('06_discussion.md', T61, 'decomp', 'reader_test', 'mh_sample', HANS)
L('06_discussion.md', ['| HiTab | 256 |', '| MultiHiertt | 642 |'], 'decomp', 'reader_test')
L('06_discussion.md', '두 방법 모두 정답 셀 전부를 문맥에 포함한 문항에서는 답변 정확도에 차이가 없었다.', 'step4', 'decomp', 'reader_test')
L('06_discussion.md', 'MultiHiertt를 처음 머리글 규칙으로 비교하면', 'decomp', 'reader_test')
L('06_discussion.md', '**표 전체와의 비교.** 검색 없이 표 전체를', 'stats3b', 'hitab_oracle', HANS, 'reader_test')
L('05_results.md', '**참고 조건의 해석.**', 'stats3b', HANS, 'reader_test', 'token_totals')
L('06_discussion.md', '**MultiHiertt 산술.**', 'mh_ans')
L('06_discussion.md', '처음 규칙의 교집합 결과(38:76)를 보고', 'decomp', 'reader_dev', 'mh_sample', 'alpha_dev')
L('06_discussion.md', '**가설은 지지되지 않았다.**', 'reader_dev')
T63 = ['| HiTab 질문의 표 안 | 36 |', '| | | C: 정답 셀 순위 51 이하 | 16 |', '| HiTab 538개 표를 한 색인에 | 85 |',
       '| | | B: 정답 셀 순위 21~50 | 25 |', '| | | C: 정답 셀 순위 51 이하 | 20 |', '| MultiHiertt 문서 안 | 394 |',
       '| | | F: 정답 셀 일부 누락, 누락 셀의 표는', '| | | G: 정답 셀 일부 누락, 누락 셀 중', '| | | H: 정답 표의 셀은 있으나']
L('06_discussion.md', ['표 6-3. 검색 실패 문항의 분류', *T63, 'MultiHiertt 실패 394건 중', '**재정렬 진단.**', '**정답 표를 못 찾은 40건의 판정.**'],
  'bottleneck', HMETA)
L('06_discussion.md', '**표 단위 색인의 입력 절단.**', 'truncation')
L('06_discussion.md', '**템플릿 선택에 test 표본 사용.**', HANS, 'hitab_answer300')
L('06_discussion.md', '**고정 청크 대비 답변 우위의 집중.**', 'reader_test')
L('06_discussion.md', '**표본 평가.**', HANS, HRE, HMETA, 'mh_pop', 'mh_sample')
L('06_discussion.md', '**MultiHiertt 평가 분할과 사후 변경.**', 'mh_sample')
L('06_discussion.md', '**문장 고유화의 식별자.**', 'prereg_uniq', 'hitab_dup', HMETA, 'recheck')
L('06_discussion.md', '**생성 방식.**', 'batchcheck')
L('07_conclusion.md', '1. **검색 정확도.**', 'rerun_cmp', HRE, 'mh_rerun', HMETA)
L('07_conclusion.md', '3. **답변 정확도.**', *CLAIM)
L('07_conclusion.md', '남은 과제는 다음과 같다.', 'mh_sample', 'hitab_dup')
TA1 = ['| sleaf(이전 조건) |', '| 고정 청크 | .6867', '| TableRAG(Yu) 청크 | .6000', '| RowCol | .4700', '| RandRow | .1533',
       '| TableRAG 셀 검색 재구현(path) | .1400', '| TableRAG 셀 검색 재구현(leaf) | .0833']
L('09_appendix.md', ['검색 문맥에서 필요한 줄만 LLM이', '표 A-1.', *TA1, 'sleaf(이전 조건)는 필터 후'], 'hitab_answer300')
L('09_appendix.md', 'HiTab 단일 셀 조회 991건에서 이 조건의', 'hitab_oracle', HMETA)
L('09_appendix.md', ['| 한 건씩 재실행 대', '| continuous batching 대', '한 건씩 생성하는 방식은 재실행해도'], 'batchcheck')
L('09_appendix.md', '본 방법의 HiTab 단일 셀 조회 991건 검색 정확도는', HRE)
TG1 = [('| 참고: 검색 없이 문서 표 전체 입력 | 검색 없음 |', 'fulltable'), ('| 고정 청크 | 처음 규칙 검색 |', 'chunk'), ('| TableRAG(Yu) 청크 | 처음 규칙 검색 |', 'trag_hetero'),
       ('| RowCol (셀 문장을 이은 변형, 4.2절) |', 'rowcol'), ('| RowCol (값만) |', 'rowcol_values'),
       ('| RowCol (셀 문장을 이은 변형) | 머리글 고친', 'rowcol_hv33'), ('| RandRow (셀 문장을 이은 변형) |', 'randrow'),
       ('| RandRow (값만) |', 'randrow_values'), ('| TableRAG 셀 검색 재구현(path) | 처음 규칙 검색', 'tablerag_path'),
       ('| TableRAG 셀 검색 재구현(path) | 머리글 고친', 'tablerag_path_hv33'), ('| TableRAG 셀 검색 재구현(leaf) | 처음 규칙 검색', 'tablerag_leaf'),
       ('| TableRAG 셀 검색 재구현(leaf) | 머리글 고친', 'tablerag_leaf_hv33')]
L('09_appendix.md', ['RowCol·RandRow·TableRAG 셀 검색 재구현의 882건 답변은', '표 F-1. 처음 규칙의 본 방법', *[a for a, _ in TG1]], 'v1cmp', 'mh_sample')
TG2 = [('| 본 방법 | 20.00 | .8776', 'cell'), ('| 고정 청크 | 51.25', 'chunk'), ('| TableRAG(Yu) 청크 | 46.93', 'trag_hetero')]
L('09_appendix.md', ['표 F-2. 처음 규칙의 정답률 분해', *[a for a, _ in TG2], '두 방법이 모두 검색에 성공한 문항에서 처음 규칙의'], 'decomp', 'reader_test', 'mh_sample')
L('09_appendix.md', ['β 선택에 쓴 MultiHiertt dev(validation) 911건은', '표 G-1.', '| 후보 | 911건 맞힘', '332건 기준에서도 β=1.0이 1위다.'], 'alpha_dev', 'mh_dev_pop')
TH1 = [('| β=1.0 (라벨 없음) |', '1.0'), ('| β=0.9 |', '0.9'), ('| β=0.8 |', '0.8'), ('| β=0.7 |', '0.7'), ('| β=0.6 |', '0.6'),
       ('| β=0.5 |', '0.5'), ('| β=0.4 |', '0.4'), ('| β=0.3 |', '0.3'), ('| β=0.2 |', '0.2'), ('| β=0.1 |', '0.1'), ('| 접두어 |', 'prefix')]
L('09_appendix.md', [a for a, _ in TH1], 'alpha_dev')
L('06_discussion.md', '**정답 근거 주석의 범위.**', 'mh_lookup1_evidence')
L('09_appendix.md', ['6.5절의 수치는 MultiHiertt 답변 표본 882건 중', '1. 정답과 근거 셀 값을 비교했다.', '2. 이 102문항을 분류 전에',
                     '3. 211문항 전부를 읽고', '표 I-1. 조회·셀 1개 211문항의 판정', '| 정답 ≠ 근거 셀 값, 키워드 있음 |',
                     '| 정답 ≠ 근거 셀 값, 키워드 없음 |', '| 정답 = 근거 셀 값 |', '| 합계 | 211 |', '키워드가 있는 91문항은 모두'],
  'mh_lookup1_evidence', 'mh_sample')

# 2026-09-28 원고 점검 반영
T53S = ['| **본 방법** | 20.0 |', '| sleaf (잎 라벨 머리말 변형) | 20.0', '| 표 단위 | 118.1', '| 행 단위 | 22.8', '| 고정 청크 | 67.3',
        '| TableRAG(Yu) 청크 | 54.4', '| TableRAG 셀 검색 재구현(path) | 20.5', '| TableRAG 셀 검색 재구현(leaf) | 20.8', '| RowCol | 22.9']
L('05_results.md', ['**538개 표를 한 색인에 넣은 조건.**', '표 5-3. HiTab 538개 표를', *T53S, 'b는 본 방법만 맞힌 문항 수', '이 범위에서 본 방법(.9142)은'],
  'rerun_cmp', 'fix0928', HMETA)
L('06_discussion.md', '셋째, **색인 단위의 크기**', HRE)
L('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', 'fix0928')
L('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', 'fix0928')
L('07_conclusion.md', '3. **답변 정확도.**', 'fix0928')
_D_ROWS = [i for i, x in enumerate(_lines('06_discussion.md'), 1) if x.startswith('| | | D: ')]   # 표 6-3 D 행: 두 범위에서 글자가 같아 줄 번호로
assert len(_D_ROWS) == 2
for _i in _D_ROWS:
    LS.setdefault(('06_discussion.md', _i), []).append('bottleneck')

# 원고에 sleaf 로 이름을 밝혀 쓴 줄 — 이 줄의 sleaf 값은 '본 방법으로 쓴 sleaf' 가 아니다
SLEAF_NAMED = {(f, AT(f, a)) for f, a in [
    ('01_intro.md', '2. **셀 단위 검색을 발표된'), ('03_method.md', '잎 라벨 머리말의 유무는'),
    ('05_results.md', '| sleaf (잎 라벨 머리말 변형) | .9586'), ('05_results.md', '주 모집단인 단일 셀 조회에서'),
    ('05_results.md', '| sleaf (잎 라벨 머리말 변형) | 19.41'), ('05_results.md', '본 방법의 답변 정확도는 .7900이다.'),
    ('05_results.md', '| sleaf (잎 라벨 머리말 변형) | .9528'), ('05_results.md', 'sleaf를 뺀 비교군 6개와의 24개'),
    ('06_discussion.md', '**템플릿 선택에 test 표본 사용.**'),
    ('09_appendix.md', '검색 문맥에서 필요한 줄만 LLM이'), ('09_appendix.md', '| sleaf(이전 조건) |'),
    ('09_appendix.md', 'sleaf(이전 조건)는 필터 후'),
    ('05_results.md', '| sleaf (잎 라벨 머리말 변형) | 20.0'), ('05_results.md', '이 범위에서 본 방법(.9142)은')]}

# 표 행처럼 한 줄이 한 조건일 때 먼저 볼 필드(정규식)
PREFER = {}


def PF(f, anchor, *pats):
    PREFER[f, AT(f, anchor)] = list(pats)


for _a, _arm in zip(T52, ['s3c', 'sleaf', 'chunk', 'trag_hetero', 'rowcol', 'randrow', 'tablerag_path', 'tablerag_leaf', 'table']):
    PF('05_results.md', _a, rf'^{_arm}\.', rf'^{_arm} 전달', rf'표 단위' if _arm == 'table' else rf'^{_arm}\.')
for _a, _arm in zip(T53, ['s3c', 'sleaf', 'chunk', 'trag_hetero', 'rowcol', 'randrow', 'tablerag_path', 'tablerag_leaf', 'fulltable']):
    _o = 'ours' if _arm == 'sleaf' else _arm
    PF('05_results.md', _a, rf'^decomp hitab {_arm} ', rf'^item3b_hitab300 {_arm} ', rf'기준만:{_arm}만', rf'^{_arm}\.', rf'^{_o}[\. ]', rf':{_arm}만')
for _a, _arm in zip(T54, ['s3c', 'sleaf', 'chunk', 'trag_hetero', 'rowcol', 'tablerag_path', 'tablerag_leaf', 'randrow']):
    PF('05_results.md', _a, rf'^{_arm} ', rf's3c만:{_arm}만 ', rf'^MH {_arm} 전달')
for _a, _c in zip(T56, ['cell_uniq', 'chunk_final', 'fulltable', 'trag_hetero']):
    _t = {'cell_uniq': 'cell', 'chunk_final': 'chunk'}.get(_c, _c)
    PF('05_results.md', _a, '^' + re.escape(_c), rf'^test {_t} ', rf'^test \S+ cell_vs_chunk' if _c == 'chunk_final' else rf'item3b_mh882_pooled {_c} ',
       rf'기준만:{_c}만', rf'^decomp mh {_c} ', rf'^MH {_c} ')
for _a, _c in [('| v1 (사전등록 당시) | .5024', 'cell.'), ('| v3.3 (머리글 수정) | .5355', 'cell_hv33r'), ('| **v3.3u (주 조건)** |', 'cell_uniq')]:
    PF('05_results.md', _a, '^' + re.escape(_c))
for _a, _r in [('| v1 (사전등록 당시) | .9434', 'v1'), ('| v3.3 (머리글 수정) | .9623', 'v33'), ('| **v3.3u (+ 문장 고유화', 'v33u')]:
    PF('05_results.md', _a, rf'^{_r} ')
for _a, (_blk, _arm) in zip(T61, [('hitab', 's3c'), ('hitab', 'chunk'), ('mh', 'cell_uniq'), ('test', 'chunk')]):
    if _blk == 'test':
        PF('06_discussion.md', _a, r'^decomp ALL chunk ', r'^test chunk ')
    else:
        PF('06_discussion.md', _a, rf'^decomp {_blk} {_arm} ')
PF('06_discussion.md', '| HiTab | 256 |', r'^inter hitab ')
PF('06_discussion.md', '| MultiHiertt | 642 |', r'^inter ALL ')
for _a, _arm in zip(TA1, ['ours', 'chunk', 'trag_hetero', 'rowcol', 'randrow', 'tablerag_path', 'tablerag_leaf']):
    PF('09_appendix.md', _a, '^' + re.escape(_arm))
for _a, _n in TG1:
    PF('09_appendix.md', _a, rf'^v1cmp {_n} ', rf'v1만:{_n}만$')
for _a, _c in TG2:
    PF('09_appendix.md', _a, rf'^decomp mh {_c} ')
for _a, _c in TH1:
    PF('09_appendix.md', _a, rf'^α {re.escape(_c)} ')
PF('05_results.md', 'v1에서 v3.3으로 머리글을 고치면', r'^diff_v33_minus_v1 ALL', r'^diff_v33u_minus_v33 ALL', r'^v33만:v1만', r'^v33u만:v33만')
PF('05_results.md', '**머리글 규칙에 대한 민감도.**', r'item3b_mh882_pooled cell ', r'기준만:cell만', r'chunk 최종만:처음만', r'^chunk_v1', r'^chunk_final 맞힘')
PF('05_results.md', '**주 비교: 고정 청크(최종 규칙).**', r'^test ', r'chunk_final', r'cell−chunk_final')
PF('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', r'fulltable', r'trag_hetero')
PF('09_appendix.md', '| 한 건씩 재실행 대', r'^b1_20')
for _a in T63[:2]:
    PF('06_discussion.md', _a, r'^hitab_intable ')
for _a in T63[2:5]:
    PF('06_discussion.md', _a, r'^hitab_538 ', r'^n_tables')
for _a in T63[5:]:
    PF('06_discussion.md', _a, r'^mh_indoc ')
PF('09_appendix.md', '두 방법이 모두 검색에 성공한 문항에서 처음 규칙의', r'^inter mh ')
PF('03_method.md', '**고유 라벨(HiTab).**', r'^α 1\.0 n911', r'^α prefix n911', r'채점 문항', r'표 근거만')
PF('09_appendix.md', '| continuous batching 대', r'^cb_120')
PF('05_results.md', '본 방법은 .7900 =', r'^s3c\.')
PF('05_results.md', '주 모집단인 단일 셀 조회에서', r'^chunk\.', r'^s3c\.', r'^sleaf\.')
PF('06_discussion.md', '두 방법 모두 정답 셀 전부를 문맥에 포함한 문항에서는 답변 정확도에 차이가 없었다.', r'^hitab300', r'^mh882', r'answer_given_hit rate')
PF('06_discussion.md', '**가설은 지지되지 않았다.**', r'^dev multihiertt', r'^dev hitab')
PF('06_discussion.md', '**재정렬 진단.**', r'^rerank')
PF('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', r'판정', r'hitab_538 A', r'^n_tables$')
for _a, _arm in zip(T53S, ['s3c', 'sleaf', 'table', 'row', 'chunk', 'trag_hetero', 'tablerag_path', 'tablerag_leaf', 'rowcol']):
    PF('05_results.md', _a, rf'^hitab_split {_arm} ', rf'^hitab_split s3c만:{_arm}만', rf'\(대 {_arm}\)$', rf'^hitab_split {_arm}만 맞힘')
PF('05_results.md', '이 범위에서 본 방법(.9142)은', r'^hitab_split ')
for _a in ('**538개 표를 한 색인에 넣은 조건.**', 'b는 본 방법만 맞힌 문항 수'):
    PF('05_results.md', _a, r'^type_accuracy\.single_cell\.n$', r'^n_tables$')
PF('06_discussion.md', '셋째, **색인 단위의 크기**', r'^chunk\.arithmetic', r'^s3c\.arithmetic')
PF('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', r'^chunk 처음 대 최종')
for _i, _scope in zip(_D_ROWS, ('hitab_intable', 'hitab_538')):   # 표 6-3 순서: 질문의 표 안, 538개 표 한 색인
    PREFER['06_discussion.md', _i] = [rf'^{_scope} D']
PF('07_conclusion.md', '3. **답변 정확도.**', r'^s3c\.answer_correct$', r'^item3b_hitab300 chunk correct$', r'^n$', r'^test cell(_vs_chunk)? ', r'^test chunk ', r'^inter ',
   r'accuracy', r'^산술·셀 2개\+')
for _f, _a in [('00a_abstract_ko.md', 'HiTab과 MultiHiertt에서'), ('00b_abstract_en.md', 'On HiTab and MultiHiertt'),
               ('01_intro.md', '3. **답변 정확도까지'), ('07_conclusion.md', '3. **답변 정확도.**')]:
    PF(_f, _a, r'^s3c\.answer_correct$', r'^item3b_hitab300 chunk correct$', r'^n$', r'^test cell(_vs_chunk)? ', r'^test chunk ', r'^inter ', r'accuracy')

# 표 행의 열 번호(줄 앞의 '|' 개수) -> 먼저 볼 필드 정규식
COLPREF = {}


def CP(f, anchors, cols):
    for a in anchors:
        COLPREF[f, AT(f, a)] = cols


CP('05_results.md', T52, {2: ['single_cell'], 3: ['multi_cell'], 4: ['arithmetic'], 5: ['전달']})
CP('05_results.md', T53, {2: ['cells'], 3: ['retrieval', '검색 정확도'], 4: [r'answer rate', r'\.answer$', r'answer_base$', r'correct$', '답변 정확도$'],
                          5: ['만:', 'Holm', r' p$'], 6: ['given_hit', 'given_retrieval_hit', 'retrieval_hits', '검색 성공 수'], 7: ['tokens']})
MHR = {2: [r' lookup_m1( |$)'], 3: [r' lookup_m2\+( |$)'], 4: [r' arith_m1( |$)'], 5: [r' arith_m2\+( |$)'], 6: [r' ALL( |$)'], 7: ['전달']}
CP('05_results.md', T54, MHR)
for _a, _g in (('| 조회·셀 1개 | 212 |', 'lookup_m1'), ('| 조회·셀 2개+ | 367 |', r'lookup_m2\+'), ('| 산술·셀 1개 | 71 |', 'arith_m1'),
               ('| 산술·셀 2개+ | 2,235 |', r'arith_m2\+'), ('| 전체 | 2,885 | .8634', 'ALL')):
    CP('05_results.md', [_a], {2: [rf'tdesc {_g} n$'], 3: [rf'tdesc {_g} 본 방법$'], 4: [rf'tdesc {_g} table_description$'],
                               5: [rf'tdesc {_g} 본 방법만'], 6: [rf'tdesc {_g} Holm']})
CP('05_results.md', ['| v1 (사전등록 당시) | .9434', '| v3.3 (머리글 수정) | .9623', '| **v3.3u (+ 문장 고유화'], MHR)
CP('05_results.md', ['| 방법 | 조회·셀 1개 (211)'], {2: [r'lookup_m1'], 3: [r'lookup_m2\+'], 4: [r'arith_m1'], 5: [r'arith_m2\+'], 6: [r'^test cell n']})
CP('05_results.md', T56, {2: [r'lookup_m1'], 3: [r'lookup_m2\+'], 4: [r'arith_m1'], 5: [r'arith_m2\+'], 6: ['correct', '맞힘'],
                          7: [r'cell_vs_chunk', '기준만', r' p$', 'Holm'], 8: [r'weighted_em', r'ci95'], 9: ['cells', '전달'], 10: ['tokens', 'input_tokens']})
MHA = {2: [r'\.lookup_m1'], 3: [r'\.lookup_m2\+'], 4: [r'\.arith_m1'], 5: [r'\.arith_m2\+'],
       6: [r'^\w+\.weighted_em', r'^\w+\.ci95'], 7: [r'vs_cell\.(weighted|ci95)']}
CP('05_results.md', ['| v1 (사전등록 당시) | .5024', '| v3.3 (머리글 수정) | .5355', '| **v3.3u (주 조건)** |'], MHA)
CP('05_results.md', ['| 질문이 속한 표 안 | **.9637**', '| 538개 표를 한 색인에 |'],
   {2: ['t_s3c_'], 3: ['t_s3frame_'], 4: ['t_s2_'], 5: ['라벨 효과'], 6: ['문장 틀 효과']})
CP('06_discussion.md', T61, {3: ['cells'], 4: ['retrieval_success'], 5: ['answer_given_hit'], 6: ['answer_given_miss'], 7: ['answer ', 'correct', 'answer rate']})
CP('06_discussion.md', ['| HiTab | 256 |', '| MultiHiertt | 642 |'],
   {2: [r'both n', r' n$'], 3: ['ours', 'cell'], 4: ['other', 'chunk'], 5: ['만:'], 6: [r' p$'], 7: ['only ours'], 8: ['only other']})
CP('09_appendix.md', TA1, {2: ['answer_base'], 3: ['answer_filtered'], 4: [r'\.delta'], 5: ['p_value'], 6: ['lines_mean', 'lines_kept_mean']})
CP('09_appendix.md', ['| 한 건씩 재실행 대', '| continuous batching 대'],
   {2: [r' n$'], 3: ['출력 문자열 일치 수'], 4: ['추출 답 일치 수'], 5: ['정답 수'], 6: ['정오 불일치'], 7: [r'정오 불일치.* p$']})
CP('09_appendix.md', [a for a, _ in TG1], {4: ['correct'], 5: ['만:'], 6: [r'\d p$|[a-z] p$'], 7: ['Holm']})
CP('09_appendix.md', [a for a, _ in TG2], {2: ['cells'], 3: ['retrieval_success'], 4: ['answer_given_hit'], 5: ['answer_given_miss'], 6: ['answer ']})
CP('09_appendix.md', [a for a, _ in TH1], {2: ['n911'], 3: ['rank911'], 4: ['n332'], 5: ['rank332']})
CP('05_results.md', T53S, {2: ['전달'], 3: [' correct$'], 4: [' accuracy$'], 5: [r'맞힘 b'], 6: [r'만 맞힘 c'], 7: [r'만 p$'], 8: ['Holm p$']})
# 2026-09-28 TableRAG(Chen) 공식 숫자 열 규칙 대체: 표 5-2·5-3·5-5 의 TableRAG 행, 낮은 이유, 부록 H 는 원천 tro
for _a, _arm in ((T52[6], 'tablerag_path'), (T52[7], 'tablerag_leaf')):
    L('05_results.md', _a, 'tro')
    PF('05_results.md', _a, rf'^TRO {_arm}\.', rf'^TRO {_arm} 전달')
for _a, _arm in ((T53S[6], 'tablerag_path'), (T53S[7], 'tablerag_leaf')):
    L('05_results.md', _a, 'tro')
    PF('05_results.md', _a, rf'^TRO split (s3c만:)?{_arm}', rf'^TRO split s3c만 맞힘 b \(대 {_arm}\)')
for _a, _arm in ((T54[5], 'tablerag_path'), (T54[6], 'tablerag_leaf')):
    L('05_results.md', _a, 'tro')
    PF('05_results.md', _a, rf'^TRO {_arm} ', rf'^TRO s3c만:{_arm}만 ', rf'^TRO MH {_arm} 전달')
L('05_results.md', '**TableRAG 셀 검색 재구현이 낮은 이유.**', 'tro')
PF('05_results.md', '**TableRAG 셀 검색 재구현이 낮은 이유.**', r'^TRO 상한')
L('09_appendix.md', '| TableRAG(Chen) | 숫자 열 판정 |', 'tro', HMETA)
L('06_discussion.md', '**머리글 규칙 수정 이력.**', 'header_history')
for _a, _arm in ((T53[6], 'tablerag_path'), (T53[7], 'tablerag_leaf')):
    L('05_results.md', _a, 'tro')
    PF('05_results.md', _a, rf'^TRO hitab300 (s3c만:)?{_arm}')
L('09_appendix.md', 'TableRAG 셀 검색 재구현 두 행은 숫자·날짜 열 판정을', 'tro')
for _a, _arm in ((TA1[5], 'tablerag_path'), (TA1[6], 'tablerag_leaf')):
    L('09_appendix.md', _a, 'tro')
    PF('09_appendix.md', _a, rf'^TRO A {_arm}\.')


def EX(f, anchor, tok, setname, pat, occ=None):
    k = (f, AT(f, anchor), tok) + ((occ,) if occ else ())
    EXPECT[k] = (setname, pat)


# 쓰인 수치가 가리키는 원천 필드(안 맞으면 불일치로 적는다)
EXPECT = {}
# 2026-09-28 재계산으로 연결한 글 기록 출처 수치
_U = '같은 문서 안에 값을 뺀 문장이 똑같은'
EX('03_method.md', _U, '27.6%', 'recheck', 'train 색인 v1 중복 비율')
EX('03_method.md', _U, '429,048', 'recheck', 'train 색인 v1 셀 수')
EX('03_method.md', _U, '11.8%', 'recheck', 'train 색인 v3.3 중복 비율')
EX('03_method.md', _U, '423,473', 'recheck', 'train 색인 v3.3 셀 수')
EX('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', '100%', 'recheck', 'question_type 일치율(채점 2,885)')
EX('04_setup.md', '본 연구가 쓴 MultiHiertt 사본은', '7,830', 'recheck', '공식·재포장 train 대조 uid 수')
EX('04_setup.md', '본 연구가 쓴 MultiHiertt 사본은', '6', 'recheck', '정답 유효숫자 자릿수')
_R = '이 규칙을 머리글 고친 규칙(v3.3)의 train 검색 색인'
EX('03_method.md', _R, '423,473', 'recheck', 'train 색인 v3.3 셀 수')
EX('03_method.md', _R, '14.4%', 'recheck', '바뀐 셀 비율(train 색인)')
for _t, _k in (('2.7%', 'table_text'), ('6.3%', 'row_col_number'), ('6.8%', 'table_number')):
    EX('03_method.md', _R, _t, 'recheck', f'{_k} 비율(train 색인')
    EX('06_discussion.md', '**문장 고유화의 식별자.**', _t, 'recheck', f'{_k} 비율(train 색인')
EX('03_method.md', _R, '0', 'recheck', '적용 후 같은 문장 셀 수')
EX('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', '1,566', 'mh_test', 'n_questions')
EX('04_setup.md', '[^1]: test.json의 sha256은', '15bfe9cc1241e29050a5bcaf7b9639d6907e7895cd07a1f1b874c2fda905ad01', 'mh_test', 'sha256')
EX('04_setup.md', '[^1]: test.json의 sha256은', 'f18473da528dede3d9ce2274366d9ee8102ea0fd', 'mh_test', 'hf_commit')

EX('05_results.md', 'RandRow·path·leaf와의 비교는 9칸', 'p<.0001', HRE, 'RandRow·path·leaf 9칸 중 최대 p')
EX('05_results.md', 'RandRow·path·leaf와의 비교는 9칸', 'p=7.6×10⁻⁶', HRE, 'RandRow·path·leaf 9칸 중 최대 p')
EX('05_results.md', '결과는 예측과 반대였다.', '.8492', 'mh_label', '라벨 없음(v2) by_layer.ALL.doc')
EX('05_results.md', '결과는 예측과 반대였다.', '.8301', 'mh_label', 'L1 by_layer.ALL.doc')
EX('05_results.md', '결과는 예측과 반대였다.', '.2662', 'mh_label', '라벨 없음(v2) by_layer.ALL.corpus')
EX('05_results.md', '결과는 예측과 반대였다.', '.2272', 'mh_label', 'L1 by_layer.ALL.corpus')
EX('04_setup.md', 'MultiHiertt 답변 실행은 transformers의 continuous batching', '10.4', 'batchcheck', 'b1_20 generation_seconds/n')
EX('09_appendix.md', 'sleaf(이전 조건)는 필터 후', '2', 'hitab_answer300', 'ours 그중 1번 줄')
EX('05_results.md', '본 방법은 .7900 =', '0', HANS, 's3c.answer_given_miss')
EX('09_appendix.md', '| continuous batching 대', '35', 'batchcheck', 'cb_120 정답 수(기존', 2)
EX('09_appendix.md', '| continuous batching 대', '1.0', 'batchcheck', 'cb_120 정오 불일치(기존만:이번만) p')
EX('04_setup.md', 'MultiHiertt 답변 실행은 transformers의 continuous batching', 'p=1.0', 'batchcheck', 'cb_120 정오 불일치(기존만:이번만) p')
EX('06_discussion.md', '**생성 방식.**', '1:1', 'batchcheck', 'cb_120 정오 불일치(기존만:이번만)')
EX('05_results.md', '| 질문이 속한 표 안 | **.9637**', 'p=.81', 'hitab_labelabl', 'gold 문장 틀 효과 s3frame만:s2만 p')
EX('05_results.md', '| 538개 표를 한 색인에 |', 'p=.085', 'hitab_labelabl', 'split 문장 틀 효과 s3frame만:s2만 p')
EX('05_results.md', 'sleaf를 뺀 비교군 6개의 24칸은 모두', 'p<.05', 'mh_rerun', 's3c 대 비교군 24칸 중 최대 p')
EX('05_results.md', 'sleaf를 뺀 비교군 6개의 24칸은 모두', 'p=.021', 'mh_rerun', 's3c 대 비교군 24칸 중 최대 p')
EX('05_results.md', 'sleaf를 뺀 비교군 6개와의 24개', 'p=4.4×10⁻⁵⁰', 'mh_rerun', 's3c만:chunk만 ALL p')
EX('05_results.md', 'sleaf를 뺀 비교군 6개와의 24개', 'p=.0040', 'rerun_cmp', 'mh_doc s3c만:sleaf만 Holm p')
EX('09_appendix.md', '| 고정 청크 | .6867', '<10⁻⁵', 'hitab_answer300', 'chunk.mcnemar_base_vs_filtered.p_value')
EX('09_appendix.md', '| TableRAG(Yu) 청크 | .6000', '<10⁻⁵', 'hitab_answer300', 'trag_hetero.mcnemar_base_vs_filtered.p_value')
EX('09_appendix.md', '| RowCol | .4700', '<10⁻⁵', 'hitab_answer300', 'rowcol.mcnemar_base_vs_filtered.p_value')
EX('05_results.md', '처음 규칙(v1)은 14건의 정답 셀을', '99:29', 'mh_rules', '공통 2,871 v33u만:v1만')
EX('05_results.md', '처음 규칙(v1)은 14건의 정답 셀을', 'p=3.8×10⁻¹⁰', 'mh_rules', '공통 2,871 v33u만:v1만 p')
EX('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', 'p=8.8×10⁻⁵', 'lookup_arith_posthoc', 'T2 1위 산술 대 조회 Fisher p')
EX('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', 'p=1', 'lookup_arith_posthoc', 'T3 m=1 조회 대 산술 Fisher p')
EX('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', 'p=.48', 'lookup_arith_posthoc', 'T4b 푸아송 이항 p')
EX('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', '95.8%', 'lookup_arith_posthoc', '최종 산술1 예산20 정확도', 2)
EX('05_results.md', '예산 20 기준 검색 성공률은 정답 셀이 1개인 문항에서', '32', 'lookup_arith_posthoc', 'T4b 관측 1위 적중', 2)
P11 = 'v1 n_scored'      # 2026-09-28: 사전등록 11줄 글 대신 처음 규칙 검색 결과 파일의 채점 수
EX('04_setup.md', '**MultiHiertt 답변 표본.**', '2,871', 'mh_pop', P11, 1)
EX('04_setup.md', '**데이터 사용 이력.**', '2,871', 'mh_pop', P11)
EX('06_discussion.md', '**표본 평가.**', '2,871', 'mh_pop', P11)
# 주장 문장의 검정값(초록·1장·7장·5장·6장)
for _f, _a in [('00a_abstract_ko.md', 'HiTab과 MultiHiertt에서'), ('01_intro.md', '3. **답변 정확도까지'), ('07_conclusion.md', '3. **답변 정확도.**')]:
    EX(_f, _a, 'p=.0052', 'reader_test', 'test cell_vs_chunk p')
    EX(_f, _a, 'p=.86', 'reader_test', 'inter ALL cell만:chunk만 p')
for _f, _a in [('01_intro.md', '3. **답변 정확도까지'), ('07_conclusion.md', '3. **답변 정확도.**')]:
    EX(_f, _a, 'p=.0019', 'stats3b', 'item3b_hitab300 chunk Holm p')
EX('01_intro.md', '3. **답변 정확도까지', 'p=.078', 'stats3b', 'item3b_mh882_pooled fulltable Holm p')
EX('01_intro.md', '2. **셀 단위 검색을 발표된', 'p=.0040', 'rerun_cmp', 'mh_doc s3c만:sleaf만 Holm p')
for _f, _a in [('00a_abstract_ko.md', 'HiTab과 MultiHiertt에서'), ('00b_abstract_en.md', 'On HiTab and MultiHiertt')]:
    EX(_f, _a, 'p=.0019', 'stats3b', 'item3b_hitab300 chunk Holm p')
EX('05_results.md', '| 고정 청크 (최종 규칙) |', 'p=.0052', 'reader_test', 'test cell_vs_chunk p')
EX('05_results.md', '**주 비교: 고정 청크(최종 규칙).**', 'p=.0052', 'reader_test', 'test cell_vs_chunk p')
EX('05_results.md', '**주 비교: 고정 청크(최종 규칙).**', 'p=.0005', 'reader_test', 'test arith_m2+ cell_vs_chunk p')
EX('05_results.md', '| 참고: 검색 없이 문서 표 전체 입력 | .5592', 'p=.078', 'stats3b', 'item3b_mh882_pooled fulltable Holm p')
EX('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', 'p=.039', 'stats3b', 'item3b_mh882_pooled fulltable p')
EX('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', 'p=.0056', 'stats3b', 'item3b_mh882_pooled trag_hetero Holm p')
EX('05_results.md', '| TableRAG(Yu) 청크 | .5545', 'p=.0056', 'stats3b', 'item3b_mh882_pooled trag_hetero Holm p')
EX('05_results.md', '**나머지 비교군.**', 'p=2.1×10⁻⁶', 'v1cmp', 'v1cmp 셀 표현 비교군 9개 중 최대 Holm p')
EX('06_discussion.md', '**표 전체와의 비교.** 검색 없이 표 전체를', 'p=.078', 'stats3b', 'item3b_mh882_pooled fulltable Holm p')
EX('06_discussion.md', '**고정 청크 대비 답변 우위의 집중.**', 'p=.0005', 'reader_test', 'test arith_m2+ cell_vs_chunk p')

EX('05_results.md', '주 모집단인 단일 셀 조회에서', '3:3', HRE, 'chunk.multi_cell s3c만:상대만')
EX('05_results.md', '주 모집단인 단일 셀 조회에서', 'p=1', HRE, 'chunk.multi_cell s3c만:상대만 p')
EX('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', '.078', 'stats3b', 'item3b_mh882_pooled fulltable Holm p')
EX('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', '882', 'reader_test', 'test cell n')
EX('06_discussion.md', '두 방법 모두 정답 셀 전부를 문맥에 포함한 문항에서는 답변 정확도에 차이가 없었다.', '0', 'step4', 'hitab300_s3c_vs_chunk only_other_retrieved ours_correct', 2)
EX('06_discussion.md', '두 방법 모두 정답 셀 전부를 문맥에 포함한 문항에서는 답변 정확도에 차이가 없었다.', '7', 'step4', 'mh882_final_vs_chunk_final only_other_retrieved ours_correct', 2)
EX('06_discussion.md', '두 방법 모두 정답 셀 전부를 문맥에 포함한 문항에서는 답변 정확도에 차이가 없었다.', '7', 'step4', 'mh882_final_vs_chunk_final neither_retrieved ours_correct', 3)
EX('06_discussion.md', '**가설은 지지되지 않았다.**', '20', 'reader_dev', 'dev multihiertt_dev_primary lookup_m1 n', 1)
EX('06_discussion.md', '| | | C: 정답 셀 순위 51 이하 | 20 |', '20', 'bottleneck', 'hitab_538 C_rank51plus')
EX('06_discussion.md', '**재정렬 진단.**', 'p=.0104', 'bottleneck', 'rerank mh_indoc Holm p', 2)
EX('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', '20', 'bottleneck', '판정 (가) 예', 1)
EX('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', '20', 'bottleneck', '판정 (가) 아니오', 2)
EX('09_appendix.md', '332건 기준에서도 β=1.0이 1위다.', '2', 'alpha_dev', 'α 1.0~0.6 n332_correct 최대−최소')
EX('01_intro.md', '3. **답변 정확도까지', '237', 'stats3b', 'item3b_hitab300 fulltable correct', 3)
EX('06_discussion.md', '**표 전체와의 비교.** 검색 없이 표 전체를', '237', HANS, 's3c.answer_correct', 1)
EX('05_results.md', '**참고 조건의 해석.**', '237', HANS, 's3c.answer_correct', 1)
EX('05_results.md', '**참고 조건의 해석.**', '13', 'token_totals', '(정수 부분')
EX('05_results.md', '이 범위에서 본 방법(.9142)은', '2.9×10⁻⁸', 'rerun_cmp', 'hitab_split s3c만:table만 Holm p')
EX('06_discussion.md', '셋째, **색인 단위의 크기**', 'p=.88', HRE, 'chunk.arithmetic s3c만:상대만 p')
EX('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', '57', 'fix0928', 'chunk 처음 대 최종 context_differs')
EX('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', '12', 'fix0928', 'chunk 처음 대 최종 retrieval_differs')
EX('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', '6', 'fix0928', 'chunk 처음 대 최종 retrieval_final_only')
EX('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', '7', 'fix0928', '판정 (나) 자동 검증 아니오 (기간 있는 예)')
EX('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', '20', 'bottleneck', '판정 (가) 예', 3)
EX('06_discussion.md', '**정답 표를 못 찾은 40건의 판정.**', '20', 'fix0928', '판정 (나) 자동 검증 아니오 ((가)=예 전체)', 4)
EX('06_discussion.md', '고정 청크(최종 규칙)의 검색 성공 687건은', '687', 'fix0928', 'chunk 처음 대 최종 retrieval_success_final')
EX('07_conclusion.md', '3. **답변 정확도.**', '195', 'fix0928', '산술·셀 2개+ 둘 다 검색 성공(최종 대 최종) n')
EX('07_conclusion.md', '3. **답변 정확도.**', '.2667', 'fix0928', '산술·셀 2개+ 둘 다 검색 성공 cell_acc')
EX('07_conclusion.md', '3. **답변 정확도.**', '.2359', 'fix0928', '산술·셀 2개+ 둘 다 검색 성공 chunk_acc')

# 원천을 못 찾은 수치의 사유(자동 대조 실패 시 적는다)
NOTFOUND = {}
# 2026-09-28 A: 결과 파일에 없는 입력 상수를 지운 뒤 원천이 없어진 수치(글 기록·우연히 같은 값으로 다시 맞추지 않는다)
# 다른 논문이 보고한 값 — 결과 파일과 대조하지 않고 '외부 문헌' 출처로 적는다 (2026-09-28)
EXTERNAL = {}
_Z = '외부 문헌: Zhao et al. (2022) MultiHiertt, Table 2 (arXiv:2206.01347)'
_Q = '외부 문헌: Qwen2.5 Technical Report (arXiv:2412.15115) 본문 "other models to process up to 131,072 tokens"(YaRN·DCA), Hugging Face Qwen/Qwen2.5-7B-Instruct 모델 카드 "Full 131,072 tokens" (2026-09-28 확인)'
EXTERNAL['05_results.md', AT('05_results.md', '**참고 조건의 해석.**'), '131,072'] = _Q
for _t in ('100', '4', '84.9%'):
    EXTERNAL['06_discussion.md', AT('06_discussion.md', '**정답 근거 주석의 범위.**'), _t] = _Z

# =============================================================== 2. kind (자동 규칙 + ===== MANUAL: KO)
K_RES, K_N, K_ST, K_SEC, K_CITE, K_SET, K_ETC = ('결과수치', '표본·개수', '통계(p·불일치쌍·CI)', '절·표·그림 번호',
                                                   '연도·인용', '설정값', '기타')
MODEL = re.compile(r'(Qwen|bge|NF4|bfloat|WSL|float64)')

# (파일, 줄) 또는 (파일, 줄, 토큰) -> kind
KIND_OVERRIDE = {}


def KO(f, lines, kind, toks=None):
    """toks 원소는 토큰 문자열 또는 (토큰, 몇 번째) — 같은 줄에 같은 토큰이 여러 번이면 몇 번째인지로 가른다."""
    for ln in ([lines] if isinstance(lines, int) else lines):
        if toks is None:
            KIND_OVERRIDE[f, ln] = kind
        else:
            for t in toks:
                KIND_OVERRIDE[(f, ln) + (tuple(t) if isinstance(t, tuple) else (t,))] = kind


def auto_kind(f, line, typ, s, a, b):
    before, after = line[max(0, a - 14):a], line[b:b + 8]
    if f.startswith('08_refs'):
        return K_CITE
    if typ in ('p', 'cmp', 'ratio'):
        return K_ST
    if typ == 'sha':
        return K_RES
    if typ in ('date', 'fn'):          # fn = 각주 표시 [^N]
        return K_ETC
    if typ == 'arxiv':
        return K_CITE
    if typ == 'tab':
        return K_SEC
    if typ == 'ver':
        return K_SEC if after.startswith('절') or line.startswith('#') else K_SET
    if typ == 'ident':
        return K_SET if MODEL.match(s) else K_ETC
    stripped = line.lstrip('#').lstrip()
    if line.startswith('#') and line.find(s) == len(line) - len(stripped):
        return K_SEC
    if before.endswith('제') or re.match(r'(절|장)', after):
        return K_SEC
    if a == 0 and re.match(r'\d+\. ', line):
        return K_ETC
    if re.fullmatch(r'(19|20)\d\d', s) and not re.match(r'(건|개)', after):
        return K_CITE
    if s == '95%' and re.match(r'\s?(CI|신뢰|범위)', after):
        return K_SET
    if '[' in line[:a] and line.rfind('[', 0, a) > line.rfind(']', 0, a):
        return K_ST
    if s.endswith('%p'):
        return K_RES
    if re.search(r'(seed |α=|β=|K=|B=)$', before):
        return K_SET
    if s in ('1', '2') and re.search(r'셀 ?$', before) and after.startswith('개'):
        return K_ETC                                   # 그룹 이름 "셀 1개", "셀 2개 이상"
    if s == '1' and after.startswith('위'):
        return K_ETC                                   # 순위 "1위"
    if s == '20' and ('예산' in line[max(0, a - 30):b + 10] or re.match(r'(셀|개|에 이르)', after) or '상위' in before):
        return K_SET
    if re.match(r'(토큰)', after):
        return K_SET if s in ('512', '384', '64') else K_RES
    if re.search(r'(count=|\()$', before) and re.fullmatch(r'[\d,]+', s):
        return K_N
    if re.match(r'(토큰)', after):
        return K_SET if s in ('512', '384', '64') else K_RES
    if re.match(r'(시간)', after):
        return K_SET
    if re.match(r'(건|개|문장|문서|표|행|셀|질의|회|그룹|칸)', after) or re.search(r'(query count=)$', before):
        return K_N
    if s.endswith('%'):
        return K_RES
    if s.startswith(('.', '+.', '−.')):
        return K_RES
    return K_RES


# ------------------------------------------------ KIND_OVERRIDE 수동 목록
KO('01_intro.md', 7, K_ETC)                               # 예시 값 52.1
KO('01_intro.md', 11, K_SET)                              # 8GB
KO('01_intro.md', 27, K_ETC, ['1'])
KO('01_intro.md', 28, K_ETC, ['2'])
KO('01_intro.md', 29, K_ETC, ['3'])
KO('01_intro.md', 30, K_ETC, ['4'])
KO('01_intro.md', 31, K_ETC, ['5'])
KO('01_intro.md', 37, K_SEC)
KO('02_related.md', [7, 13, 15, 17, 31], K_CITE)           # 선행 논문이 보고한 값
KO('02_related.md', 31, K_SEC, ['4.5'])
KO('02_related.md', 13, K_SEC, ['4.2'])
KO('02_related.md', 25, K_SEC)
KO('02_related.md', 47, K_SET, ['20'])
KO('03_method.md', [17, 53, 57, 58, 59, 60, 61, 62, 63, 65], K_ETC)   # 표 3-1 예시 텍스트의 값
KO('03_method.md', [17, 53], K_SEC, ['3-1'])
KO('03_method.md', 35, K_ETC, ['2015'])
KO('03_method.md', 39, K_SET, ['3'])
KO('03_method.md', 43, K_ETC, ['2', '3'])
KO('03_method.md', 43, K_N, ['0'])
KO('03_method.md', [75, 77], K_SET)
KO('03_method.md', [87, 88], K_SET)
KO('03_method.md', 87, K_CITE, ['2024'])
KO('03_method.md', 88, K_CITE, ['2025'])
KO('03_method.md', 90, K_N, ['60'])
KO('04_setup.md', [12, 16, 17, 21, 55], K_N)
KO('04_setup.md', 13, K_ETC, ['1'])
KO('04_setup.md', [16, 17, 21, 55], K_ETC, ['1', '2'])
KO('00b_abstract_en.md', 5, K_SET, ['20'])
KO('04_setup.md', 13, K_N, ['4.06'])
KO('04_setup.md', [16, 17], K_ETC, ['4'])
KO('04_setup.md', 17, K_SET, ['300'])
KO('04_setup.md', 19, K_ETC, ['1개', '2개'])
KO('04_setup.md', 21, K_ETC, ['1', '2'])
KO('04_setup.md', 21, K_RES, ['100%'])
KO('04_setup.md', 23, K_ETC, ['346930.6', '346931'])
KO('04_setup.md', 23, K_RES, ['6'])
KO('04_setup.md', [35, 36], K_SET)
KO('04_setup.md', [36, 37, 38, 39, 40], K_CITE, ['2024', '2025'])
KO('04_setup.md', 29, K_SET, ['20'])
KO('04_setup.md', 47, K_CITE, ['1,000', '200'])
KO('04_setup.md', 55, K_SET, ['300'])
KO('04_setup.md', 59, K_SET, ['384'])
KO('04_setup.md', 59, K_N, ['60', '23'])
KO('04_setup.md', 59, K_RES, ['169.8'])
KO('04_setup.md', 63, K_SET, ['42'])
KO('04_setup.md', 63, K_ETC, ['7'])
KO('04_setup.md', 65, K_SET, ['20260913', '300'])
KO('04_setup.md', 67, K_SET)
KO('04_setup.md', 71, K_SET)
KO('04_setup.md', 73, K_SET, ['5'])
KO('04_setup.md', 73, K_N, ['120'])
KO('04_setup.md', 73, K_RES, ['10.1', '3.08'])
KO('05_results.md', [13, 30, 54, 85, 105, 117, 131], K_SET, ['20'])
KO('05_results.md', 46, K_SET, ['20'])
KO('05_results.md', [52, 129], K_SET, ['4', '384'])
KO('05_results.md', 69, K_ETC, ['7'])
KO('05_results.md', 69, K_SET, ['.05'])
KO('05_results.md', 75, K_RES, ['0'])
KO('05_results.md', 77, K_SET, ['19'])
KO('05_results.md', 77, K_RES, ['645'])
KO('05_results.md', 79, K_ETC, ['6'])
KO('05_results.md', 99, K_ETC, ['1'])
KO('05_results.md', 99, K_N, ['24'])
KO('05_results.md', 125, K_N, ['8'])
KO('05_results.md', [146, 148, 168], K_ETC, ['0'])
KO('05_results.md', 146, K_SET, ['95%'])
KO('05_results.md', 164, K_ETC, ['0'])
KO('05_results.md', 166, K_ETC, ['2015', '222', '1,767', '813', '732'])
KO('05_results.md', 172, K_ETC)
KO('05_results.md', 180, K_ETC, ['1'])
KO('05_results.md', 180, K_SET, ['20'])
KO('05_results.md', 182, K_RES, ['2.09', '1.34', '0.67', '2.69'])
KO('05_results.md', 182, K_SET, ['95%'])
KO('05_results.md', 182, K_N, ['85', '211', '46', '71', '126'])
KO('06_discussion.md', 11, K_SET, ['20'])
KO('06_discussion.md', 17, K_SET, ['19'])
KO('06_discussion.md', 19, K_SET, ['384'])
KO('06_discussion.md', 29, K_SET, ['4'])
KO('07_conclusion.md', [7, 8, 9, 10], K_ETC, ['1', '2', '3', '4'])
KO('07_conclusion.md', 3, K_SET, ['20'])
KO('09_appendix.md', 5, K_ETC, ['7'])
KO('09_appendix.md', 5, K_SET, ['4'])
KO('09_appendix.md', [11, 12, 13, 14, 15, 16, 17], K_RES)
KO('09_appendix.md', [11, 15, 16, 17], K_ST, ['.568', '.327', '.050', '.824'])
KO('09_appendix.md', 23, K_SET, ['4'])
KO('09_appendix.md', [33, 34], K_N)
KO('09_appendix.md', 36, K_SET, ['5', 'p≥.05'])
KO('09_appendix.md', 36, K_ETC, ['1'])
KO('09_appendix.md', 19, K_N, ['88', '95', '76', '77', '43', '52', '2', '22'])
KO('09_appendix.md', 34, K_ST, ['1.0'])
KO('09_appendix.md', 65, K_SET, ['1', '5', '10', '20'])


# 위 KO 목록의 줄 번호는 초안 스냅숏(커밋 ed5e128)의 thesis/src 기준이다. 지금 원고의 줄로 옮긴다
# (바뀌지 않은 줄, 또는 같은 길이의 덩어리로 고친 줄만. 지운 줄은 버린다). 새로 생긴 줄은 아래 KOA 로 앵커 지정.
BASE = 'ed5e128'


@lru_cache(None)
def _linemap(f):
    import difflib, subprocess
    old = subprocess.run(['git', 'show', f'{BASE}:rag-agent/thesis/src/{f}'], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.split('\n')
    m = {}
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, old, _lines(f), autojunk=False).get_opcodes():
        if tag == 'equal' or (tag == 'replace' and i2 - i1 == j2 - j1):
            m.update({i1 + k + 1: j1 + k + 1 for k in range(i2 - i1)})
    return m


KIND_OVERRIDE = {(k[0], _linemap(k[0])[k[1]]) + k[2:]: v for k, v in KIND_OVERRIDE.items() if k[1] in _linemap(k[0])}


def KOA(f, anchor, kind, toks=None):
    KO(f, AT(f, anchor), kind, toks)


KOA('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', K_SEC, ['5.4'])
KOA('04_setup.md', '**MultiHiertt.** MultiHiertt의 공개 test 파일', K_N, [('2', 1)])      # 정답 셀이 빈 칸으로 파싱된 2건
KOA('04_setup.md', '**TableRAG 셀 검색 재구현의 범위.**', K_ETC, ['1', '2', ('3', 2)])   # (1)(2)(3) 항목 번호
KOA('04_setup.md', '**TableRAG 셀 검색 재구현의 범위.**', K_SET, [('3', 1)])             # 자주 나오는 값 3개
KOA('05_results.md', '괄호 안은 검색에 성공한 질의 수다.', K_SET, ['8'])                    # Holm 묶음 크기
KOA('05_results.md', '**나머지 비교군.**', K_SET, ['12', '9'])
KOA('05_results.md', 'MultiHiertt가 함께 제공하는 셀 문장(table_description', K_ETC, ['1'])   # 1회 실행
KOA('01_intro.md', '본 연구가 MT2Net과 다른 점은 네 가지다.', K_ETC, ['1', '2', '3', '4'])       # (1)~(4) 항목 번호
KOA('02_related.md', '**MT2Net(Zhao et al., 2022).**', K_ETC, ['1', '2', '3', '4'])
KOA('01_intro.md', '2. **셀 단위 검색을 발표된', K_SET, ['1,000'])                         # 고정 청크 설정
KOA('06_discussion.md', '**머리글 규칙 수정 이력.**', K_SET, ['1,000'])                        # 고정 청크 설정(1,000자)
KOA('05_results.md', '**TableRAG 셀 검색 재구현이 낮은 이유.**', K_ETC, ['1'])                 # "상한이 1보다 낮다"의 비교 기준
KOA('02_related.md', '**MixRAG.** Zhang et al.(2026)은', K_CITE)                          # 선행 논문이 보고한 값
KOA('02_related.md', '| MT2Net (Zhao et al., 2022) |', K_CITE)
KOA('09_appendix.md', 'RowCol·RandRow·TableRAG 셀 검색 재구현의 882건 답변은', K_SET, ['12'])
KOA('06_discussion.md', '**정답 근거 주석의 범위.**', K_CITE, ['2'])
KOA('09_appendix.md', '키워드가 있는 91문항은 모두', K_ETC, ['12'])                          # 질문 인용 "less than 12 months"                     # 인용 위치 Table 2
# 2026-09-28 인용 대조 반영: 선행 논문이 보고한 값·인용 쪽 번호(원고 줄 전체)
KOA('01_intro.md', '표 RAG에서 흔한 방식은', K_CITE)
KOA('02_related.md', '**HiTab.** Cheng et al.(2022)은', K_CITE)
KOA('02_related.md', '검색된 문맥에 질문과 관련은 있으나', K_CITE)
KOA('02_related.md', '**유형별 보고.**', K_CITE)
for _a in ('| HiTab 채점기 |', '| TableRAG(Chen) | 스키마 문서 내용 |', '| TableRAG(Chen) | 스키마 검색 |', '| TableRAG(Chen) | 행 라벨 문서 |',
           '| TableRAG(Chen) | 셀 인코딩 예산 B |', '| TableRAG(Chen) | 질의 확장과 검색 개수 K |',   # 숫자 열 판정 행은 2026-09-28 결과값이 들어가 대조한다
           '| RowCol | 색인 텍스트 |', '| RandRow | 색인·검색 |', '| RandRow | 리더 입력 |', '| TableRAG(Yu) | 청크 크기·겹침 단위 |',
           '| TableRAG(Yu) | 마크다운 변환 |', '| TableRAG(Yu) | 머리말 이름 |'):
    KOA('09_appendix.md', _a, K_CITE)                                                   # 부록 H: 원 논문·코드 설정값
KOA('06_discussion.md', '**가설은 지지되지 않았다.**', K_ETC, ['1'])                      # 정답 셀이 1개인
KOA('06_discussion.md', '| HiTab 질문의 표 안 | 36 |', K_SET, ['21', '50'])               # 버킷 정의
KOA('06_discussion.md', '| | | B: 정답 셀 순위 21~50 | 25 |', K_SET, ['21', '50'])
KOA('06_discussion.md', '| | | H: 정답 표의 셀은 있으나', K_ETC, ['0'])
KOA('06_discussion.md', 'MultiHiertt 실패 394건 중', K_SET, ['50'])
KOA('06_discussion.md', '**재정렬 진단.**', K_SET, ['50'])
KOA('09_appendix.md', 'β 선택에 쓴 MultiHiertt dev(validation) 911건은', K_SET, ['1.0', '0.1', '0.9', '11'])
KOA('05_results.md', 'sleaf를 뺀 비교군 6개의 24칸은 모두', K_SET, ['6'])
KOA('05_results.md', 'sleaf를 뺀 비교군 6개와의 24개', K_SET, ['6'])
KOA('05_results.md', '그룹별 값은 탐색적 결과다. Holm p는', K_SET, ['14'])
KOA('05_results.md', '**주 비교: 고정 청크(최종 규칙).**', K_SEC, ['9'])
KOA('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', K_SET, ['14'])
KOA('05_results.md', '**표 전체와 TableRAG(Yu) 청크.**', K_ETC, ['0'])
KOA('06_discussion.md', '처음 규칙의 교집합 결과(38:76)를 보고', K_SET, ['20'])
KOA('06_discussion.md', '| | | C: 정답 셀 순위 51 이하 | 16 |', K_SET, ['51'])
KOA('06_discussion.md', '| | | C: 정답 셀 순위 51 이하 | 20 |', K_SET, ['51'])   # α 후보 범위
KOA('09_appendix.md', '332건 기준에서도 β=1.0이 1위다.', K_SET, ['1.0', '0.6'])
KOA('05_results.md', 'b는 본 방법만 맞힌 문항 수', K_SET, ['8'])                    # Holm 묶음 크기
KOA('05_results.md', '이 범위에서 본 방법(.9142)은', K_SET, ['7'])                   # 비교군 수
KOA('09_appendix.md', '"머리글 규칙 영향"은 두 규칙에서', K_SET, ['3', '12'])         # 규칙 불일치 행 수, Holm 묶음 크기
KOA('04_setup.md', '**HiTab 답변 표본.**', K_ETC, ['8'])                            # 검색 방법 수
KOA('03_method.md', "> In the table 'agri-food industry sub-groups … 2011'", K_ETC)   # 표 3-1 예시 텍스트
KOA('03_method.md', '> food service / french-language workers:', K_ETC)
KOA('07_conclusion.md', '3. **답변 정확도.**', K_SET, ['4'])                       # 4비트
for _i, _line in enumerate(_lines('06_discussion.md'), 1):
    if _line.startswith('| | | D: '):
        KO('06_discussion.md', _i, K_SET, ['20'])                                   # 버킷 정의


def kind_of(f, ln, line, typ, s, a, b, occ=1):
    k = KIND_OVERRIDE.get((f, ln, s, occ)) or KIND_OVERRIDE.get((f, ln, s))
    if k:
        return k
    if typ == 'num' and KIND_OVERRIDE.get((f, ln)):       # 줄 단위 지정은 일반 수에만
        return KIND_OVERRIDE[f, ln]
    return auto_kind(f, line, typ, s, a, b)


# =============================================================== 5. matching
def close(x, c, d):
    if d == 0 and abs(c - round(c)) > 1e-9 and abs(c) < 100:
        return False              # 정수로 쓴 수는 정수 원천값과만(평균 토큰 수처럼 100 이상인 평균값은 반올림 허용)
    return abs(x - c) <= 0.5 * 10 ** -d + 1e-12


def p_ok(q, c):
    op, v = q['op'], q['v']
    if op == '<': return c < v
    if op == '>': return c > v
    if op == '≤': return c <= v
    if op == '≥': return c >= v
    if q['sig']:
        return c > 0 and abs(float(f'{c:.{q["sig"] - 1}e}') - v) <= 1e-9 * v
    if q['dec'] is not None:
        return close(v, c, q['dec'])
    return abs(c - v) <= 1e-12


def fmt(c):
    if c.typ == 'r':
        return f'{c.value[0]}:{c.value[1]}'
    v = c.value
    if isinstance(v, float):
        if v != 0 and abs(v) < 1e-3:
            return f'{v:.2e}'
        return f'{v:.4f}'.rstrip('0').rstrip('.') if abs(v) < 100 else f'{v:.1f}'
    return str(v)


def match(s, cands):
    if re.fullmatch(r'[0-9a-f]{40}|[0-9a-f]{64}', s):          # sha256·커밋 해시
        return next(((c, '') for c in cands if c.typ == 'h' and c.value == s), (None, ''))
    q = parse(s)
    for c in cands:
        if q['k'] == 'ratio':
            if c.typ == 'r' and tuple(c.value) == q['v']:
                return c, ''
            if c.typ == 'r' and tuple(c.value) == q['v'][::-1]:
                return c, ' (순서 반대로 기록된 쌍)'
        elif q['k'] == 'p':
            if c.typ == 'p' and p_ok(q, c.value):
                return c, ''
        else:
            x = q['v'] / 100 if q['k'] in ('pct', 'pp') else q['v']
            d = q['dec'] + 2 if q['k'] in ('pct', 'pp') else q['dec']
            if c.typ == 'v' and close(x, c.value, d):
                return c, ''
    if q['k'] == 'num':                        # 표 칸의 p값(접두사 없이 쓴 것)
        for c in cands:
            if c.typ == 'p' and close(q['v'], c.value, q['dec']):
                return c, ''
    return None, ''


MATCHABLE = {K_RES, K_N, K_ST}
RESULT_EXT = ('.json', '.jsonl', '.csv')            # 코드가 만든 결과 파일. 그 밖(.md 등)은 글로 된 기록


def _pref_score(c, lp, cp):
    hit = lambda pats: any(re.search(x, c.field) for x in pats)
    return (0 if lp and hit(lp) else 1) + (0 if cp and hit(cp) else 1)


def main():
    rows = []
    for fpath in sorted(glob.glob(P('thesis', 'src', '*.md'))):
        f = os.path.basename(fpath)
        for ln, line in enumerate(open(fpath, encoding='utf-8').read().split('\n'), 1):
            names = LS.get((f, ln), [])
            base = [c for n in names for c in SETS[n]()]
            lp = PREFER.get((f, ln))
            seen = Counter()
            for typ, s, a, b in tokens(line):
                seen[s] += 1
                occ = seen[s]
                col = line[:a].count('|') if line.lstrip().startswith('|') else 0
                cp = COLPREF.get((f, ln), {}).get(col)
                cands = sorted(base, key=lambda c: _pref_score(c, lp, cp)) if (lp or cp) else base
                kind = kind_of(f, ln, line, typ, s, a, b, occ)
                rng = re.match(r'[~–][+−]?[\d.,]+(%p|%)', line[b:]) if typ == 'num' else None
                s_eval = s + rng.group(1) if rng else s      # "1.2~1.4%p" 의 1.2 → 1.2%p 로 대조
                row = dict(file=f, line=ln, number=s, context=context(line, a, b), kind=kind,
                           source_file='', source_field_or_value='', match='해당 없음', sleaf_as_ours='', text_only='')
                if (f, ln, s) in EXTERNAL:
                    row.update(source_file=EXTERNAL[f, ln, s], match='외부 문헌')
                elif kind in MATCHABLE:
                    exp = EXPECT.get((f, ln, s, occ)) or EXPECT.get((f, ln, s))
                    nf = NOTFOUND.get((f, ln, s)) or NOTFOUND.get((f, ln, '*'))
                    top = [x for x in cands if _pref_score(x, lp, cp) == _pref_score(cands[0], lp, cp)] if (lp or cp) and cands else []
                    c, note = (None, '') if nf else ((match(s_eval, top) if top else (None, '')) if (top and match(s_eval, top)[0]) else match(s_eval, cands))
                    if exp:
                        pat = exp[1]
                        ec = next(x for x in SETS[exp[0]]() if (re.search(pat[3:], x.field) if pat.startswith('re:') else pat in x.field))
                        ok, _ = match(s_eval, [ec])
                        other = (f' ※ {c.src} [{c.field}] 은 쓰인 값과 같음'
                                 if (c and not ok and c.src.endswith('.md')) else '')     # 문서 본문이 논문 값과 같을 때만 덧붙임
                        row.update(source_file=ec.src, source_field_or_value=f'{ec.field} = {fmt(ec)}{other}',
                                   match='일치' if ok else f'불일치(원천 값 {fmt(ec)})')
                        row['sleaf_as_ours'] = {'sleaf': '예', 'ours': '아니오'}.get(ec.tag, '')
                    elif c:
                        if not c.src.endswith(RESULT_EXT):     # 글 기록에서 찾은 값: 이 줄의 원천 집합 안 결과 파일에서 같은 값이 나오면 그쪽이 출처
                            c, note = next((m for m in [match(s_eval, [x for x in cands if x.src.endswith(RESULT_EXT)])] if m[0]), (c, note))
                        row.update(source_file=c.src, source_field_or_value=f'{c.field} = {fmt(c)}{note}', match='일치')
                        row['sleaf_as_ours'] = {'sleaf': '예', 'ours': '아니오'}.get(c.tag, '')
                    else:
                        row.update(match='출처 못 찾음', source_field_or_value=nf or '')
                    if row['match'] == '일치' and not row['source_file'].endswith(RESULT_EXT):
                        row['text_only'] = '글 기록에만 있음'
                    if row['sleaf_as_ours'] == '예' and (f, ln) in SLEAF_NAMED:
                        row['sleaf_as_ours'] = '아니오(sleaf로 표기)'
                rows.append(row)
    return rows


def write(rows):
    cols = ['file', 'line', 'number', 'context', 'kind', 'source_file', 'source_field_or_value', 'match', 'sleaf_as_ours', 'text_only']
    with open(os.path.join(HERE, 'numbers.csv'), 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    mk = lambda r: '불일치' if r['match'].startswith('불일치') else r['match']
    summary = dict(
        n_rows=len(rows),
        by_kind=dict(Counter(r['kind'] for r in rows).most_common()),
        by_match=dict(Counter(mk(r) for r in rows).most_common()),
        by_kind_match={k: dict(Counter(mk(r) for r in rows if r['kind'] == k)) for k in sorted({r['kind'] for r in rows})},
        mismatch=[f"{r['file']}:{r['line']}:{r['number']} → {r['match']} [{r['source_file']}]" for r in rows if mk(r) == '불일치'],
        not_found=[f"{r['file']}:{r['line']}:{r['number']} ({r['source_field_or_value']})" for r in rows if r['match'] == '출처 못 찾음'],
        sleaf_as_ours_yes=[f"{r['file']}:{r['line']}:{r['number']}" for r in rows if r['sleaf_as_ours'] == '예'],
        text_sourced=dict(
            n=sum(1 for r in rows if r['text_only']),
            by_reason=dict(Counter(r['text_only'].split(':')[0] for r in rows if r['text_only'])),
            rows=[f"{r['file']}:{r['line']}:{r['number']} [{r['source_file']}] {r['text_only'].split(':')[0]}" for r in rows if r['text_only']]),
    )
    with open(os.path.join(HERE, 'summary.json'), 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    return summary


if __name__ == '__main__':
    if os.environ.get('DUMP_TOKENS'):
        for fp in sorted(glob.glob(P('thesis', 'src', '*.md'))):
            for i, line in enumerate(open(fp, encoding='utf-8').read().split('\n'), 1):
                for typ, s, a, b in tokens(line):
                    print(f'{os.path.basename(fp)}:{i}\t{typ}\t{s}\t{context(line, a, b, 40)}')
    else:
        rows = main()
        s = write(rows)
        print(json.dumps({k: s[k] for k in ('n_rows', 'by_kind', 'by_match')}, ensure_ascii=False, indent=1))
