# 오류 원인 자동 분류 초안 — PREREG-2026-10-22-reader-sweep.md §3. 2026-10-22 T2-1 스크래치에서 옮김. 882 cot 레코드(answer_correct/pred/answer/raw 필드)로 시험. 마지막 줄 채점(fin_inc) 레코드 형식에 맞게 필드 이름을 바꿔 써야 한다.
import json, sys, re, math, collections, random
sys.path.insert(0, '.')
from rag_agent.eval.multihiertt_em import str_to_num, mh_exact_match

def num(s):
    try: return float(str(s).replace('$','').replace(',','').replace('%',''))
    except: return None

def classify(r):
    gold = r['answer']; pred = r['pred']; prog = (r['kind']=='arith')
    if r['marker_found'] in ('False', False): return 'A 표시 없음'
    if not str(pred).strip(): return 'B 빈 답'
    if re.search(r'cannot|not (provided|available|possible|enough)|insufficient|unknown|n/a', str(pred), re.I): return 'C 답 거부(정보 없음 서술)'
    g = num(gold); p = num(pred)
    if prog:
        if g is not None and g < 0 and p is not None and math.isclose(abs(g), p, rel_tol=1e-6): return 'D 채점기 음수 규칙(절댓값 일치)'
        if p is None: return 'E 숫자 아닌 답(산술)'
        if g is None: return 'E2 정답 숫자 아님'
        if g != 0 and math.isclose(abs(g), abs(p), rel_tol=1e-6): return 'D2 부호만 다름'
        if r['answer_correct_docmath'] in ('1', 1, True): return 'F 척도·반올림 (DocMath 채점은 정답)'
        if g != 0 and abs(p - g)/abs(g) < 0.02: return 'G 2% 이내 (반올림)'
        if g != 0 and p != 0:
            ratio = abs(p/g)
            for k in (10,100,1000,10000,100000,1e6):
                if math.isclose(ratio, k, rel_tol=1e-3) or math.isclose(ratio, 1/k, rel_tol=1e-3): return 'H 10^n 척도 차이'
        return 'Z 계산·셀 선택 오류(잔여)'
    else:
        gl = gold if isinstance(gold, list) else [gold]
        if len(gl) > 1: return 'I 다중 정답(span 여러 개)'
        if p is not None and g is not None:
            if math.isclose(p, g, rel_tol=1e-6): return 'J 숫자 같음, 문자열 불일치(형식)'
            if g != 0 and abs(p-g)/abs(g) < 0.02: return 'G 2% 이내 (반올림)'
            return 'Z 다른 셀 값(잔여)'
        if p is None and g is not None: return 'K 정답 숫자인데 답은 텍스트'
        if p is not None and g is None: return 'L 정답 텍스트인데 답은 숫자'
        return 'Z 텍스트 불일치(잔여)'

def run(path, title):
    rows=[json.loads(l) for l in open(path)]
    hit=[r for r in rows if str(r['retrieval_correct'])=='1']
    wrong=[r for r in hit if str(r['answer_correct'])!='1']
    print(f"\n### {title}: 전체 {len(rows)}, 검색 성공 {len(hit)}, 그중 오답 {len(wrong)}")
    by=collections.defaultdict(collections.Counter)
    for r in wrong: by[r['layer']][classify(r)]+=1
    cats=sorted({c for v in by.values() for c in v})
    layers=['lookup_m1','lookup_m2+','arith_m1','arith_m2+']
    print('| 분류 | '+' | '.join(layers)+' | 합 |'); print('|---|'+'---:|'*(len(layers)+1))
    for c in cats:
        vals=[by[l][c] for l in layers]; print(f"| {c} | "+' | '.join(map(str,vals))+f" | {sum(vals)} |")
    print('| 합 | '+' | '.join(str(sum(by[l].values())) for l in layers)+f" | {len(wrong)} |")
    return wrong

random.seed(0)
w1=run('results/mh_arms/cap300_20260924/cell_uniq.json'.replace('.json','.jsonl'),'본 방법 k=20 (cell_uniq)')
w2=run('results/mh_arms/cap300_20260924/fulltable.jsonl','표 전체 입력 (fulltable)')
print("\n### 잔여 오류 예시 (본 방법, 무작위 12)")
res=[r for r in w1 if classify(r).startswith('Z')]
for r in random.sample(res, min(12,len(res))):
    print(f"- [{r['layer']}] Q: {r['question'][:110]}\n  gold={r['answer']!r} pred={r['pred']!r}\n  raw끝: {r['raw'][-220:].replace(chr(10),' / ')}")
