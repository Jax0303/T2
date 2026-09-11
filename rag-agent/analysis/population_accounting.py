"""Paired accounting on a single explicit population; no causal ceiling claims."""
from collections import Counter

def _by_id(rows):
    out={r["query_id"]:r for r in rows}
    if len(out)!=len(rows):
        raise ValueError("duplicate query IDs")
    return out

def account(records, retrieved, gold, keep):
    records=[r for r in records if "correct" in r]
    R,A,G=_by_id(records),_by_id(retrieved),_by_id(gold)
    if R.keys()!=A.keys() or R.keys()!=G.keys():
        raise ValueError("leg populations differ")
    selected=[r for r in records if keep(r)]
    n=len(selected)
    if not n:
        raise ValueError("empty population")
    hit=[r["query_id"] for r in selected if r["correct"]]
    miss=[r["query_id"] for r in selected if not r["correct"]]
    ids=hit+miss
    for q in ids:
        if A[q]["retrieval_correct"]!=R[q]["correct"] or A[q]["answer"]!=R[q]["answer"] or G[q]["answer"]!=R[q]["answer"]:
            raise ValueError(f"leg metadata differs: {q}")
    def count(qs,leg):return sum(int(leg[q]["answer_correct"]) for q in qs)
    ah,am,gh,gm=count(hit,A),count(miss,A),count(hit,G),count(miss,G)
    buckets=Counter()
    for q in ids:
        a,g=A[q]["answer_correct"],G[q]["answer_correct"]
        if a and g: buckets["both_correct"]+=1
        elif a: buckets["gold_only_fail"]+=1
        elif not R[q]["correct"]: buckets["retrieval_miss"]+=1
        elif not g: buckets["reader_ceiling"]+=1
        else: buckets["distractor"]+=1
    result={"n":n,"n_hit":len(hit),"n_miss":len(miss),
            "retrieved_correct":ah+am,"gold_correct":gh+gm,
            "hit_answer_correct":ah,"miss_answer_correct":am,
            "hit_gold_correct":gh,"miss_gold_correct":gm,
            "retrieval_accuracy":len(hit)/n,"answer_em":(ah+am)/n,"gold_em":(gh+gm)/n,
            "conditional_hit_em":ah/len(hit) if hit else None,
            "conditional_miss_em":am/len(miss) if miss else None,
            "hit_paired_gap":(gh-ah)/n,"miss_paired_gap":(gm-am)/n,
            "both_wrong":sum(not A[q]["answer_correct"] and not G[q]["answer_correct"] for q in ids),
            "buckets":dict(buckets)}
    assert gh+gm-ah-am == (gh-ah)+(gm-am)
    assert sum(buckets.values()) == n
    return result
