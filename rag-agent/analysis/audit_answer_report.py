#!/usr/bin/env python3
"""Build the final audit report from validated, complete fixed-context legs."""
import json, hashlib, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from analysis.result_sources import current_answer_rows, retrieval_path, read_jsonl, ORIGINAL, ANSWERS, INVALIDATED
from analysis.population_accounting import account
from analysis.answer_stats import mcnemar_exact, paired_bootstrap, holm
from rag_agent.eval.metrics import hitab_exact_match_text
from scripts.answer_accuracy import summarize
from analysis.accuracy_tables import leg_file

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def primary(r):return r["mode"]=="all" and r["m"]==1 and (r.get("aggregation") or "none")=="none"
def main():
    out=ANSWERS
    records=read_jsonl(retrieval_path("t_s3c_hybrid","_records.jsonl"))
    ret=read_jsonl(leg_file("retrieved","jsonl"))
    gold=read_jsonl(leg_file("gold","jsonl"))
    populations={"primary":primary,"data":lambda r:r["mode"]=="all","header":lambda r:r["mode"]=="any"}
    accounting={name:account(records,ret,gold,keep) for name,keep in populations.items()}
    (out/"POPULATION_ACCOUNTING.json").write_text(json.dumps(accounting,indent=2))
    ids=sorted(r["query_id"] for r in records if "correct" in r and primary(r))
    baseline={r["query_id"]:r for r in ret}
    result={}; ps={}
    for tag in ("t_rowcol_values","t_tablerag_leaf","t_tablerag_path"):
        rows=current_answer_rows(tag)
        if not rows:
            result[tag]={"complete":False}
            continue
        rmap={r["query_id"]:r for r in rows}
        assert sum(int(hitab_exact_match_text(r["pred"],r["answer"])) != r["answer_correct"] for r in rows)==0
        old={r["query_id"]:r for r in read_jsonl(ORIGINAL/(tag+"_answer_retrieved.jsonl"))}
        a=[baseline[q]["answer_correct"] for q in ids]
        b=[rmap[q]["answer_correct"] for q in ids]
        o=[old[q]["answer_correct"] for q in ids]
        loss,gain,p=mcnemar_exact(a,b)
        oldloss,oldgain,oldp=mcnemar_exact(o,b)
        comparison={"complete":True,"n":len(ids),"correct":sum(b),"em":sum(b)/len(ids),
                    "old_invalidated_correct":sum(o),"old_invalidated_em":sum(o)/len(ids),
                    "baseline_correct":sum(a),"vs_s3c":{"gain":gain,"loss":loss,"p":p,**paired_bootstrap(a,b)},
                    "vs_invalidated_leg":{"gain":oldgain,"loss":oldloss,"p":oldp,**paired_bootstrap(o,b)},
                    "scored_n":len(rows),"inference_seconds":sum(r["seconds"] for r in rows)}
        source={r["query_id"]:r for r in read_jsonl(retrieval_path(tag,"_records.jsonl")) if "correct" in r}
        comparison["groups"]={}
        for name,keep in {"primary":primary,"lookup_multi":lambda r:r["mode"]=="all" and r["m"]>1 and (r.get("aggregation") or "none")=="none",
                          "arithmetic":lambda r:r["mode"]=="all" and (r.get("aggregation") or "none")!="none",
                          "header":lambda r:r["mode"]=="any","data":lambda r:r["mode"]=="all","all":lambda r:True}.items():
            ks=[q for q,r in source.items() if keep(r)]
            hit=[q for q in ks if source[q]["correct"]]
            comparison["groups"][name]={"n":len(ks),"old_correct":sum(old[q]["answer_correct"] for q in ks),
                 "correct":sum(rmap[q]["answer_correct"] for q in ks),"n_hit":len(hit),
                 "hit_correct":sum(rmap[q]["answer_correct"] for q in hit)}
        result[tag]=comparison;ps[tag]=p
    for tag,p in holm(ps).items():result[tag]["vs_s3c"]["holm_p_three_arms"]=p
    (out/"COMPARISON.json").write_text(json.dumps(result,indent=2))
    if not all(r["complete"] for r in result.values()):
        raise RuntimeError("reader remeasurement is incomplete; no final verdict")
    rows=["# 수정된 검색 문맥의 답변 EM 재측정","",
          "학습·dev·리더 교체 없이, 6d88d4d의 수정 문맥을 기존 Qwen2.5-7B-Instruct 4bit와 BASE prompt로 다시 평가했다.",
          "S3c 원본 50문항은 예측과 채점이 모두 보관본과 일치했다. S3c 전체를 새로 생성한 것은 아니다.",
          "각 비교군 1,581문항을 모두 새로 생성하고, 주지표 991문항을 동일하게 골랐다.",
          "입력·답변의 문맥 해시, 문항 집합, 정답 주석 및 재채점 일치를 확인했다.","",
          "| 조건 | 수정 전 EM (무효 입력) | 수정 후 EM | 정답 / 991 |",
          "|---|---:|---:|---:|",
          "| S3c 기준선 (문맥 변화 없음) | 0.7164 | 0.7164 | 710 |"]
    for tag,r in result.items():
        rows.append(f"| {tag} | {r['old_invalidated_em']:.4f} | {r['em']:.4f} | {r['correct']} |")
    rows+=["","이 변화는 평가 결함 수정 효과다. 본 방법의 EM을 올린 새 방법 실험으로 해석하지 않는다.",
           "비교 대상은 공통 검색기·리더 위의 색인/검색 컴포넌트이며 각 논문 전체 시스템의 재현이 아니다.","",
           "## 집계 정정","",
           "- 주지표 991: retrieved 710, gold-only 948. 실제 retrieved 오답 281.",
           "- 검색 실패 오답 81 / 검색 성공이며 두 조건 오답 19 / 검색 성공이며 gold-only만 정답 181.",
           "- gold-only만 오답인 18문항은 retrieved 실패에 포함하지 않는다. 진단 파일 전체는 299문항.",
           "- 두 조건 모두 오답 25 = 검색 성공 19 + 검색 실패 6. 기존 reader_ceiling 19를 25로 고치는 것은 중복 분류다.",
           "- 데이터셀 1,245에서 검색 성공군 EM은 748/1,108=0.6751, 실패군은 7/137=0.0511.",
           "- gold-only/oracle은 특정 입력 정책의 결과다. 고정 리더의 절대 상한이나 0.8 불가능성 증명이 아니다.","",
           "## 한계와 출처","",
           "- test를 이미 여러 차례 분석했다. 이번 작업은 설정을 고정한 버그 수정 후 재측정이며 독립적인 새 test 성능이 아니다.",
           "- 헤더답 29문항의 기존 gold_v2는 64개 gold 좌표 제한 제거 전 입력이다. 이번 retrieved 비교군에는 그 제한이 없지만 gold 진단의 해당 부분은 별도 재실행이 필요하다.",
           "- GPU를 다른 프로그램과 공유했으므로 처리 시간은 속도 비교 근거로 사용하지 않는다.",
           "- 원본 실험 파일은 보존했다. 수정 결과는 이 폴더, 실제 입력은 ../audit_fix_20260910/.",
           "- 재현 설정: RUN_CONFIG.json, RUNTIME.json, TOKEN_AUDIT.json, BASELINE_CHECK.json.",
           "- 통계: COMPARISON.json, POPULATION_ACCOUNTING.json. 원본 분류: ../retrieval_accuracy/FAILURES.json.",
           "- 실행 절차: ../../REMEASURE-2026-09-10-audit.md.",
           ""]
    (out/"VERDICT.md").write_text("\n".join(rows))
    print(json.dumps({tag:{k:r[k] for k in ("n","correct","em","old_invalidated_em")} for tag,r in result.items()},indent=2))

if __name__=="__main__":main()
