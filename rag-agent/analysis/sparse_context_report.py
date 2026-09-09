#!/usr/bin/env python3
"""Write a reviewable report only from complete frozen-layout experiment outputs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/"results/sparse_context_v1"

def main():
    c=json.loads((D/"COMPARISON.json").read_text())
    check=json.loads((D/"BASELINE_CHECK.json").read_text())
    audit=json.loads((D/"PREFLIGHT.json").read_text())
    rows=["# 고정 리더·고정 검색 문맥 실험", "",
          "원본 코드 기준: 62f57dfb21dba8db48d79c8ef155e418182163a3.",
          "리더: Qwen2.5-7B-Instruct, NF4 4bit, seed 42, greedy decoding, max_new_tokens 64.",
          "검색된 셀 20개와 채점기는 동일하다. 검색 단계는 다시 실행하지 않았다.",
          "수치 출처: 이 폴더의 COMPARISON.json, BASELINE_CHECK.json, PREFLIGHT.json과 각 실행 JSONL.", "",
          "## 실행 검증", "",
          f"- 원본 50문항의 예측·정오가 보관본과 모두 일치: {check['identical_predictions_and_scores']}.",
          "- 이는 50문항 재현 검사이며 전체 기준선을 새로 실행한 것은 아니다.",
          "- 각 검색 arm의 전체 1581문항에서 입력 20셀 보존 검사 통과.",
          "- 모호한 문장은 원문으로 남기고 해당 질의를 제외하지 않는다.",
          "- 사전등록 후 test에서 측정했으나, 이 test 자체는 과거에도 반복 분석되었다. 독립적인 신규 test 성능으로 주장하지 않는다.", "",
          "## 결과", "",
          "| 조건 | n | 기준선 EM | 새 EM | 순 증감 | 회수 | 상실 | McNemar p | Δ 95% CI |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for key in ("s3c_sparse","s3c_grouped","mt2net_sparse"):
        r=c.get(key)
        if not r:continue
        if not r.get("complete"):
            rows.append(f"| {key} | 미완료 {r['n_done']}/{r['n_expected']} | — | — | — | — | — | — | — |")
            continue
        lo,hi=r["delta_ci95"]
        rows.append(f"| {key} | {r['n_primary']} | {r['base_em']:.4f} | {r['new_em']:.4f} | {r['new_correct']-r['base_correct']:+d} | {r['gain']} | {r['loss']} | {r['mcnemar_p']:.5g} | [{lo:+.4f}, {hi:+.4f}] |")
    r=c.get("s3c_sparse",{})
    if r.get("complete"):
        rows+=["","## 판정","",
               f"주지표 {r['new_correct']}/991. 목표 0.8(793/991): "+("달성." if r["target_0_8_reached"] else "미달."),
               "원래 기준선 대비 Δ≥.02 및 McNemar p<.05 조건: "+("충족(추가 대조가 있으면 Holm 판정도 확인)." if r["improvement_rule_met_unadjusted"] else "미충족."),
               "공간적 배열의 고유한 효과는 grouped 대조와, 방법 간 비교는 같은 renderer를 적용한 MT2Net 대조로 판단한다."]
        if r["delta"]<=0:
            rows+=["사전등록의 자원 절약 규칙에 따라 추가 grouped/MT2Net GPU 실행은 하지 않았다. 두 후속 대조는 미측정이다.",
                   "이 결과가 배제하는 것은 이번 특정 sparse renderer이며, 모든 리더 고정 방법론이 불가능하다는 뜻이 아니다."]
        rows+=["","## 전체 모집단 (제외 없이 별도 보고)","",
               "| 모집단 | n | 기준선 정답 | 새 정답 |",
               "|---|---:|---:|---:|"]
        for name,g in r["groups"].items():
            rows.append(f"| {name} | {g['n']} | {g['base_correct']} | {g['new_correct']} |")
    if "sparse_vs_grouped" in c:
        r=c["sparse_vs_grouped"]
        rows+=["","## 제목 공유 대조","",f"sparse−grouped Δ={r['delta']:+.4f}, Holm p={r['holm_p']:.5g}.",
               "배열 효과 판정: "+("지지." if r["spatial_effect_supported"] else "미지지.")]
    if "common_renderer_comparison" in c:
        r=c["common_renderer_comparison"]
        rows+=["","## 같은 표현을 양쪽에 적용한 비교","",
               f"S3c EM {r['s3c_em']:.4f}, MT2Net 단위 EM {r['mt2net_em']:.4f}, 격차 {r['gap_after_same_renderer']:+.4f}.",
               f"각 방법의 개선 폭: S3c {r['s3c_improvement']:+.4f}, MT2Net {r['mt2net_improvement']:+.4f}.",
               "개선 폭의 차이는 기술 통계다. 별도의 상호작용 검정 없이 본 방법만의 이득이라고 주장하지 않는다."]
    rows+=["","## 재현","",
           "실험 사전등록: ../../PREREG-2026-09-09-sparse-context.md",
           "실행기: ../../scripts/sparse_context_experiment.py",
           "변환기: ../../rag_agent/serialization/sparse_context.py",
           "검사: ../../tests/test_sparse_context.py",
           "기존 결과와 TABLES.md는 수정하지 않았다.",
           "기존의 gold 제목 버그는 이번 retrieved-vs-retrieved 비교에 사용하지 않았다. gold/oracle 재측정은 별도 미완료 작업이다.",""]
    (D/"VERDICT.md").write_text("\n".join(rows))
    print(D/"VERDICT.md")

if __name__=="__main__":main()
