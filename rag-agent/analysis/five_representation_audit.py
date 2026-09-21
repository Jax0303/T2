"""Verified, descriptive gap audit; no prompt or scoring-rule selection."""
import argparse
import itertools
import json
from pathlib import Path
import random
import statistics
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.repo))
    from analysis.validated_tables import load_manifest, paired_counts, build, read_leg, validate_answers
    from rag_agent.eval.artifacts import file_digest, read_records, validate_retrieval, digest

    manifest = args.repo / "analysis/comparison_manifest.five.v2.json"
    spec, base, arms, primary = load_manifest(manifest)
    report = build(manifest)  # also validates the gold diagnostic
    assert len(arms) == 4 and len(primary) == 991
    common = {q for q in primary if all(r[q]["correct"] for r, _, _, _ in arms.values())}
    result = {"manifest_sha256": file_digest(manifest), "primary_n": len(primary),
              "human_review_completed": False, "common_hit_ids": sorted(common),
              "arms": {}, "pairwise_common_hits": []}
    labels = {a["tag"]: a["label"] for a in spec["arms"]}
    for tag, (r, a, rm, am) in arms.items():
        c = paired_counts(r, a, primary)
        assert abs(c["retrieval"] - c["em"] - c["gap"]) < 1e-12
        tokens = [a[q]["n_tok"] for q in primary]
        cells = [r[q]["cells_in_context"] for q in primary]
        result["arms"][tag] = {
            "label": labels[tag], "primary": c,
            "common_hit": paired_counts(r, a, common) if common else None,
            "primary_context": {"mean_cells": statistics.mean(cells),
                                "max_cells": max(cells), "mean_tokens": statistics.mean(tokens),
                                "max_tokens": max(tokens)},
            "embedding_documents": rm["embedding_input_audit"]["documents"],
            "retrieval_sha256": rm["records_sha256"], "answers_sha256": am["records_sha256"]}
    for left, right in itertools.combinations(arms, 2):
        lr, la, _, _ = arms[left]
        rr, ra, _, _ = arms[right]
        ids = sorted(q for q in primary if lr[q]["correct"] and rr[q]["correct"])
        both = sum(bool(la[q]["answer_correct"] and ra[q]["answer_correct"]) for q in ids)
        lo = sum(bool(la[q]["answer_correct"] and not ra[q]["answer_correct"]) for q in ids)
        ro = sum(bool(ra[q]["answer_correct"] and not la[q]["answer_correct"]) for q in ids)
        result["pairwise_common_hits"].append({"left": left, "right": right, "n": len(ids),
             "both_correct": both, "left_only": lo, "right_only": ro,
             "neither": len(ids) - both - lo - ro,
             "left_em": (both+lo)/len(ids) if ids else None,
             "right_em": (both+ro)/len(ids) if ids else None, "query_ids": ids})
    diagnostic_dir = args.repo / "results/slide_audit_20260912"
    path_records_file = diagnostic_dir / "rowcol_path_records.jsonl"
    path_r = read_records(path_records_file)
    path_a, path_am = read_leg(diagnostic_dir / "rowcol_path_answer_primary.jsonl", verified=True)
    path_rm = json.loads((diagnostic_dir / "rowcol_path.json").read_text())
    leaf_r, leaf_a, leaf_rm, leaf_am = arms["rowcol_v2"]
    assert set(path_r) == set(leaf_r)
    assert path_rm["records_sha256"] == file_digest(path_records_file)
    assert path_rm["derivation"]["source_records_sha256"] == leaf_rm["records_sha256"]
    assert path_am["retrieval_records_sha256"] == path_rm["records_sha256"]
    assert path_am["query_ids_sha256"] == digest(sorted(primary))
    assert path_am["primary_only"] and path_am["condition"] == "retrieved"
    for field in ("reader_details", "prompt_sha256", "seed", "max_new_tokens", "scorer"):
        assert path_am[field] == leaf_am[field], field
    for q, row in path_r.items():
        validate_retrieval(row)
        for field in ("question", "answer", "mode", "m", "aggregation", "correct", "excluded", "gold_cells", "context_cells"):
            assert row.get(field) == leaf_r[q].get(field), (q, field)
    validate_answers(path_r, path_a, "path diagnostic", ids=primary)
    for q, a in path_a.items():
        assert a["context_sha256"] == a["source_context_sha256"] == path_r[q]["context_sha256"]
        assert a["cells_in_context"] == path_r[q]["cells_in_context"]
    result["rowcol_path_posthoc_diagnostic"] = {
        "primary": paired_counts(path_r, path_a, primary),
        "five_arm_common_hit": paired_counts(path_r, path_a, common) if common else None,
        "answers_sha256": path_am["records_sha256"],
        "retrieval_sha256": path_rm["records_sha256"]}
    lines = ["# 네 표현의 검색–답변 간극 감사", "",
             "전체 원본 시스템이 아닌 공통 검색기·리더 아래의 HiTab 표현 적응 비교다. test 결과를 본 후 시행한 재측정/사후 진단이다.", "",
             "| 표현 | 검색 | EM | 차이(pp) | hit·오답 | miss·정답 | 평균 셀 | 평균 입력 토큰 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for tag, d in result["arms"].items():
        c, x = d["primary"], d["primary_context"]
        lines.append(f"| {labels[tag]} | {c['retrieval']:.4f} | {c['em']:.4f} | {100*c['gap']:.2f} | {c['hit_wrong']} | {c['miss_correct']} | {x['mean_cells']:.1f} | {x['mean_tokens']:.1f} |")
    lines += ["", "차이 = (hit·오답 − miss·정답)/991. 동일 리더라도 전달 정보, 문맥 크기, 방해 셀, 검색 성공 문항 구성에 따라 간극이 다를 수 있다. 이 표만으로 각각의 인과적 기여량을 알 수는 없다.", "",
              f"## 네 표현 모두 검색 성공한 {len(common)}문항", "",
              "선택된 부분집합의 기술 통계다. 전체 성능이나 모든 난도에서의 우열로 일반화하지 않는다.", "",
              "| 표현 | 정답 | EM |", "|---|---:|---:|"]
    for tag, d in result["arms"].items():
        c = d["common_hit"]
        if c:
            lines.append(f"| {labels[tag]} | {c['hit_correct']}/{c['n']} | {c['em']:.4f} |")
    lines += ["", "## 쌍별 공통 검색 성공", "", "| 왼쪽 | 오른쪽 | query count | 왼쪽 EM | 오른쪽 EM | 왼쪽만 정답 | 오른쪽만 정답 |", "|---|---|---:|---:|---:|---:|---:|"]
    for p in result["pairwise_common_hits"]:
        if p["n"]:
            lines.append(f"| {labels[p['left']]} | {labels[p['right']]} | {p['n']} | {p['left_em']:.4f} | {p['right_em']:.4f} | {p['left_only']} | {p['right_only']} |")
    d = result["rowcol_path_posthoc_diagnostic"]
    c = d["primary"]
    lines += ["", "## 별도: 같은 검색 셀의 RowCol 전체 헤더 경로 복원", "",
              f"주지표 {c['n']}문항: 검색 {c['retrieval']:.4f}, EM {c['em']:.4f}, 차이 {100*c['gap']:.2f}pp. 네 원래 표현의 표와 구분한 사후 진단이다."]
    c = d["five_arm_common_hit"]
    if c:
        lines += [f"위 네 표현의 공통 hit {c['n']}문항에서 경로 복원 EM은 {c['em']:.4f} ({c['hit_correct']}/{c['n']})이다."]
    lines += ["", "## 비교의 한계", "",
              "- 같은 셀 예산 중단 규칙을 썼지만 큰 단위를 통째로 전달하므로 실제 셀 수·입력 길이는 다르다.",
              "- RowCol leaf 결과에는 헤더 경로 정보 손실이 있다. 같은 셀에 경로를 복원한 별도 진단(991건 EM .3290→.4289)을 함께 공개해야 한다. 길이 증가도 수반한 사후 개입이며 원본 논문 재현 성능이 아니다.",
              "- 두 문자 청킹은 별개 구현이다. 고정 청킹은 자체 행 보존 Markdown, Huawei 착안 청킹은 native splitter만 재현한다. 전체 TableRAG로 부르지 않는다.",
              "- 검증기는 모든 답변을 같은 기존 점수 함수로 재채점하고 원본 검색 문맥 hash·모델·프롬프트·모집단을 대조했다. 사람이 의미적 정답을 검수한 것은 아니다.",
              "- 사람 검수용 파일에는 방법명과 자동 점수 없이 모든 주지표 예측을 포함했다. 실제 검수와 채점 규칙 동결 전에는 별도 점수를 발표하지 않는다."]
    # Complete, blinded review queue, including already-correct predictions.
    groups = {}
    for tag, (r, a, _, _) in arms.items():
        for q in sorted(primary):
            key = (q, a[q]["pred"])
            if key not in groups:
                groups[key] = {"question": r[q]["question"], "prediction": a[q]["pred"],
                               "reference_answer": a[q]["answer"], "mappings": []}
            groups[key]["mappings"].append({"arm": tag, "query_id": q,
                                            "automatic_correct": a[q]["answer_correct"]})
    items = list(groups.values())
    random.Random(42).shuffle(items)
    blind, key_rows = [], []
    for i, d in enumerate(items, 1):
        rid = f"review-{i:05d}"
        blind.append({"review_id": rid, **{k:v for k,v in d.items() if k != "mappings"},
                      "human_label": None, "human_reason": None})
        key_rows.append({"review_id": rid, "mappings": d["mappings"]})
    assert sum(len(d["mappings"]) for d in key_rows) == 4*991
    result["blinded_review_unique_predictions"] = len(blind)
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts = {"GAP_REPORT.md": "\n".join(lines)+"\n", "VERIFIED_TABLES.md": report,
                 "audit.json": json.dumps(result, ensure_ascii=False, indent=2)+"\n",
                 "blind_review.jsonl": "".join(json.dumps(d, ensure_ascii=False)+"\n" for d in blind),
                 "review_key.jsonl": "".join(json.dumps(d, ensure_ascii=False)+"\n" for d in key_rows)}
    for name in artifacts:
        if (args.out/name).exists():
            raise FileExistsError(args.out/name)
    for name, body in artifacts.items():
        (args.out/name).write_text(body, encoding="utf-8")
    print("\n".join(lines[:15]))


if __name__ == "__main__":
    main()
