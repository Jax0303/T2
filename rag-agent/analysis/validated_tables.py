# SPDX-License-Identifier: MIT
"""Manifest-selected results, exact populations, and paired gap accounting.

Historical records remain inspectable but never become validated v2 runs merely
because their arithmetic can be recomputed. No missing-file or subset fallback.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rag_agent.eval.artifacts import (digest, file_digest, read_records,
                                     require_same_ids, validate_retrieval)
from rag_agent.eval.metrics import hitab_exact_match_text

DEFAULT_MANIFEST = ROOT / "analysis/comparison_manifest.json"
ROLES = {"reference", "representation", "ablation", "noncomparable", "invalid"}


def paired_counts(retrieval, answers, ids):
    if not ids:
        raise ValueError("empty reporting population")
    counts = {"hit_correct": 0, "hit_wrong": 0, "miss_correct": 0, "miss_wrong": 0}
    for q in ids:
        hit = bool(retrieval[q]["correct"])
        correct = bool(answers[q]["answer_correct"])
        counts[("hit" if hit else "miss") + ("_correct" if correct else "_wrong")] += 1
    n = len(ids)
    hits = counts["hit_correct"] + counts["hit_wrong"]
    return {**counts, "n": n, "retrieval": hits / n,
            "em": (counts["hit_correct"] + counts["miss_correct"]) / n,
            "gap": (counts["hit_wrong"] - counts["miss_correct"]) / n,
            "conditional_em": counts["hit_correct"] / hits if hits else None}


def validate_answers(retrieval, answers, label):
    scored = {q: r for q, r in retrieval.items() if "correct" in r}
    require_same_ids(scored, answers, label)
    for q, a in answers.items():
        r = scored[q]
        if "question" in a and a["question"] != r["question"]:
            raise ValueError(f"{label}/{q}: question differs from retrieval")
        for key in ("answer", "mode", "aggregation"):
            if a.get(key) != r.get(key):
                raise ValueError(f"{label}/{q}: {key} differs from retrieval")
        if a["retrieval_correct"] != r["correct"]:
            raise ValueError(f"{label}/{q}: retrieval flag differs")
        if a["answer_correct"] != int(hitab_exact_match_text(a["pred"], a["answer"])):
            raise ValueError(f"{label}/{q}: stored EM differs from rescoring")


def read_leg(path, *, verified):
    rows = read_records(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if verified and (meta.get("records_sha256") != file_digest(path)
                     or meta.get("context_version") != 2):
        raise ValueError(f"{path}: missing/stale v2 answer metadata")
    return rows, meta


def load_manifest(path=DEFAULT_MANIFEST):
    path = Path(path)
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("version") != 2 or spec.get("evidence") not in {"historical", "verified_v2"}:
        raise ValueError("manifest must declare version=2 and historical/verified_v2 evidence")
    verified = spec["evidence"] == "verified_v2"
    base = (path.parent / spec["results_dir"]).resolve()
    arms = spec["arms"]
    if len({a["tag"] for a in arms}) != len(arms):
        raise ValueError("duplicate arm tag in manifest")
    if any(a["role"] not in ROLES for a in arms):
        raise ValueError("unknown comparison role")
    loaded = {}
    for arm in arms:
        if arm["role"] in {"invalid", "noncomparable"}:
            continue
        tag = arm["tag"]
        rp = base / arm["retrieval"]
        retrieval = read_records(rp)
        rm = json.loads((base / arm["retrieval_summary"]).read_text(encoding="utf-8"))
        ap = base / arm["answers"]
        answers, am = read_leg(ap, verified=verified)
        validate_answers(retrieval, answers, tag)
        if verified:
            for row in retrieval.values():
                validate_retrieval(row)
            if rm.get("records_sha256") != file_digest(rp) or rm.get("context_version") != 2:
                raise ValueError(f"{tag}: missing/stale v2 retrieval metadata")
            if rm.get("comparison_scope") != "index_representation_adaptation":
                raise ValueError(f"{tag}: retrieval hints/full systems require a separate evaluation")
            if not rm.get("provenance", {}).get("source_sha256") or not am.get("provenance", {}).get("source_sha256"):
                raise ValueError(f"{tag}: missing source provenance")
            if am.get("retrieval_records_sha256") != file_digest(rp) or am["condition"] != "retrieved":
                raise ValueError(f"{tag}: answers came from a different retrieval/context condition")
            if am.get("query_ids_sha256") != digest(sorted(answers)):
                raise ValueError(f"{tag}: answer population hash differs")
            for q, answer in answers.items():
                r = retrieval[q]
                if "question" not in answer:
                    raise ValueError(f"{tag}/{q}: v2 answers must record the question")
                if (answer.get("context_sha256") != r["context_sha256"]
                        or answer.get("source_context_sha256") != r["context_sha256"]
                        or answer["cells_in_context"] != r["cells_in_context"]):
                    raise ValueError(f"{tag}/{q}: reader evidence differs from scored evidence")
        loaded[tag] = (retrieval, answers, rm, am)
    ref = spec["reference"]
    if ref not in loaded or next(a for a in arms if a["tag"] == ref)["role"] != "reference":
        raise ValueError("manifest reference must be an included reference arm")
    rr, ra, rrm, ram = loaded[ref]
    primary = {q for q, r in rr.items() if r.get("mode") == "all" and r.get("m") == 1
               and (r.get("aggregation") or "none") == "none" and "correct" in r}
    expected = spec["expected_counts"]
    observed = {"queries": len(rr), "scored": len(ra), "primary": len(primary),
                "excluded": sum("correct" not in r for r in rr.values())}
    if observed != expected:
        raise ValueError(f"manifest population differs: expected={expected}, observed={observed}")
    for tag, (r, a, rm, am) in loaded.items():
        require_same_ids(rr, r, tag)
        require_same_ids(ra, a, tag)
        for q in rr:
            for key in ("mode", "m", "answer", "question", "aggregation", "excluded", "table_id"):
                if rr[q].get(key) != r[q].get(key):
                    raise ValueError(f"{tag}/{q}: {key} differs across arms")
            if verified and rr[q].get("gold_cells") != r[q].get("gold_cells"):
                raise ValueError(f"{tag}/{q}: gold coordinates differ")
        if verified:
            for key in ("dataset", "split", "corpus", "alpha", "budget_cells", "budget_policy",
                        "encoder_details"):
                if rm.get(key) != rrm.get(key) or key not in rm:
                    raise ValueError(f"{tag}: uncontrolled retrieval setting {key}; use a separate manifest")
            for key in ("reader_details", "prompt_sha256", "seed", "max_new_tokens", "scorer"):
                if am.get(key) != ram.get(key) or key not in am:
                    raise ValueError(f"{tag}: uncontrolled reader setting {key}; use a separate manifest")
    return spec, base, loaded, primary


def build(path=DEFAULT_MANIFEST):
    spec, base, loaded, primary = load_manifest(path)
    verified = spec["evidence"] == "verified_v2"
    lines = ["# HiTab 검색·답변 비교", "",
             "검증된 새 실행(v2)." if verified else
             "**기존 실행의 감사용 재집계입니다. 수정 코드의 성능 결과가 아닙니다. 기존 문맥의 무결성은 보증하지 않습니다.**",
             "", "범위: 공통 검색기·리더 아래 색인 표현 비교. Huawei/Google TableRAG와 MT2Net 전체 시스템 재현 점수로 인용할 수 없습니다.",
             "", f"전체 {spec['expected_counts']['queries']}건, 채점 {spec['expected_counts']['scored']}건, "
             f"제외 {spec['expected_counts']['excluded']}건. 주지표는 단일 셀 조회 {len(primary)}건입니다.", ""]
    ref_r = loaded[spec["reference"]][0]
    populations = [("단일 셀 조회 — 주지표", primary),
                   ("데이터셀 전체 — 보조", {q for q, r in ref_r.items() if "correct" in r and r["mode"] == "all"}),
                   ("헤더답 — 별도 기준", {q for q, r in ref_r.items() if "correct" in r and r["mode"] == "any"})]
    for name, ids in populations:
        if not ids:
            continue
        lines += [f"## {name}", "", "| 표현 | n | 검색 | EM | 차이(pp) | hit→오답 | miss→정답 | hit 조건 EM | 평균 셀 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for arm in spec["arms"]:
            if arm["tag"] not in loaded:
                continue
            r, a, _, _ = loaded[arm["tag"]]
            c = paired_counts(r, a, ids)
            cond = f"{c['conditional_em']:.4f}" if c["conditional_em"] is not None else "—"
            mean = sum(r[q]["cells_in_context"] for q in ids) / len(ids)
            lines.append(f"| {arm['label']} | {len(ids)} | {c['retrieval']:.4f} | {c['em']:.4f} | "
                         f"{100*c['gap']:.2f} | {c['hit_wrong']} | {c['miss_correct']} | {cond} | {mean:.1f} |")
        lines.append("")
    outside = {q for q, r in ref_r.items() if "correct" in r and r["mode"] == "all"} - primary
    if outside:
        included = [a for a in spec["arms"] if a["tag"] in loaded]
        lines += ["## 주지표 밖 데이터셀 질의 — 유형별 전수", "",
                  "| 유형 | n | " + " | ".join(a["label"] + " EM" for a in included) + " |",
                  "|---|---:|" + "---:|" * len(included)]
        for kind in sorted({ref_r[q].get("aggregation") or "none" for q in outside}):
            ids = {q for q in outside if (ref_r[q].get("aggregation") or "none") == kind}
            scores = [sum(loaded[a["tag"]][1][q]["answer_correct"] for q in ids) / len(ids) for a in included]
            lines.append(f"| {kind} | {len(ids)} | " + " | ".join(f"{v:.4f}" for v in scores) + " |")
        lines.append("")
    excluded = {q: r["excluded"] for q, r in ref_r.items() if "correct" not in r}
    lines += ["## 제외 질의", ""] + [f"- `{q}`: {reason}" for q, reason in excluded.items()] + [""]
    lines += ["차이 = (검색 hit·답변 오답 − 검색 miss·답변 정답) / n. 검색 실패율을 답변 실패율로 간주하지 않습니다.",
              "각 방법의 hit 집합은 다르므로 조건부 EM만으로 리더 우열을 판단하지 않습니다.",
              "헤더답의 any 판정은 관련 셀 포함 여부의 대리 지표입니다. 해당 헤더 문자열까지 전달됐다는 보장은 아니므로 주지표에 합치지 않습니다.",
              "셀 예산은 마지막 단위를 통째로 넣는 중단 기준입니다. 실제 문맥 크기는 같지 않습니다.",
              "HiTab 숫자 비교 허용오차는 1e-5이며, 다중값 텍스트 파싱은 이 저장소의 어댑터입니다.", "",
              "## 비교에서 제외한 실행", ""]
    for arm in spec["arms"]:
        if arm["tag"] not in loaded:
            lines.append(f"- {arm['label']}: {arm['reason']}")
    if spec.get("gold_diagnostic"):
        gpath = base / spec["gold_diagnostic"]
        gold, gm = read_leg(gpath, verified=verified)
        rr, ra, _, ram = loaded[spec["reference"]]
        validate_answers(rr, gold, "gold diagnostic")
        if gm["condition"] != "gold":
            raise ValueError("gold diagnostic must explicitly select a gold condition")
        if verified:
            for key in ("retrieval_records_sha256", "reader_details", "prompt_sha256", "seed", "max_new_tokens"):
                if key not in gm or gm[key] != ram[key]:
                    raise ValueError(f"gold/retrieved conditions disagree on {key}")
        both = sum(gold[q]["answer_correct"] and ra[q]["answer_correct"] for q in primary)
        go = sum(gold[q]["answer_correct"] and not ra[q]["answer_correct"] for q in primary)
        ro = sum(not gold[q]["answer_correct"] and ra[q]["answer_correct"] for q in primary)
        neither = len(primary) - both - go - ro
        composed = sum((gold[q] if rr[q]["correct"] else ra[q])["answer_correct"] for q in primary)
        lines += ["", "## Gold 문맥 진단", "", f"선택 파일: `{spec['gold_diagnostic']}` — 구 gold로 자동 대체하지 않습니다.",
                  f"Gold EM {(both+go)/len(primary):.4f}; 두 조건 정답 {both}, gold만 정답 {go}, 검색 문맥만 정답 {ro}, 둘 다 오답 {neither}.",
                  f"hit에 gold, miss에 검색 답변을 고르는 오프라인 합성 EM: {composed/len(primary):.4f}. 새 생성 실험이 아닙니다.",
                  "Gold 문맥은 진단용 조건이며 답변 정확도의 수학적 상한이 아닙니다."]
    lines += ["", "## 실행 주석", ""]
    lines += [f"- {a['label']}: {a['note']}" for a in spec["arms"] if a.get("note")]
    if verified:
        for tag, (_, _, rm, am) in loaded.items():
            audit = rm.get("embedding_input_audit") or {}
            docs = audit.get("documents") or {}
            queries = audit.get("queries") or {}
            lines.append(f"- `{tag}`: 임베딩 초과 정책 `{docs.get('policy', 'BM25 only')}`, "
                         f"문서 {docs.get('n_overflow', 0)}/{docs.get('n', 0)}, "
                         f"질의 {queries.get('n_overflow', 0)}/{queries.get('n', 0)}; "
                         f"chunk {rm.get('chunk_size')} {rm.get('chunk_measure')}, "
                         f"overlap {rm.get('chunk_overlap')}; reader prompt `{am['prompt']}`.")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = build(args.manifest)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
