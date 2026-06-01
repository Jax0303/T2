#!/home/user/T2/hart-table-retrieval/.venv/bin/python3
"""
VDB / OriginalDB 분리 라우팅 + LLM 코드 생성 + 실행 검증 스크립트
=================================================================

구조:
  쿼리 → 분류
    ├─ 단순 검색 (추론 불필요)  → 원본 DB에서 직접 키워드 검색 → 답
    ├─ 추론 필요 (의미 검색)    → VDB → 관련 테이블 → LLM 코드 생성 → 실행 → 답
    └─ 수식/복잡 계산           → 테이블 + LLM 코드 생성 → 실행 → 답

데이터: HiTab dev split
VDB:   Chroma (bge-large-en-v1.5)
LLM:   Groq API (llama-3.3-70b-versatile)
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ─── Configuration ───────────────────────────────────────────────
HITAB_DIR = "/home/user/T2/hart-table-retrieval/data/hitab"
CHROMA_DIR = "/home/user/T2/hart-table-retrieval/data/chroma_db"
GROQ_MODEL = "llama-3.3-70b-versatile"
LLM_BACKEND = os.environ.get("LLM_BACKEND", "local")  # "local" | "groq"
QUERIES_PER_CLASS = 3  # 클래스별 테스트할 쿼리 수
SEED = 42
CODE_EXEC_TIMEOUT = 10  # seconds


def _make_llm():
    backend = LLM_BACKEND.lower()
    if backend == "groq":
        return GroqLLM(GROQ_MODEL)
    if backend == "local":
        return LocalQwen()
    raise ValueError(f"Unknown LLM_BACKEND={backend!r}")


# ══════════════════════════════════════════════════════════════════
#  1. DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_samples(split="dev"):
    fpath = Path(HITAB_DIR) / "data" / f"{split}_samples.jsonl"
    samples = []
    with open(fpath, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def load_table(table_id: str) -> Optional[dict]:
    for subdir in ["hmt", "raw"]:
        p = Path(HITAB_DIR) / "data" / "tables" / subdir / f"{table_id}.json"
        if p.exists():
            with open(p, encoding="utf-8") as f:
                t = json.load(f)
            t["table_id"] = table_id
            return t
    return None


def get_query(s): return s.get("sub_sentence") or s.get("question") or ""
def get_answer(s): return s.get("answer", [])
def get_table_id(s): return s.get("table_id", "")


# ══════════════════════════════════════════════════════════════════
#  2. TABLE PARSING  — 원본 DB (OriginalStore)
# ══════════════════════════════════════════════════════════════════

def _parse_paths(root: dict):
    """HiTab header tree → list of leaf paths."""
    leaf_paths = []
    by_line_idx = {}
    def walk(node, prefix):
        name = str(node.get("value") or node.get("name") or "").strip()
        path = prefix + [name] if name and name not in ("<TOP>","<LEFT>","<ROOT>") else prefix
        children = node.get("children_dict") or node.get("children") or []
        if isinstance(children, dict):
            children = list(children.values())
        if not children:
            leaf_paths.append(path)
            li = node.get("line_idx")
            if li is not None:
                by_line_idx[int(li)] = path
            return
        for ch in children:
            if isinstance(ch, dict):
                walk(ch, path)
    if root:
        walk(root, [])
    return leaf_paths, by_line_idx


@dataclass
class ParsedTable:
    table_id: str
    title: str
    data: list  # rows x cols
    col_headers: list  # list of header path lists
    row_headers: list  # list of header path lists

    @property
    def n_rows(self): return len(self.data)
    @property
    def n_cols(self): return len(self.data[0]) if self.data else 0

    def to_text(self, max_rows=30) -> str:
        """테이블을 LLM이 이해할 수 있는 텍스트로 변환."""
        lines = [f"Title: {self.title}"]
        lines.append("Columns:")
        for c, hdr in enumerate(self.col_headers):
            lines.append(f"  col[{c}]: {' > '.join(hdr) if hdr else '(blank)'}")
        lines.append("Data (row_header → values):")
        for r in range(min(self.n_rows, max_rows)):
            rh = " > ".join(self.row_headers[r]) if r < len(self.row_headers) and self.row_headers[r] else f"row_{r}"
            vals = []
            for c in range(min(self.n_cols, 15)):
                v = self.data[r][c]
                if isinstance(v, dict):
                    v = v.get("value", "")
                vals.append(str(v) if v is not None else "")
            lines.append(f"  row[{r}] ({rh}): {' | '.join(vals)}")
        if self.n_rows > max_rows:
            lines.append(f"  (...{self.n_rows - max_rows} more rows)")
        return "\n".join(lines)

    def to_csv_string(self) -> str:
        """테이블을 CSV 문자열로 변환 (코드 실행용)."""
        import csv, io
        buf = io.StringIO()
        writer = csv.writer(buf)
        # header row
        col_names = []
        for c, hdr in enumerate(self.col_headers):
            col_names.append(" > ".join(hdr) if hdr else f"col_{c}")
        # add row_header column
        writer.writerow(["row_header"] + col_names)
        for r in range(self.n_rows):
            rh = " > ".join(self.row_headers[r]) if r < len(self.row_headers) and self.row_headers[r] else f"row_{r}"
            vals = []
            for c in range(self.n_cols):
                v = self.data[r][c]
                if isinstance(v, dict):
                    v = v.get("value", "")
                vals.append(str(v) if v is not None else "")
            writer.writerow([rh] + vals)
        return buf.getvalue()


def parse_table(raw: dict) -> ParsedTable:
    table_id = raw.get("table_id") or raw.get("uid") or "unknown"
    title = raw.get("title", "") or ""

    top_paths, top_by_li = _parse_paths(raw.get("top_root") or {})
    left_paths, left_by_li = _parse_paths(raw.get("left_root") or {})

    rows = []
    for r in raw.get("data") or []:
        rows.append([cell.get("value") if isinstance(cell, dict) else cell for cell in r])

    n_cols = len(rows[0]) if rows else 0
    n_rows = len(rows)

    # col headers
    col_headers = []
    if top_by_li:
        for c in range(n_cols):
            col_headers.append(top_by_li.get(c, top_paths[c] if c < len(top_paths) else []))
    else:
        for c in range(n_cols):
            col_headers.append(top_paths[c] if c < len(top_paths) else [])

    # row headers
    row_headers = []
    if left_by_li:
        for r in range(n_rows):
            row_headers.append(left_by_li.get(r, left_paths[r] if r < len(left_paths) else []))
    else:
        for r in range(n_rows):
            row_headers.append(left_paths[r] if r < len(left_paths) else [])

    return ParsedTable(table_id, title, rows, col_headers, row_headers)


_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")


def _cell_to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if not s:
        return None
    m = _NUM_RE.fullmatch(s) or _NUM_RE.match(s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def _query_numbers(text: str) -> List[float]:
    out = []
    for m in _NUM_RE.findall(text or ""):
        try:
            out.append(float(m.replace(",", "")))
        except ValueError:
            pass
    return out


def _num_signal_match(n: float, cells: set, rel_tol: float = 0.02) -> bool:
    """쿼리 숫자 n이 실제 셀 집합과 매칭되는지 (검색 신호용).

    퍼센트 표기 불일치(쿼리 "27%"→27 vs 셀 0.27367)와 반올림을 흡수하려고
    {n, n/100, n*100} 변형을 codegen NM과 동일한 ±2% 상대오차로 비교한다.
    """
    for cand in (n, n / 100.0, n * 100.0):
        for c in cells:
            if abs(cand - c) <= max(1e-6, rel_tol * abs(cand)):
                return True
    return False


class OriginalDB:
    """원본 DB — 키워드 검색으로 테이블 찾기."""

    def __init__(self):
        self._tables: Dict[str, ParsedTable] = {}
        self._keywords: Dict[str, set] = {}  # table_id → keyword set
        # 구조 인식 검색용 인덱스: 헤더 토큰(셀 문자열 제외) + 숫자 셀
        self._header_kws: Dict[str, set] = {}   # table_id → header/title token set
        self._num_cells: Dict[str, set] = {}    # table_id → {float}

    def add(self, raw: dict):
        t = parse_table(raw)
        self._tables[t.table_id] = t
        # 키워드 인덱스 구축
        kws = set()
        kws.update(self._tokenize(t.title))
        header_kws = set(kws)
        for hdr in t.col_headers + t.row_headers:
            for seg in hdr:
                toks = self._tokenize(seg)
                kws.update(toks)
                header_kws.update(toks)
        num_cells = set()
        for row in t.data:
            for v in row:
                if isinstance(v, str):
                    kws.update(self._tokenize(v))
                f = _cell_to_float(v)
                if f is not None:
                    num_cells.add(f)
        self._keywords[t.table_id] = kws
        self._header_kws[t.table_id] = header_kws
        self._num_cells[t.table_id] = num_cells

    def get(self, table_id: str) -> Optional[ParsedTable]:
        return self._tables.get(table_id)

    @staticmethod
    def _tokenize(text: str) -> set:
        return {w.lower() for w in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", str(text)) if len(w) >= 3}

    _STOPWORDS = {
        "a","an","the","of","in","on","at","for","to","from","by","with",
        "and","or","is","was","were","be","been","are","as","it","this",
        "that","what","which","who","where","when","why","how","many",
        "much","do","does","did","has","have","had","than","then","into",
    }

    def keyword_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """키워드 매칭으로 테이블 검색 (단순 검색용)."""
        q_tokens = self._tokenize(query) - self._STOPWORDS
        if not q_tokens:
            return []
        scored = []
        for tid, kws in self._keywords.items():
            overlap = q_tokens & kws
            if overlap:
                score = len(overlap) / len(q_tokens)
                prec = len(overlap) / len(kws) if kws else 0.0
                scored.append((tid, score, prec))
        scored.sort(key=lambda x: (-x[1], -x[2], str(x[0])))
        return [(tid, score) for tid, score, _ in scored][:top_k]

    def structural_search(self, query: str, top_k: int = 5,
                          w_kw: float = 0.6, w_num: float = 0.4) -> List[Tuple[str, float]]:
        """구조 인식 원본 검색 — verifier 신호를 1차 검색기로 사용.

        VDB 직렬화가 버리는 신호 두 개로만 랭킹:
          (1) 헤더 트리 토큰 겹침 (셀 문자열 제외)
          (2) 쿼리 숫자 ↔ 실제 숫자 셀 겹침
        score = w_kw * header_token_overlap + w_num * numeric_cell_overlap
        (쿼리에 숫자가 없으면 키워드 겹침만으로 랭킹.)
        """
        q_tokens = self._tokenize(query) - self._STOPWORDS
        q_nums = _query_numbers(query)
        if not q_tokens and not q_nums:
            return []
        scored = []
        for tid in self._tables:
            hk = self._header_kws.get(tid, set())
            inter = q_tokens & hk
            kw_ov = (len(inter) / len(q_tokens)) if q_tokens else 0.0
            if q_nums:
                cells = self._num_cells.get(tid, set())
                matched = sum(1 for n in q_nums if _num_signal_match(n, cells))
                num_ov = matched / len(q_nums)
                score = w_kw * kw_ov + w_num * num_ov
            else:
                score = kw_ov
            if score > 0:
                # tie-break: 같은 점수면 헤더 정밀도(matched/헤더크기) 높은(=더 집중된)
                # 테이블 우선, 그다음 table_id로 결정적 정렬.
                prec = (len(inter) / len(hk)) if hk else 0.0
                scored.append((tid, score, prec))
        scored.sort(key=lambda x: (-x[1], -x[2], str(x[0])))
        return [(tid, score) for tid, score, _ in scored][:top_k]

    def __len__(self):
        return len(self._tables)


# ══════════════════════════════════════════════════════════════════
#  3. VECTOR DB (Chroma)  — 선택적 사용
# ══════════════════════════════════════════════════════════════════

class VectorDB:
    """VDB wrapper — 추론이 필요한 쿼리에서 의미 검색."""

    def __init__(self, chroma_dir: str, device: str = "cpu"):
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", device=device)
            client = chromadb.PersistentClient(path=chroma_dir)
            self.collection = client.get_collection("plain_markdown_bge_large_en_v1_5")
            self._available = True
            print(f"  VectorDB loaded: {self.collection.count()} vectors")
        except Exception as e:
            print(f"  VectorDB unavailable: {e}")
            self._available = False

    def search(self, query: str, top_k: int = 5,
               allowed_ids: Optional[set] = None) -> List[Tuple[str, float]]:
        """의미 검색. allowed_ids가 주어지면 그 후보군 안에서만 랭킹한다
        (원본 검색기와 동일한 후보 집합 위에서 공정 비교하기 위함)."""
        if not self._available:
            return []
        emb = self.model.encode([query], convert_to_numpy=True, show_progress_bar=False)[0].tolist()
        # 후보군 제한 시: pool 멤버가 전체 코퍼스에서 깊이 묻혀 있어도 누락되지 않도록
        # 충분히 깊게(코퍼스 크기 한도 내) 뽑아 클라이언트에서 필터한다.
        # → 코퍼스가 pool보다 커도 VDB의 pool-내 랭킹이 정확해 baseline이 불리해지지 않음.
        if allowed_ids is None:
            n_results = top_k * 4
        else:
            try:
                corpus_n = self.collection.count()
            except Exception:
                corpus_n = 20000
            n_results = min(corpus_n, max(len(allowed_ids) * 4, 2000))
        res = self.collection.query(
            query_embeddings=[emb], n_results=n_results,
            include=["metadatas", "distances"],
        )
        per_table = OrderedDict()
        for vec_id, meta, dist in zip(res["ids"][0], res["metadatas"][0], res["distances"][0]):
            tid = meta.get("table_id") or meta.get("uid") or vec_id.split("__")[0]
            if allowed_ids is not None and tid not in allowed_ids:
                continue
            score = 1.0 - float(dist)
            if tid not in per_table or score > per_table[tid]:
                per_table[tid] = score
        ranked = sorted(per_table.items(), key=lambda x: -x[1])
        return ranked[:top_k]


# ══════════════════════════════════════════════════════════════════
#  4. QUERY CLASSIFIER — 라우팅 결정
# ══════════════════════════════════════════════════════════════════

@dataclass
class QueryRoute:
    route: str        # "direct_lookup" | "vdb_codegen" | "codegen"
    needs_code: bool
    reason: str

_ARITH_PAT = re.compile(
    r"\b(sum of|combined|altogether|together|average|mean|median|"
    r"difference|differ|differs|gap|change|"
    r"increase[sd]?|decrease[sd]?|grew|grow(?:n|s)?|drop(?:ped|s)?|fell|rose|risen|"
    r"declines? of|declines? in|increases? of|increases? in|decreases? of|decreases? in|"
    r"ratio|fraction|proportion|percentage|percent|per ?cent|share of|out of|"
    r"accounted for|accounts for|account for|"
    r"divided by|multiplied|product of|range of|spread of|"
    r"rose (?:from|to|by)|fell (?:from|to|by)|grew (?:from|to|by)|"
    r"rose [a-z ]*from|fell [a-z ]*from|"
    r"how much (?:more|less|higher|lower)|total of|sum to|"
    r"by \d+(?:\.\d+)?\s?%|by \d+(?:\.\d+)?\s?per ?cent)"
    r"\b", re.IGNORECASE
)
_CMP_PAT = re.compile(
    r"\b(greater|less|more|fewer|higher|lower|exceed|exceeds|"
    r"bigger|smaller|above|below|than|compared|opposite|how many|"
    r"versus|vs\.?|relative to|times (?:more|less|higher|lower|the|as)|"
    r"twice|thrice|half (?:of|as))"
    r"\b", re.IGNORECASE
)
# Statement with embedded percentage or ratio number → needs calc/verify
_PCT_NUM_PAT = re.compile(r"\b\d+(?:\.\d+)?\s?%|\b\d+(?:\.\d+)?\s?per ?cent\b", re.IGNORECASE)
# "X to Y", "from X to Y" with numbers → change
_RANGE_NUM_PAT = re.compile(r"\bfrom\s+[\d.]+\s*%?\s+to\s+[\d.]+", re.IGNORECASE)
_ARG_PAT = re.compile(
    r"\b(highest|lowest|largest|smallest|maximum|minimum|most|least|biggest|"
    r"top|bottom|best|worst|peak)"
    r"\b", re.IGNORECASE
)
_SIMPLE_PAT = re.compile(
    r"^(what is the|what was the|what are|how much is|how many)\b", re.IGNORECASE
)
_ENTITY_PAT = re.compile(
    r"^\s*(?:who|which|what|where|in what|in which)\b", re.IGNORECASE
)
_MATH_SYM = re.compile(r"[+\-*/]")


def classify_query(q: str) -> QueryRoute:
    """쿼리 분류: 단순검색 vs VDB+코드생성 vs 코드생성."""
    q = (q or "").strip()

    # 수학 기호 2개 이상 → 무조건 코드 생성
    if len(_MATH_SYM.findall(q)) >= 2:
        return QueryRoute("codegen", True, "math symbols detected")

    # 산술 키워드 → 코드 생성 (VDB로 테이블 찾은 후)
    arith_hits = _ARITH_PAT.findall(q)
    if arith_hits:
        return QueryRoute("vdb_codegen", True, f"arithmetic: {arith_hits[0]}")

    # 비교/카운트 → 코드 생성
    if _CMP_PAT.search(q):
        return QueryRoute("vdb_codegen", True, "comparison/count")

    # argmax/argmin → 코드 생성 (정렬 필요)
    if _ARG_PAT.search(q):
        return QueryRoute("vdb_codegen", True, "arg-style query")

    # "from X to Y" 형태 (변화량) → 코드 생성
    if _RANGE_NUM_PAT.search(q):
        return QueryRoute("vdb_codegen", True, "range/change pattern")

    # 퍼센트 숫자 끼고 동사가 변화류면 코드 생성
    # (이미 _ARITH_PAT에서 대부분 잡히지만, "by 4%" 같은 표현 추가 안전망)
    if _PCT_NUM_PAT.search(q) and re.search(
        r"\b(by|of|from|to|than|over|under|above|below)\b", q, re.IGNORECASE
    ):
        return QueryRoute("vdb_codegen", True, "percent number with relator")

    # 엔티티 질문 ("who/which had...") → VDB + 코드
    if _ENTITY_PAT.match(q):
        return QueryRoute("vdb_codegen", True, "entity question needs reasoning")

    # 나머지 → 단순 검색
    return QueryRoute("direct_lookup", False, "simple lookup")


# ══════════════════════════════════════════════════════════════════
#  5. LLM CODE GENERATOR  (Groq)
# ══════════════════════════════════════════════════════════════════

CODEGEN_SYSTEM = """\
You are a Python code generator for table question answering.
You are given a pandas DataFrame `df` and a question about the table.
Write Python code that computes the answer and stores it in `result`.

About the DataFrame:
- `df` has a column called "row_header" containing the row labels.
- Other columns are named by their column header paths (e.g., "revenue > 2017").
- All cell values are strings. Convert to float when doing arithmetic:
  use `pd.to_numeric(df[col], errors='coerce')` to convert a column.

Safe helpers (already defined — USE THESE):
- `find_col(*substrs)` → first column name whose lowercase contains EVERY substr (str), or raises ValueError listing all columns.
- `find_rows(*substrs)` → DataFrame of rows where row_header contains EVERY substr (case-insensitive). Empty df if none.
- `cell(row_substrs, col_substrs)` → float at intersection. row_substrs/col_substrs can be str or tuple-of-str. Raises ValueError if row/col not uniquely identified.
- `colnum(col)` → pd.to_numeric(df[col], errors='coerce'). Shortcut.

Rules:
- `df` is already loaded. `find_col`, `find_rows`, `cell`, `colnum` are predefined. Do NOT redefine them.
- You may use: pandas (as `pd`), math, re. No other imports.
- ALWAYS use `find_col` to locate a column — do NOT hardcode the full path; pass distinguishing substrings.
- ALWAYS check `len(rows) > 0` before `.iloc[0]`, or use `cell(...)` which handles it for you.
- Store the final answer in a variable called `result`.
- For numeric answers: `result` should be a number (int or float).
- For name/entity answers: `result` should be a string. Use the value FROM `row_header`, not your own paraphrase.
- `print(result)` at the end.
- No try/except. If you can't compute it, let it crash — the harness logs the error.
- Keep it simple and direct. No unnecessary steps.

Example 1 — Sum across years:
Q: "What is the sum of revenue for 2017 and 2018?"
```python
c17 = find_col("revenue", "2017")
c18 = find_col("revenue", "2018")
result = float(colnum(c17).sum() + colnum(c18).sum())
print(result)
```

Example 2 — Which had the highest:
Q: "Which region had the highest sales?"
```python
col = find_col("sales")
idx = colnum(col).idxmax()
result = df.loc[idx, "row_header"]
print(result)
```

Example 3 — Difference between two row entities:
Q: "Difference between A and B in column X."
```python
result = cell("A", "X") - cell("B", "X")
print(result)
```

Example 4 — Ratio / share:
Q: "Theft accounted for what fraction of female offences?"
```python
theft = cell("theft", "female")
total = cell("total offences", "female")
result = theft / total
print(result)
```

Example 5 — Answer is a ROW LABEL (which row had largest/smallest):
Q: "Which region had the largest decline in robberies?"
```python
col = find_col("robber", "change")           # all substrings must appear in the column
vals = colnum(col)
idx = vals.idxmin()                          # most negative = largest decline
result = df.loc[idx, "row_header"]
print(result)
```

Example 6 — Compare two row entities (do NOT pass a tuple of entities):
Q: "Was chinese employment higher than white?"
```python
chinese = cell("chinese", "employment rate")
white   = cell("white",   "employment rate")
result  = "chinese" if chinese > white else "white"
print(result)
```
"""

DIRECT_ANSWER_SYSTEM = """\
You are a precise table QA assistant. Given a table and a question,
answer directly from the table data. Think step by step.
Output ONLY the final answer value (number or name). No explanation.
If the answer is a number, output just the number (no units, no commas, no '%').
"""


class LocalQwen:
    """Lazy wrapper around rag_agent.llm.local_qwen.LocalQwenLLM."""
    def __init__(self, model_name: str = "Qwen/Qwen2.5-7B-Instruct", quantization: str = "4bit"):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from rag_agent.llm.local_qwen import LocalQwenLLM
        self._impl = LocalQwenLLM(model_name=model_name, quantization=quantization)
        self.model = self._impl.name

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        return self._impl.complete(system, user, max_tokens=max_tokens)


class GroqLLM:
    def __init__(self, model: str = GROQ_MODEL):
        from groq import Groq
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError("GROQ_API_KEY not set")
        self.client = Groq(api_key=key, timeout=60.0)
        self.model = model

    def complete(self, system: str, user: str, max_tokens: int = 600) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(6):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last_err = e
                msg = str(e)
                # Retry on rate-limit / 5xx / timeout / connection issues
                retriable = (
                    "429" in msg or "rate" in msg.lower() or
                    "502" in msg or "503" in msg or "504" in msg or
                    "timeout" in msg.lower() or "timed out" in msg.lower() or
                    "connection" in msg.lower()
                )
                if not retriable:
                    raise
                # Try to parse Retry-After (sec); fall back to exp backoff capped at 60s
                wait = None
                m = re.search(r"try again in ([\d.]+)s", msg)
                if m: wait = float(m.group(1)) + 0.5
                if wait is None:
                    m = re.search(r"retry[-_ ]?after[:= ]+([\d.]+)", msg, re.IGNORECASE)
                    if m: wait = float(m.group(1)) + 0.5
                if wait is None:
                    wait = min(60.0, 3.0 * (2 ** attempt))
                time.sleep(wait)
                continue
        raise RuntimeError(f"Groq retries exhausted: {last_err}")


def generate_code(llm, query: str, table: ParsedTable) -> str:
    """LLM에게 코드 생성 요청."""
    table_text = table.to_text()
    # Row-header 카탈로그(첫 N개)로 컬럼/행 헷갈림 줄이기
    rh_sample = []
    for r in range(min(table.n_rows, 20)):
        if r < len(table.row_headers) and table.row_headers[r]:
            rh_sample.append(" > ".join(table.row_headers[r]))
    rh_block = "Row labels (row_header values):\n" + "\n".join(f"  - {rh}" for rh in rh_sample)
    if table.n_rows > 20:
        rh_block += f"\n  - (...{table.n_rows - 20} more rows)"

    user_prompt = (
        f"Table:\n{table_text}\n\n{rh_block}\n\n"
        f"Question: {query}\n\n"
        "Reminder: pass distinguishing SUBSTRINGS to find_col/find_rows/cell. "
        "Multiple substrings to one helper = AND-narrow (the column/row must contain ALL of them). "
        "Do NOT pass multiple entities — call helpers once per entity.\n"
        "Write Python code only inside one ```python ... ``` block. No explanation."
    )
    raw = llm.complete(CODEGEN_SYSTEM, user_prompt, max_tokens=600)
    # Strict: prefer first ```python ... ``` block; else first ``` block
    m = re.search(r"```python\s*\n(.*?)\n```", raw, re.DOTALL)
    if not m:
        m = re.search(r"```\s*\n(.*?)\n```", raw, re.DOTALL)
    if m:
        return m.group(1).strip()
    # Fallback — only keep lines until the first triple-backtick if any
    code = raw.split("```", 1)[0].strip()
    return code


def direct_answer(llm: GroqLLM, query: str, table: ParsedTable) -> str:
    """단순 검색: LLM이 테이블에서 직접 답변."""
    table_text = table.to_text()
    user_prompt = f"Table:\n{table_text}\n\nQuestion: {query}"
    return llm.complete(DIRECT_ANSWER_SYSTEM, user_prompt, max_tokens=200)


# ══════════════════════════════════════════════════════════════════
#  6. SANDBOXED CODE EXECUTOR
# ══════════════════════════════════════════════════════════════════

def execute_code(code: str, table: ParsedTable, python_bin: str = "/home/user/T2/hart-table-retrieval/.venv/bin/python3") -> Tuple[bool, str, str]:
    """코드를 subprocess에서 안전하게 실행.

    Returns: (success, stdout, stderr)
    """
    csv_data = table.to_csv_string()

    # Wrap the user code with DataFrame loading + safe helpers
    wrapper = textwrap.dedent(f"""\
import pandas as pd
import math
import re
import io

csv_data = {json.dumps(csv_data)}
df = pd.read_csv(io.StringIO(csv_data))


def _norm(s):
    return str(s).lower().strip()


def _as_subs(x):
    if isinstance(x, (list, tuple)):
        return [_norm(s) for s in x]
    return [_norm(x)]


def find_col(*substrs):
    subs = [_norm(s) for s in substrs]
    cands = [c for c in df.columns if c != "row_header" and all(s in _norm(c) for s in subs)]
    if not cands:
        raise ValueError(f"no column matches {{subs}}; columns are: {{list(df.columns)}}")
    if len(cands) > 1:
        # Prefer shortest (most specific) name
        cands.sort(key=len)
    return cands[0]


def find_rows(*substrs):
    subs = [_norm(s) for s in substrs]
    mask = df["row_header"].apply(lambda v: all(s in _norm(v) for s in subs))
    return df.loc[mask]


def colnum(col):
    return pd.to_numeric(df[col], errors='coerce')


def cell(row_subs, col_subs):
    rsub = _as_subs(row_subs)
    csub = _as_subs(col_subs)
    rows = find_rows(*rsub)
    if len(rows) == 0:
        raise ValueError(f"no row matches {{rsub}}; sample row_headers: {{list(df['row_header'].head(10))}}")
    col = find_col(*csub)
    val = pd.to_numeric(rows[col], errors='coerce').dropna()
    if len(val) == 0:
        raise ValueError(f"row {{rsub}} × col {{col!r}} has no numeric value")
    return float(val.iloc[0])


# --- User generated code ---
{code}
""")

    # Write to temp file and execute
    # TODO(security): In production, use Docker/nsjail for isolation.
    # Current scope: only LLM-generated code runs here (no user input).
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, dir="/tmp") as f:
        f.write(wrapper)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            [python_bin, tmp_path],
            capture_output=True, text=True,
            timeout=CODE_EXEC_TIMEOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return (proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip())
    except subprocess.TimeoutExpired:
        return (False, "", "TIMEOUT")
    finally:
        os.unlink(tmp_path)


# ══════════════════════════════════════════════════════════════════
#  7. METRICS (from existing eval)
# ══════════════════════════════════════════════════════════════════

_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+\.?\d*")
_OP_RE = re.compile(r"[+\-*/]")

def _to_nums(s):
    if isinstance(s, (int, float)): return [float(s)]
    if isinstance(s, list):
        out = []
        for x in s: out.extend(_to_nums(x))
        return out
    if s is None: return []
    out = []
    for m in _NUM_RE.findall(str(s)):
        try: out.append(float(m.replace(",","")))
        except: pass
    return out

def numeric_match(pred, gold, rel_tol=0.02) -> bool:
    if pred is None: return False
    pred_s = str(pred).strip().lower()
    g_nums = _to_nums(gold)
    p_nums = _to_nums(pred)
    if g_nums:
        p_variants = [
            {round(x, 2) for x in p_nums},
            {round(x * 100, 2) for x in p_nums},
            {round(x / 100, 4) for x in p_nums},
            {round(abs(x), 2) for x in p_nums},
        ]
        for g in g_nums:
            g_cands = [round(g, 2), round(g * 100, 2), round(g / 100, 4), round(abs(g), 2)]
            ok = False
            for gc in g_cands:
                for pv in p_variants:
                    if gc in pv: ok = True; break
                    for pn in pv:
                        if abs(pn - gc) / max(abs(gc), 1e-9) < rel_tol:
                            ok = True; break
                    if ok: break
                if ok: break
            if not ok: return False
        return True
    for gs in (str(x).strip().lower() for x in (gold if isinstance(gold, list) else [gold]) if x):
        if gs in pred_s or pred_s in gs: return True
    return False

def difficulty_class(sample: dict) -> str:
    agg = tuple(sorted(set(sample.get("aggregation") or ["none"])))
    fs = sample.get("answer_formulas") or []
    ops = max((len(_OP_RE.findall(f.lstrip("="))) for f in fs), default=0) if fs else 0
    if ops >= 2: return "multi_op_formula"
    if any(a in agg for a in ("div","sum","diff","average","range")): return "arithmetic_agg"
    if any(a in agg for a in ("pair-argmax","pair-argmin","topk-argmax","topk-argmin","kth-argmax")): return "pair_or_topk_arg"
    if any(a in agg for a in ("argmax","argmin","max","min")): return "single_arg"
    if any(a in agg for a in ("greater_than","less_than","opposite","counta")): return "comparison_or_count"
    if ops == 1: return "single_op_formula"
    return "simple_lookup"


# ══════════════════════════════════════════════════════════════════
#  8. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════

def run_pipeline(per_class: int = QUERIES_PER_CLASS, ablation: str = "adaptive",
                 out_name: Optional[str] = None, verbose: bool = True,
                 w_num: float = 0.4):
    """전체 평가 파이프라인.

    ablation:
      - "adaptive"            : 현재 시스템 (classify_query → 라우팅)
      - "always-codegen"      : 라우터 무시, 항상 VDB → codegen
      - "gold-table-codegen"  : retrieval 스킵, gold table 강제 + 항상 codegen (천장)
      - "always-direct"       : 라우터 무시, 항상 VDB → LLM 직접 답변
      - "always-original"     : VDB 안 씀, 항상 원본 구조검색(헤더+숫자) → codegen (원본-only 주장)
      - "always-keyword"      : VDB 안 씀, 항상 순수 어휘검색 → codegen (구조/숫자 신호 없는 baseline)
    """
    import random
    rng = random.Random(SEED)

    print("=" * 70)
    print(f"  VDB/OriginalDB 평가  [ablation={ablation}, per_class={per_class}]")
    print("=" * 70)

    # 1. Load data
    print("\n[1/4] Loading HiTab dev samples...")
    samples = load_samples("dev")
    print(f"  → {len(samples)} samples loaded")

    # 2. Build OriginalDB
    print("\n[2/4] Building OriginalDB...")
    orig_db = OriginalDB()
    table_cache = {}
    for s in samples:
        tid = get_table_id(s)
        if tid not in table_cache:
            raw = load_table(tid)
            if raw and "data" in raw:
                table_cache[tid] = raw
                orig_db.add(raw)
    print(f"  → {len(orig_db)} tables in OriginalDB")

    # 3. Try VDB (원본-only ablation에서는 불필요 → 스킵)
    if ablation in ("always-original", "always-keyword"):
        print(f"\n[3/4] Skipping VectorDB (ablation={ablation}, 원본만 검색)")
        vdb = None
    else:
        print("\n[3/4] Loading VectorDB...")
        vdb = VectorDB(CHROMA_DIR, device="cpu")

    # 4. LLM
    print(f"\n[4/4] Loading LLM (backend={LLM_BACKEND})...")
    llm = _make_llm()
    print("  → Connected")

    # Select diverse test queries
    HARD_CLASSES = [
        "multi_op_formula", "arithmetic_agg", "pair_or_topk_arg",
        "single_arg", "comparison_or_count",
    ]
    buckets = defaultdict(list)
    for s in samples:
        cls = difficulty_class(s)
        if cls in HARD_CLASSES:
            buckets[cls].append(s)

    chosen = []
    for cls in HARD_CLASSES:
        bucket = buckets.get(cls, [])
        rng.shuffle(bucket)
        for s in bucket[:per_class]:
            chosen.append((cls, s))

    print(f"\n{'='*70}")
    print(f"  Running {len(chosen)} queries ({per_class} per class) [{ablation}]")
    print(f"{'='*70}")

    results = []
    class_stats = defaultdict(lambda: {"n": 0, "correct": 0, "code_gen": 0, "code_exec_ok": 0})
    # 공정 비교용 후보군: 모든 검색기(VDB/structural/keyword)가 이 동일 집합 위에서 랭킹.
    pool = set(orig_db._tables)

    for i, (cls, sample) in enumerate(chosen, 1):
        query = get_query(sample)
        gold_tid = get_table_id(sample)
        gold_ans = get_answer(sample)
        gold_formula = (sample.get("answer_formulas") or [""])[0]

        if verbose:
            print(f"\n{'─'*70}")
            print(f"[{i}/{len(chosen)}] Class: {cls}")
            print(f"  Query:  {query}")
            print(f"  Gold:   answer={gold_ans}  formula={gold_formula}")

        # ── Step A: Classify query (ablation 분기) ──
        if ablation == "always-codegen":
            route = QueryRoute("vdb_codegen", True, f"ablation={ablation}")
        elif ablation == "always-direct":
            route = QueryRoute("vdb_codegen", False, f"ablation={ablation}")
        elif ablation == "gold-table-codegen":
            route = QueryRoute("codegen", True, f"ablation={ablation}")
        elif ablation == "always-original":
            route = QueryRoute("original_codegen", True, f"ablation={ablation}")
        elif ablation == "always-keyword":
            route = QueryRoute("keyword_codegen", True, f"ablation={ablation}")
        else:  # adaptive
            route = classify_query(query)
        if verbose: print(f"  Route:  {route.route} ({route.reason})")

        # ── Step B: Find table ──
        found_table = None
        search_method = ""
        gold_rank = None              # gold 테이블의 1-indexed 랭크 (top-K 밖이면 None)
        retrieval_attempted = True    # oracle ablation에서만 False
        TOPK = 10                     # nDCG@10 까지 보려고 10개 랭킹

        if ablation == "gold-table-codegen":
            # retrieval 스킵, gold table 직접 사용 → 검색 지표 집계에서 제외
            retrieval_attempted = False
            found_table = orig_db.get(gold_tid)
            search_method = "gold_table (oracle)"
            if found_table is None and verbose:
                print(f"  ⚠ Gold table {gold_tid} not in OriginalDB")
        else:
            # 공정 비교: 모든 검색기가 동일한 후보군(dev 테이블 전체) 위에서 랭킹.
            if route.route == "original_codegen":
                hits = orig_db.structural_search(query, top_k=TOPK, w_num=w_num)
                retriever = "structural"
            elif route.route == "keyword_codegen":
                hits = orig_db.keyword_search(query, top_k=TOPK)
                retriever = "keyword"
            elif route.route == "direct_lookup":
                hits = orig_db.keyword_search(query, top_k=TOPK)
                retriever = "keyword"
            else:
                hits = vdb.search(query, top_k=TOPK, allowed_ids=pool) if vdb else []
                retriever = "VDB"
                if not hits:  # VDB 실패 → 원본 키워드 폴백
                    hits = orig_db.keyword_search(query, top_k=TOPK)
                    retriever = "keyword-fallback"

            if hits:
                found_table = orig_db.get(hits[0][0])
                search_method = f"{retriever} (score={hits[0][1]:.3f})"
                hit_tids = [h[0] for h in hits]
                if gold_tid in hit_tids:
                    gold_rank = hit_tids.index(gold_tid) + 1
                if verbose:
                    if gold_rank:
                        print(f"  Search: ✓ Gold table at rank {gold_rank} ({retriever})")
                    else:
                        print(f"  Search: ✗ Gold table NOT in top-{TOPK} ({retriever})")

        if found_table is None:
            if verbose: print(f"  ⚠ No table found, skipping")
            results.append({"class": cls, "query": query, "answer": None,
                            "correct": False, "error": "no_table",
                            "gold_rank": None, "retrieval_attempted": retrieval_attempted})
            class_stats[cls]["n"] += 1
            continue

        if verbose: print(f"  Table:  {found_table.table_id} ({search_method})")

        # ── Step C: Generate answer ──
        t0 = time.time()

        if route.needs_code:
            # 코드 생성 경로
            code = generate_code(llm, query, found_table)
            class_stats[cls]["code_gen"] += 1
            if verbose:
                print(f"  Code Generated:")
                for line in code.split("\n"):
                    print(f"    │ {line}")

            # 실행
            ok, stdout, stderr = execute_code(code, found_table)
            elapsed = time.time() - t0

            if ok and stdout:
                pred = stdout.strip().split("\n")[-1]  # last line = result
                class_stats[cls]["code_exec_ok"] += 1
                if verbose: print(f"  Exec:   ✓ result = {pred!r}")
            else:
                pred = ""
                if verbose:
                    print(f"  Exec:   ✗ FAILED")
                    if stderr:
                        for line in stderr.split("\n")[-3:]:
                            print(f"    │ {line}")

                # 실패시 LLM 직접 답변 fallback
                if verbose: print(f"  Fallback: LLM direct answer...")
                pred = direct_answer(llm, query, found_table)
                elapsed = time.time() - t0
                if verbose: print(f"  Direct: {pred!r}")
        else:
            # 단순 검색 → LLM 직접 답변
            pred = direct_answer(llm, query, found_table)
            elapsed = time.time() - t0
            code = None
            if verbose: print(f"  Direct: {pred!r}")

        # ── Step D: Evaluate ──
        nm = numeric_match(pred, gold_ans)
        class_stats[cls]["n"] += 1
        class_stats[cls]["correct"] += int(nm)
        if verbose:
            print(f"  NM:     {'✓ CORRECT' if nm else '✗ WRONG'}")
            print(f"  Time:   {elapsed:.1f}s")

        results.append({
            "class": cls, "query": query, "gold_table": gold_tid,
            "gold_answer": gold_ans, "gold_formula": gold_formula,
            "route": route.route, "route_reason": route.reason,
            "search_method": search_method,
            "found_table": found_table.table_id,
            "table_correct": found_table.table_id == gold_tid,
            "gold_rank": gold_rank, "retrieval_attempted": retrieval_attempted,
            "code": code, "answer": pred,
            "correct": nm, "elapsed_s": round(elapsed, 2),
        })
        # Rate limit (Groq TPM 보호)
        time.sleep(1.5)

    # ══════════════════════════════════════════════════════════════
    #  SUMMARY
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*70}\n")

    print(f"{'Class':26s} {'n':>3s}  {'NM':>5s}  {'CodeGen':>7s}  {'ExecOK':>6s}")
    print(f"{'─'*26} {'─'*3}  {'─'*5}  {'─'*7}  {'─'*6}")
    total_n = total_correct = 0
    for cls in HARD_CLASSES:
        st = class_stats[cls]
        n = st["n"]
        if n == 0: continue
        nm_rate = st["correct"] / n
        cg = st["code_gen"]
        ex = st["code_exec_ok"]
        print(f"{cls:26s} {n:3d}  {nm_rate:5.3f}  {cg:7d}  {ex:6d}")
        total_n += n
        total_correct += st["correct"]

    nm_rate = (total_correct / total_n) if total_n else 0.0
    ci_lo, ci_hi = _bootstrap_ci([r["correct"] for r in results if "correct" in r], B=2000)
    if total_n:
        print(f"{'─'*26} {'─'*3}  {'─'*5}")
        print(f"{'OVERALL':26s} {total_n:3d}  {nm_rate:5.3f}   95% CI [{ci_lo:.3f}, {ci_hi:.3f}]")

    # ── 검색 지표 (retrieval) — NM보다 검정력이 높고 '검색이 더 정확한가'를 직접 측정 ──
    retr_rows = [r for r in results if r.get("retrieval_attempted")]
    retrieval_overall = _retrieval_metrics([r.get("gold_rank") for r in retr_rows])
    retrieval_by_class = {}
    if retr_rows:
        print(f"\n{'Retrieval':26s} {'n':>3s}  {'R@1':>5s}  {'R@5':>5s}  {'MRR':>5s}  {'nDCG10':>6s}")
        print(f"{'─'*26} {'─'*3}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
        for cls in HARD_CLASSES:
            cls_ranks = [r.get("gold_rank") for r in retr_rows if r.get("class") == cls]
            if not cls_ranks:
                continue
            m = _retrieval_metrics(cls_ranks)
            retrieval_by_class[cls] = m
            print(f"{cls:26s} {m['n']:3d}  {m['R@1']:5.3f}  {m['R@5']:5.3f}  {m['MRR']:5.3f}  {m['nDCG@10']:6.3f}")
        mo = retrieval_overall
        print(f"{'─'*26} {'─'*3}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*6}")
        print(f"{'OVERALL':26s} {mo['n']:3d}  {mo['R@1']:5.3f}  {mo['R@5']:5.3f}  {mo['MRR']:5.3f}  {mo['nDCG@10']:6.3f}")

    # Save results
    fname = out_name or f"codegen_eval_{ablation}_n{total_n}.json"
    out_path = Path(__file__).parent.parent / "results" / fname
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "config": {"model": GROQ_MODEL, "seed": SEED, "per_class": per_class,
                       "ablation": ablation, "w_num": w_num},
            "class_stats": {k: dict(v) for k, v in class_stats.items()},
            "overall": {"n": total_n, "correct": total_correct,
                       "nm_rate": nm_rate, "ci95": [ci_lo, ci_hi]},
            "retrieval": {"overall": retrieval_overall, "by_class": retrieval_by_class},
            "rows": results,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\nResults saved to {out_path}")

    return {"ablation": ablation, "n": total_n, "correct": total_correct,
            "nm_rate": nm_rate, "ci95": (ci_lo, ci_hi),
            "class_stats": {k: dict(v) for k, v in class_stats.items()},
            "rows": results, "out_path": str(out_path)}


def _bootstrap_ci(correct_flags: list, B: int = 2000, alpha: float = 0.05) -> Tuple[float, float]:
    """Percentile bootstrap 95% CI for mean of 0/1 array."""
    import random as _r
    arr = [1 if x else 0 for x in correct_flags]
    n = len(arr)
    if n == 0: return (0.0, 0.0)
    rng = _r.Random(0)
    means = []
    for _ in range(B):
        s = 0
        for _i in range(n):
            s += arr[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(B * alpha / 2)]
    hi = means[int(B * (1 - alpha / 2))]
    return (lo, hi)


def _retrieval_metrics(ranks: list) -> dict:
    """검색 지표 (단일 gold 테이블, binary relevance).

    ranks: gold 테이블의 1-indexed 랭크 리스트. top-K 밖이거나 미검색이면 None.
      R@1   = gold가 1위인 비율
      R@5   = gold가 top-5 안에 든 비율
      MRR   = mean(1/rank)
      nDCG@10 = mean(1/log2(rank+1)) for rank<=10  (gold 1개이므로 IDCG=1)
    """
    import math
    n = len(ranks)
    if n == 0:
        return {"n": 0, "R@1": 0.0, "R@5": 0.0, "MRR": 0.0, "nDCG@10": 0.0}
    return {
        "n": n,
        "R@1": sum(1 for r in ranks if r == 1) / n,
        "R@5": sum(1 for r in ranks if r and r <= 5) / n,
        "MRR": sum((1.0 / r) for r in ranks if r) / n,
        "nDCG@10": sum((1.0 / math.log2(r + 1)) for r in ranks if r and r <= 10) / n,
    }


def ask_one(query: str, *, table_id: Optional[str] = None, verbose: bool = True) -> dict:
    """단건 쿼리 실행 — CLI/REPL 용.

    table_id 를 직접 주면 검색을 건너뛰고 그 테이블로만 답함.
    """
    # 1) 자원 로드 (지연 캐시)
    cache = ask_one.__dict__.setdefault("_cache", {})
    if "orig_db" not in cache:
        if verbose: print("[init] loading samples + building OriginalDB ...")
        samples = load_samples("dev")
        orig_db = OriginalDB()
        for s in samples:
            tid = get_table_id(s)
            if tid in orig_db._tables: continue
            raw = load_table(tid)
            if raw and "data" in raw:
                orig_db.add(raw)
        cache["orig_db"] = orig_db
        if verbose: print(f"[init]  → {len(orig_db)} tables")
    if "vdb" not in cache:
        if verbose: print("[init] loading VectorDB ...")
        cache["vdb"] = VectorDB(CHROMA_DIR, device="cpu")
    if "llm" not in cache:
        if verbose: print(f"[init] connecting Groq ({GROQ_MODEL}) ...")
        cache["llm"] = _make_llm()
    orig_db, vdb, llm = cache["orig_db"], cache["vdb"], cache["llm"]

    # 2) 라우팅
    route = classify_query(query)
    if verbose:
        print(f"\nQuery : {query}")
        print(f"Route : {route.route}  ({route.reason})")

    # 3) 테이블 결정
    found = None
    search_method = ""
    if table_id:
        found = orig_db.get(table_id)
        search_method = f"forced table_id={table_id}"
    elif route.route == "direct_lookup":
        hits = orig_db.keyword_search(query, top_k=5)
        if hits:
            found = orig_db.get(hits[0][0])
            search_method = f"OriginalDB keyword (score={hits[0][1]:.2f})"
    else:
        hits = vdb.search(query, top_k=5)
        if hits:
            found = orig_db.get(hits[0][0])
            search_method = f"VDB semantic (score={hits[0][1]:.3f})"
        if found is None:
            hits = orig_db.keyword_search(query, top_k=5)
            if hits:
                found = orig_db.get(hits[0][0])
                search_method = f"OriginalDB keyword fallback (score={hits[0][1]:.2f})"

    if found is None:
        if verbose: print("Table : (not found)")
        return {"query": query, "route": route.route, "answer": None, "error": "no_table"}

    if verbose:
        print(f"Table : {found.table_id}   [{search_method}]")
        print(f"Title : {found.title}")

    # 4) 답변 생성
    t0 = time.time()
    code = None
    if route.needs_code:
        code = generate_code(llm, query, found)
        if verbose:
            print("Code  :")
            for line in code.split("\n"):
                print(f"  │ {line}")
        ok, stdout, stderr = execute_code(code, found)
        if ok and stdout:
            answer = stdout.strip().split("\n")[-1]
            if verbose: print(f"Exec  : OK → {answer!r}")
        else:
            if verbose:
                print("Exec  : FAILED")
                if stderr:
                    for line in stderr.split("\n")[-3:]:
                        print(f"  │ {line}")
                print("Fallback to LLM direct answer ...")
            answer = direct_answer(llm, query, found)
            if verbose: print(f"Direct: {answer!r}")
    else:
        answer = direct_answer(llm, query, found)
        if verbose: print(f"Direct: {answer!r}")

    elapsed = time.time() - t0
    if verbose: print(f"Time  : {elapsed:.1f}s")

    return {
        "query": query, "route": route.route, "route_reason": route.reason,
        "table": found.table_id, "search": search_method,
        "code": code, "answer": answer, "elapsed_s": round(elapsed, 2),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VDB/OriginalDB 라우팅 + LLM 코드생성 QA")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="단건 쿼리. 주지 않으면 전체 평가 파이프라인을 실행.")
    parser.add_argument("--table", "-t", type=str, default=None,
                        help="테이블 ID 강제 지정 (검색 스킵)")
    parser.add_argument("--repl", action="store_true",
                        help="대화형 REPL 모드")
    parser.add_argument("--per-class", type=int, default=QUERIES_PER_CLASS,
                        help="클래스별 샘플 수 (기본 3)")
    parser.add_argument("--ablation", type=str, default="adaptive",
                        choices=["adaptive", "always-codegen", "gold-table-codegen", "always-direct", "always-original", "always-keyword"],
                        help="평가 모드")
    parser.add_argument("--quiet", action="store_true",
                        help="per-query 출력 줄이기 (긴 실행시 사용)")
    parser.add_argument("--w-num", type=float, default=0.4,
                        help="structural_search 숫자-셀 가중치 (always-original 전용). "
                             "0으로 주면 헤더 토큰만 사용 → 숫자 누수 분해용.")
    parser.add_argument("--out", type=str, default=None,
                        help="결과 JSON 파일명 (results/ 하위)")
    args = parser.parse_args()

    if args.repl:
        print("REPL 모드 — 빈 줄/exit 입력시 종료")
        while True:
            try:
                q = input("\n>>> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q or q.lower() in ("exit", "quit"):
                break
            try:
                ask_one(q)
            except Exception as e:
                print(f"ERROR: {e}")
    elif args.query:
        ask_one(args.query, table_id=args.table)
    else:
        run_pipeline(per_class=args.per_class, ablation=args.ablation,
                     verbose=not args.quiet, w_num=args.w_num, out_name=args.out)
