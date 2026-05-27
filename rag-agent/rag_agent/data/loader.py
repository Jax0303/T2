import json
import logging
import os
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = "data/hitab"


def _find_data_root(data_dir: str) -> Path:
    """Find HiTab root by checking common subdir structures."""
    p = Path(data_dir)
    if (p / "data" / "train_samples.jsonl").exists():
        return p
    if (p / "HiTab" / "data" / "train_samples.jsonl").exists():
        return p / "HiTab"
    if (p / "train_samples.jsonl").exists():
        return p.parent
    raise FileNotFoundError(
        f"HiTab data not found under {p}. Expected train_samples.jsonl in data/"
    )


def load_samples(
    data_dir: str = DEFAULT_DATA_DIR,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> List[dict]:
    """Load HiTab QA samples for the given split."""
    root = _find_data_root(data_dir)
    fname = f"{split}_samples.jsonl"
    fpath = root / "data" / fname
    if not fpath.exists():
        raise FileNotFoundError(f"{fpath} not found")

    samples = []
    with open(fpath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
            if max_samples and len(samples) >= max_samples:
                break

    logger.info("Loaded %d samples from %s", len(samples), fpath)
    return samples


def load_table(table_id: str, data_dir: str = DEFAULT_DATA_DIR) -> Optional[dict]:
    """Load a single table by table_id."""
    root = _find_data_root(data_dir)
    tables_dir = root / "data" / "tables"

    # Try multiple directory layouts: some HiTab downloads nest tables
    # under data/tables/{hmt,raw}/, others under data/tables/tables/{hmt,raw}/.
    search_roots = [tables_dir]
    nested = tables_dir / "tables"
    if nested.is_dir():
        search_roots.insert(0, nested)  # prefer nested if it exists

    for tdir in search_roots:
        for subdir in ["hmt", "raw"]:
            p = tdir / subdir / f"{table_id}.json"
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    table = json.load(f)
                table["table_id"] = table_id
                return table

    logger.warning("Table %s not found", table_id)
    return None


def load_hitab(
    data_dir: str = DEFAULT_DATA_DIR,
    split: str = "dev",
    max_samples: Optional[int] = None,
) -> List[dict]:
    """
    Load HiTab. Returns list of samples, each with a 'table' field attached.
    """
    samples = load_samples(data_dir, split, max_samples)

    # Attach tables
    table_cache: Dict[str, dict] = {}
    for s in samples:
        tid = s.get("table_id")
        if tid and tid not in table_cache:
            t = load_table(tid, data_dir)
            if t is not None:
                table_cache[tid] = t
        if tid in table_cache:
            s["table"] = table_cache[tid]

    return samples


def get_table_from_sample(sample: dict) -> dict:
    if "table" in sample:
        return sample["table"]
    return sample


def get_query_from_sample(sample: dict) -> str:
    for key in ["question", "sub_sentence", "query"]:
        v = sample.get(key)
        if v:
            return v
    return ""


def get_table_id(sample: dict) -> str:
    if "table_id" in sample:
        return sample["table_id"]
    table = get_table_from_sample(sample)
    if isinstance(table, dict):
        return table.get("table_id", table.get("uid", "unknown"))
    return sample.get("id", "unknown")


def get_answer(sample: dict):
    return sample.get("answer", [])


def print_sample_structure(sample: dict, max_depth: int = 3):
    def _describe(obj, depth=0, prefix=""):
        indent = "  " * depth
        if depth >= max_depth:
            print(f"{indent}{prefix}...")
            return
        if isinstance(obj, dict):
            print(f"{indent}{prefix}dict with keys: {list(obj.keys())[:10]}")
            for k, v in list(obj.items())[:10]:
                _describe(v, depth + 1, f"[{k}]: ")
        elif isinstance(obj, list):
            print(f"{indent}{prefix}list of length {len(obj)}")
            if obj:
                _describe(obj[0], depth + 1, "[0]: ")
        else:
            val_str = str(obj)
            if len(val_str) > 80:
                val_str = val_str[:80] + "..."
            print(f"{indent}{prefix}{type(obj).__name__} = {val_str}")

    _describe(sample)
