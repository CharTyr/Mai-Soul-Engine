import json
from typing import Any


def normalize_evidence(evidence: Any, *, max_items: int = 8, max_len: int = 160) -> list[str]:
    if not isinstance(max_items, int) or max_items <= 0:
        max_items = 8
    if not isinstance(max_len, int) or max_len <= 0:
        max_len = 160

    if evidence is None:
        return []
    if not isinstance(evidence, list):
        evidence = [evidence]

    out: list[str] = []
    seen: set[str] = set()
    for item in evidence:
        s = str(item).replace("\n", " ").replace("\r", " ").strip()
        while "  " in s:
            s = s.replace("  ", " ")
        if not s:
            continue
        if len(s) > max_len:
            s = f"{s[:max_len]}..."
        key = s.casefold()
        if key in seen:
            continue
        out.append(s)
        seen.add(key)
        if len(out) >= max_items:
            break
    return out


def parse_evidence_json(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except Exception:
        data = []
    return normalize_evidence(data)


def dumps_evidence_json(evidence: Any) -> str:
    normalized = normalize_evidence(evidence, max_items=64)
    return json.dumps(normalized, ensure_ascii=False)


def merge_evidence(existing: Any, incoming: Any, *, max_items: int = 12) -> list[str]:
    base = normalize_evidence(existing, max_items=max_items)
    extra = normalize_evidence(incoming, max_items=max_items)
    return normalize_evidence(base + extra, max_items=max_items)

