import json
from typing import Any


def _compact(text: str, limit: int) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) > limit:
        return f"{s[:limit]}..."
    return s


def _normalize_evidence_entry(entry: Any) -> dict[str, Any] | None:
    if entry is None:
        return None
    if isinstance(entry, str):
        s = _compact(entry, 220)
        return {"text": s} if s else None
    if not isinstance(entry, dict):
        return None

    out: dict[str, Any] = {}
    for k in ("seed_id", "stream_id", "type", "event", "reasoning", "created_at"):
        v = entry.get(k, "")
        v = _compact(str(v or ""), 400 if k in ("event", "reasoning") else 120)
        if v:
            out[k] = v

    for k in ("intensity", "seed_confidence", "internalization_confidence", "dedup_similarity"):
        if k in entry:
            try:
                out[k] = float(entry.get(k, 0.0) or 0.0)
            except Exception:
                out[k] = 0.0

    evidence = entry.get("evidence", []) or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []
    cleaned: list[str] = []
    seen = set()
    for item in evidence:
        s = _compact(str(item or ""), 220)
        if not s:
            continue
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= 5:
            break
    if cleaned:
        out["evidence"] = cleaned

    if not out:
        return None
    return out


def parse_evidence_json(raw: str) -> list[dict[str, Any]]:
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []

    if not isinstance(data, list):
        return []

    result: list[dict[str, Any]] = []
    for item in data:
        normalized = _normalize_evidence_entry(item)
        if normalized:
            result.append(normalized)
    return result


def dumps_evidence_json(entries: Any) -> str:
    if not isinstance(entries, list):
        entries = []
    normalized: list[dict[str, Any]] = []
    for item in entries:
        norm = _normalize_evidence_entry(item)
        if norm:
            normalized.append(norm)
    if len(normalized) > 50:
        normalized = normalized[-50:]
    return json.dumps(normalized, ensure_ascii=False)


def append_evidence_json(raw: str, entry: Any) -> str:
    current = parse_evidence_json(raw or "[]")
    norm = _normalize_evidence_entry(entry)
    if norm:
        current.append(norm)
    if len(current) > 50:
        current = current[-50:]
    return json.dumps(current, ensure_ascii=False)

