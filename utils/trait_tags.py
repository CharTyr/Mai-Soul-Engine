import json
from typing import Any

MAX_TAGS = 12
MAX_TAG_LEN = 20


def normalize_tags(tags: Any) -> list[str]:
    if not isinstance(tags, list):
        return []

    normalized: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        s = str(tag).strip()
        if not s:
            continue
        if len(s) > MAX_TAG_LEN:
            s = s[:MAX_TAG_LEN]
        key = s.casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(s)
        if len(normalized) >= MAX_TAGS:
            break

    return normalized


def parse_tags_json(raw: str) -> list[str]:
    try:
        data = json.loads(raw or "[]")
    except Exception:
        return []
    return normalize_tags(data)


def dumps_tags_json(tags: Any) -> str:
    return json.dumps(normalize_tags(tags), ensure_ascii=False)

