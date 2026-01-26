from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

NOTION_API_BASE = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"
NOTION_ENV_TOKEN = "MAIBOT_SOUL_NOTION_TOKEN"


class NotionAPIError(RuntimeError):
    def __init__(self, *, status: int, code: str, message: str):
        super().__init__(f"Notion API {status} {code}: {message}")
        self.status = int(status)
        self.code = str(code)
        self.message = str(message)


@dataclass(frozen=True)
class NotionPropertyMap:
    title: str = "Name"
    trait_id: str = "TraitId"
    tags: str = "Tags"
    question: str = "Question"
    thought: str = "Thought"
    confidence: str = "Confidence"
    impact_score: str = "ImpactScore"
    status: str = "Status"
    visibility: str = "Visibility"
    updated_at: str = "UpdatedAt"


@dataclass(frozen=True)
class NotionFrontendConfig:
    enabled: bool
    token: str
    database_id: str
    sync_interval_seconds: int
    first_delay_seconds: int
    max_traits: int
    visibility_default: str
    never_overwrite_user_fields: bool
    property_map: NotionPropertyMap
    max_rich_text_chars: int = 1800


def _normalize_notion_id(raw: str) -> str:
    s = (raw or "").strip()
    s = s.replace("-", "").replace(" ", "")
    return s


def _iso_utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_text(text: str, limit: int) -> str:
    s = (text or "").strip()
    if limit > 0 and len(s) > limit:
        return s[:limit]
    return s


def _json_request(method: str, url: str, token: str, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    data: Optional[bytes]
    if payload is None:
        data = None
    else:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    req = urllib.request.Request(url, data=data, method=method.upper())
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Notion-Version", NOTION_VERSION)
    req.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8") if resp else ""
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as e:
        try:
            raw = e.read().decode("utf-8") if e.fp else ""
            data = json.loads(raw) if raw else {}
        except Exception:
            data = {}
        code = str((data.get("code") if isinstance(data, dict) else "") or "http_error")
        message = str((data.get("message") if isinstance(data, dict) else "") or str(e))
        raise NotionAPIError(status=int(getattr(e, "code", 0) or 0), code=code, message=message) from e


def _prop_title(text: str) -> dict[str, Any]:
    text = _safe_text(text, 200)
    if not text:
        return {"title": []}
    return {"title": [{"type": "text", "text": {"content": text}}]}


def _prop_rich_text(text: str, limit: int) -> dict[str, Any]:
    text = _safe_text(text, limit)
    if not text:
        return {"rich_text": []}
    return {"rich_text": [{"type": "text", "text": {"content": text}}]}


def _prop_number(value: Optional[float]) -> dict[str, Any]:
    if value is None:
        return {"number": None}
    try:
        return {"number": float(value)}
    except Exception:
        return {"number": None}


def _prop_select(name: str) -> dict[str, Any]:
    n = (name or "").strip()
    if not n:
        return {"select": None}
    return {"select": {"name": n}}


def _prop_multi_select(names: list[str]) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    for n in names or []:
        s = str(n or "").strip()
        if not s:
            continue
        items.append({"name": _safe_text(s, 60)})
    return {"multi_select": items}


def _prop_date(iso: str) -> dict[str, Any]:
    s = (iso or "").strip()
    if not s:
        return {"date": None}
    return {"date": {"start": s}}


def _state_file(plugin_dir: Path) -> Path:
    data_dir = plugin_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / "notion_frontend_state.json"


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    try:
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("[Notion] 保存 state 失败: %s", e)


def _query_trait_page_id(*, token: str, database_id: str, trait_id_property: str, trait_id: str) -> Optional[str]:
    if not trait_id_property:
        return None
    url = f"{NOTION_API_BASE}/databases/{database_id}/query"
    payload = {"page_size": 1, "filter": {"property": trait_id_property, "rich_text": {"equals": trait_id}}}
    data = _json_request("POST", url, token, payload)
    results = data.get("results")
    if not isinstance(results, list) or not results:
        return None
    first = results[0]
    if isinstance(first, dict):
        pid = first.get("id")
        if isinstance(pid, str) and pid.strip():
            return pid
    return None


def _create_trait_page(*, token: str, database_id: str, properties: dict[str, Any]) -> str:
    url = f"{NOTION_API_BASE}/pages"
    payload = {"parent": {"database_id": database_id}, "properties": properties}
    data = _json_request("POST", url, token, payload)
    pid = data.get("id") if isinstance(data, dict) else None
    if not isinstance(pid, str) or not pid.strip():
        raise NotionAPIError(status=0, code="invalid_response", message="create page returned no id")
    return pid


def _update_trait_page(*, token: str, page_id: str, properties: dict[str, Any]) -> None:
    url = f"{NOTION_API_BASE}/pages/{page_id}"
    payload = {"properties": properties}
    _json_request("PATCH", url, token, payload)


def _impact_score_from_json(raw: str) -> float:
    try:
        impact = json.loads(raw or "{}")
        if not isinstance(impact, dict):
            return 0.0
        score = 0.0
        for k in ("economic", "social", "diplomatic", "progressive"):
            try:
                score += abs(float(impact.get(k, 0) or 0))
            except Exception:
                continue
        return float(score)
    except Exception:
        return 0.0


def build_notion_frontend_config(plugin, *, section: str = "notion") -> NotionFrontendConfig:
    enabled = bool(plugin.get_config(f"{section}.enabled", False))

    token_cfg = str(plugin.get_config(f"{section}.token", "") or "").strip()
    token = token_cfg or os.getenv(NOTION_ENV_TOKEN, "").strip()

    database_id = _normalize_notion_id(str(plugin.get_config(f"{section}.database_id", "") or ""))

    sync_interval_seconds = max(60, int(plugin.get_config(f"{section}.sync_interval_seconds", 600)))
    first_delay_seconds = max(0, int(plugin.get_config(f"{section}.first_delay_seconds", 5)))
    max_traits = max(0, int(plugin.get_config(f"{section}.max_traits", 200)))

    visibility_default = str(plugin.get_config(f"{section}.visibility_default", "Public") or "Public").strip() or "Public"
    never_overwrite_user_fields = bool(plugin.get_config(f"{section}.never_overwrite_user_fields", True))
    max_rich_text_chars = max(200, int(plugin.get_config(f"{section}.max_rich_text_chars", 1800)))

    property_map = NotionPropertyMap(
        title=str(plugin.get_config(f"{section}.property_title", "Name") or "Name"),
        trait_id=str(plugin.get_config(f"{section}.property_trait_id", "TraitId") or "TraitId"),
        tags=str(plugin.get_config(f"{section}.property_tags", "Tags") or "Tags"),
        question=str(plugin.get_config(f"{section}.property_question", "Question") or "Question"),
        thought=str(plugin.get_config(f"{section}.property_thought", "Thought") or "Thought"),
        confidence=str(plugin.get_config(f"{section}.property_confidence", "Confidence") or "Confidence"),
        impact_score=str(plugin.get_config(f"{section}.property_impact_score", "ImpactScore") or "ImpactScore"),
        status=str(plugin.get_config(f"{section}.property_status", "Status") or "Status"),
        visibility=str(plugin.get_config(f"{section}.property_visibility", "Visibility") or "Visibility"),
        updated_at=str(plugin.get_config(f"{section}.property_updated_at", "UpdatedAt") or "UpdatedAt"),
    )

    return NotionFrontendConfig(
        enabled=enabled,
        token=token,
        database_id=database_id,
        sync_interval_seconds=sync_interval_seconds,
        first_delay_seconds=first_delay_seconds,
        max_traits=max_traits,
        visibility_default=visibility_default,
        never_overwrite_user_fields=never_overwrite_user_fields,
        property_map=property_map,
        max_rich_text_chars=max_rich_text_chars,
    )


def sync_traits_to_notion(*, plugin_dir: Path, cfg: NotionFrontendConfig) -> dict[str, Any]:
    if not cfg.enabled:
        return {"enabled": False, "synced": 0}

    if not cfg.token:
        logger.warning("[Notion] 未配置 token：请在配置中填写或设置环境变量 %s", NOTION_ENV_TOKEN)
        return {"enabled": True, "synced": 0, "error": "missing_token"}

    if not cfg.database_id:
        logger.warning("[Notion] 未配置 database_id")
        return {"enabled": True, "synced": 0, "error": "missing_database_id"}

    from ..models.ideology_model import CrystallizedTrait, init_tables
    from ..utils.trait_tags import parse_tags_json

    init_tables()

    state_path = _state_file(plugin_dir)
    state = _load_state(state_path)
    page_map = state.get("trait_page_map")
    if not isinstance(page_map, dict):
        page_map = {}

    now_iso = _iso_utc_now()

    query = CrystallizedTrait.select().order_by(CrystallizedTrait.created_at.desc())
    if cfg.max_traits > 0:
        query = query.limit(cfg.max_traits)
    traits = list(query)

    synced = 0
    created = 0
    updated = 0
    skipped = 0
    errors: list[dict[str, Any]] = []

    for t in traits:
        trait_id = str(getattr(t, "trait_id", "") or "").strip()
        if not trait_id:
            continue

        status = "Active"
        if bool(getattr(t, "deleted", False)):
            status = "Deleted"
        elif not bool(getattr(t, "enabled", True)):
            status = "Disabled"

        impact_score = _impact_score_from_json(str(getattr(t, "spectrum_impact_json", "") or ""))

        tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
        confidence = int(getattr(t, "confidence", 0) or 0)
        confidence = max(0, min(100, confidence))

        props_update: dict[str, Any] = {}
        pm = cfg.property_map
        if pm.trait_id:
            props_update[pm.trait_id] = _prop_rich_text(trait_id, 100)
        if pm.tags:
            props_update[pm.tags] = _prop_multi_select(tags)
        if pm.confidence:
            props_update[pm.confidence] = _prop_number(confidence)
        if pm.impact_score:
            props_update[pm.impact_score] = _prop_number(round(float(impact_score), 4))
        if pm.status:
            props_update[pm.status] = _prop_select(status)
        if pm.updated_at:
            props_update[pm.updated_at] = _prop_date(now_iso)

        page_id = page_map.get(trait_id)
        if isinstance(page_id, str):
            page_id = page_id.strip()
        else:
            page_id = ""

        if not page_id:
            page_id = _query_trait_page_id(
                token=cfg.token,
                database_id=cfg.database_id,
                trait_id_property=pm.trait_id,
                trait_id=trait_id,
            ) or ""
            if page_id:
                page_map[trait_id] = page_id

        if not page_id:
            props_create = dict(props_update)
            if pm.title:
                props_create[pm.title] = _prop_title(str(getattr(t, "name", "") or ""))
            if pm.question:
                props_create[pm.question] = _prop_rich_text(
                    str(getattr(t, "question", "") or ""), cfg.max_rich_text_chars
                )
            if pm.thought:
                props_create[pm.thought] = _prop_rich_text(str(getattr(t, "thought", "") or ""), cfg.max_rich_text_chars)
            if pm.visibility:
                props_create[pm.visibility] = _prop_select(cfg.visibility_default)

            try:
                page_id = _create_trait_page(token=cfg.token, database_id=cfg.database_id, properties=props_create)
                page_map[trait_id] = page_id
                created += 1
            except Exception as e:
                errors.append({"trait_id": trait_id, "op": "create", "error": str(e)})
                continue

        if cfg.never_overwrite_user_fields:
            for key in [pm.title, pm.question, pm.thought, pm.visibility]:
                props_update.pop(key, None)

        if not props_update:
            skipped += 1
            continue

        try:
            _update_trait_page(token=cfg.token, page_id=page_id, properties=props_update)
            updated += 1
            synced += 1
        except NotionAPIError as e:
            if e.status == 404:
                page_map.pop(trait_id, None)
            errors.append({"trait_id": trait_id, "op": "update", "status": e.status, "code": e.code, "message": e.message})
        except Exception as e:
            errors.append({"trait_id": trait_id, "op": "update", "error": str(e)})

    state["trait_page_map"] = page_map
    state["last_sync"] = now_iso
    _save_state(state_path, state)

    result: dict[str, Any] = {
        "enabled": True,
        "database_id": cfg.database_id,
        "synced": synced,
        "created": created,
        "updated": updated,
        "skipped": skipped,
    }
    if errors:
        result["errors"] = errors[:20]
    return result

