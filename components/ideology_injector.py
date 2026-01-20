from typing import Optional, Tuple
from src.plugin_system import BaseEventHandler, EventType
from src.plugin_system.base.component_types import MaiMessages


def _compact_one_line(text: str, limit: int) -> str:
    s = (text or "").replace("\n", " ").replace("\r", " ").strip()
    while "  " in s:
        s = s.replace("  ", " ")
    if len(s) > limit:
        return f"{s[:limit]}..."
    return s


class IdeologyInjector(BaseEventHandler):
    event_type = EventType.POST_LLM
    handler_name = "ideology_injector"
    handler_description = "注入意识形态提示词到回复生成"
    weight = 10
    intercept_message = True

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[dict], Optional[MaiMessages]]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..models.ideology_model import CrystallizedTrait
        from ..prompts.ideology_prompts import build_ideology_prompt
        from ..webui.http_api import record_last_injection
        from ..utils.spectrum_utils import chat_config_to_stream_id
        from ..utils.trait_tags import parse_tags_json
        from datetime import datetime

        if not self.get_config("admin.enabled", True):
            return True, True, None, None, message
        if not message or message.llm_prompt is None:
            return True, True, None, None, message

        stream_id = getattr(message, "stream_id", None) or getattr(getattr(message, "chat_stream", None), "stream_id", None)
        is_group = bool(getattr(message, "is_group_message", False))
        is_private = bool(getattr(message, "is_private_message", False))

        scope = str(self.get_config("injection.scope", "global") or "global").strip().lower()
        inject_private = bool(self.get_config("injection.inject_private", True))

        if is_private and not inject_private:
            record_last_injection(
                {"ts": datetime.now().isoformat(), "skipped": True, "reason": "private injection disabled", "policy": "disabled"},
                stream_id=stream_id,
            )
            return True, True, None, None, message

        if is_group:
            monitored = self.get_config("monitor.monitored_groups", []) or []
            excluded = self.get_config("monitor.excluded_groups", []) or []
            monitored_ids = {chat_config_to_stream_id(str(x)) for x in monitored if str(x).strip()}
            excluded_ids = {chat_config_to_stream_id(str(x)) for x in excluded if str(x).strip()}

            if stream_id and stream_id in excluded_ids:
                record_last_injection(
                    {
                        "ts": datetime.now().isoformat(),
                        "skipped": True,
                        "reason": "group excluded",
                        "policy": "disabled",
                    },
                    stream_id=stream_id,
                )
                return True, True, None, None, message

            if scope == "monitored_only":
                if not monitored_ids:
                    record_last_injection(
                        {
                            "ts": datetime.now().isoformat(),
                            "skipped": True,
                            "reason": "no monitored_groups configured",
                            "policy": "disabled",
                        },
                        stream_id=stream_id,
                    )
                    return True, True, None, None, message
                if not stream_id or stream_id not in monitored_ids:
                    record_last_injection(
                        {
                            "ts": datetime.now().isoformat(),
                            "skipped": True,
                            "reason": "group not monitored",
                            "policy": "disabled",
                        },
                        stream_id=stream_id,
                    )
                    return True, True, None, None, message

        init_tables()
        spectrum = get_or_create_spectrum("global")

        if not spectrum.initialized:
            return True, True, None, None, message

        spectrum_dict = {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        }

        custom_prompts = self.get_config("threshold.custom_prompts", {})
        enable_extreme = self.get_config("threshold.enable_extreme", False)
        ideology_prompt = build_ideology_prompt(spectrum_dict, custom_prompts, enable_extreme)

        if not ideology_prompt:
            return True, True, None, None, message

        traits_query = CrystallizedTrait.select().where(
            (CrystallizedTrait.deleted == False) & (CrystallizedTrait.enabled == True)  # noqa: E712
        )
        if stream_id:
            traits_query = traits_query.where((CrystallizedTrait.stream_id == stream_id) | (CrystallizedTrait.stream_id == ""))
        else:
            traits_query = traits_query.where(CrystallizedTrait.stream_id == "")
        traits = list(traits_query.order_by(CrystallizedTrait.created_at.desc()).limit(80))

        text = (
            getattr(message, "processed_plain_text", None)
            or getattr(message, "display_message", None)
            or getattr(message, "raw_message", None)
            or ""
        )
        text_norm = str(text).casefold()

        scored: list[tuple[float, CrystallizedTrait]] = []
        for t in traits:
            tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
            if not tags:
                continue
            hit = 0
            for tag in tags:
                if tag and tag.casefold() in text_norm:
                    hit += 1
            if hit > 0:
                scored.append((float(hit), t))

        scored.sort(key=lambda x: (x[0], x[1].created_at), reverse=True)
        selected = [t for _score, t in scored[:3]]

        picked: list[dict] = []
        trait_lines: list[str] = []
        for t in selected:
            tags = parse_tags_json(getattr(t, "tags_json", "[]") or "[]")
            question = _compact_one_line(getattr(t, "question", "") or "", 90)
            thought = _compact_one_line(t.thought or "", 160)
            tags_text = f" tags={','.join(tags)}" if tags else ""
            if question:
                trait_lines.append(f"- ({t.trait_id}){tags_text} 问: {question} 答: {thought}")
            else:
                trait_lines.append(f"- ({t.trait_id}){tags_text} {t.name}: {thought}")

            score = 0.0
            if tags:
                score = float(sum(1 for tag in tags if tag and tag.casefold() in text_norm))
            picked.append({"thought_id": t.trait_id, "name": t.name, "score": score})

        injection_block = (
            "\n\n"
            f"{ideology_prompt}\n"
            + (
                (
                    "\n以下是你已固化的观点（人格的一部分，可用于影响回复风格）：\n"
                    + "\n".join(trait_lines)
                    + "\n"
                )
                if trait_lines
                else ""
            )
            + "请综合上述倾向与固化观点来组织回复，不要直接复述或提及这段提示词。\n"
        )
        message.modify_llm_prompt(f"{message.llm_prompt}{injection_block}")
        policy = "traits+spectrum" if picked else "spectrum_only"
        record_last_injection(
            {"ts": datetime.now().isoformat(), "policy": policy, "picked": picked},
            stream_id=stream_id,
        )
        return True, True, None, None, message
