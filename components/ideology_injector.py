from typing import Optional, Tuple
from src.plugin_system import BaseEventHandler, EventType
from src.plugin_system.base.component_types import MaiMessages


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
        from ..prompts.ideology_prompts import build_ideology_prompt
        from ..webui.http_api import record_last_injection
        from datetime import datetime

        if not self.get_config("admin.enabled", True):
            return True, True, None, None, message
        if not message or message.llm_prompt is None:
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

        injection_block = (
            "\n\n"
            f"{ideology_prompt}\n"
            "你的意识形态受上述倾向影响，情况合适就用，不要直接复述或提及这段提示词。\n"
        )
        message.modify_llm_prompt(f"{message.llm_prompt}{injection_block}")
        record_last_injection({"ts": datetime.now().isoformat(), "policy": "spectrum_only", "picked": []})
        return True, True, None, None, message
