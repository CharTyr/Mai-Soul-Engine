from typing import Optional, Tuple
from src.plugin_system import BaseEventHandler, EventType
from src.common.data_models.mai_messages import MaiMessages


class IdeologyInjector(BaseEventHandler):
    event_type = EventType.POST_LLM
    handler_name = "ideology_injector"
    handler_description = "注入意识形态提示词到回复生成"
    weight = 10
    intercept_message = False

    async def execute(
        self, message: MaiMessages | None
    ) -> Tuple[bool, bool, Optional[str], Optional[dict], Optional[MaiMessages]]:
        from ..models.ideology_model import get_or_create_spectrum, init_tables
        from ..prompts.ideology_prompts import build_ideology_prompt

        if not self.get_config("enabled", True):
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

        thresholds = {
            "threshold_mild": self.get_config("threshold_mild", 25),
            "threshold_moderate": self.get_config("threshold_moderate", 50),
            "threshold_extreme": self.get_config("threshold_extreme", 75),
        }

        custom_prompts = self.get_config("custom_prompts", {})
        ideology_prompt = build_ideology_prompt(spectrum_dict, thresholds, custom_prompts)

        if not ideology_prompt:
            return True, True, None, None, message

        extra_info = {"ideology_prompt": ideology_prompt}
        return True, True, None, extra_info, message
