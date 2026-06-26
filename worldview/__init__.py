"""P1 三观生长：分层、切片、情绪辅助与注入摘要。"""

from .constants import (
    IDEOLOGY_LAYERS,
    LIFECYCLE_STATES,
    SPECTRUM_DIM_TO_LAYER,
    normalize_ideology_layer,
    normalize_lifecycle_state,
)
from .service import WorldviewService

__all__ = [
    "IDEOLOGY_LAYERS",
    "LIFECYCLE_STATES",
    "SPECTRUM_DIM_TO_LAYER",
    "WorldviewService",
    "normalize_ideology_layer",
    "normalize_lifecycle_state",
]