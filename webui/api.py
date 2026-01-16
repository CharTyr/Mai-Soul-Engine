from datetime import datetime
from typing import Optional


async def get_current_spectrum() -> dict:
    from ..models.ideology_model import get_or_create_spectrum, init_tables

    init_tables()
    spectrum = get_or_create_spectrum("global")

    return {
        "economic": spectrum.economic,
        "social": spectrum.social,
        "diplomatic": spectrum.diplomatic,
        "progressive": spectrum.progressive,
        "initialized": spectrum.initialized,
        "last_evolution": spectrum.last_evolution.isoformat() if spectrum.last_evolution else None,
        "updated_at": spectrum.updated_at.isoformat() if spectrum.updated_at else None,
    }


async def get_evolution_history(limit: int = 100) -> list:
    from ..models.ideology_model import EvolutionHistory, init_tables

    init_tables()
    records = EvolutionHistory.select().order_by(EvolutionHistory.timestamp.desc()).limit(limit)

    return [
        {
            "id": r.id,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "group_id": r.group_id,
            "deltas": {
                "economic": r.economic_delta,
                "social": r.social_delta,
                "diplomatic": r.diplomatic_delta,
                "progressive": r.progressive_delta,
            },
            "reason": r.reason,
        }
        for r in records
    ]


async def get_spectrum_chart_data(days: int = 30) -> dict:
    from ..models.ideology_model import EvolutionHistory, get_or_create_spectrum, init_tables
    from datetime import timedelta

    init_tables()
    cutoff = datetime.now() - timedelta(days=days)
    records = (
        EvolutionHistory.select().where(EvolutionHistory.timestamp >= cutoff).order_by(EvolutionHistory.timestamp.asc())
    )

    spectrum = get_or_create_spectrum("global")
    current = {
        "economic": spectrum.economic,
        "social": spectrum.social,
        "diplomatic": spectrum.diplomatic,
        "progressive": spectrum.progressive,
    }

    labels = []
    datasets = {
        "economic": [],
        "social": [],
        "diplomatic": [],
        "progressive": [],
    }

    for r in records:
        labels.append(r.timestamp.isoformat() if r.timestamp else "")
        for dim in datasets:
            delta = getattr(r, f"{dim}_delta", 0)
            datasets[dim].append(delta)

    return {
        "labels": labels,
        "datasets": datasets,
        "current": current,
    }


async def manual_evolution(group_id: str) -> dict:
    from ..components.evolution_task import EvolutionTaskHandler

    handler = EvolutionTaskHandler()
    evolution_rate = handler.get_config("evolution.evolution_rate", 5)
    await handler._analyze_group(group_id, evolution_rate)

    return {"success": True, "group_id": group_id}


async def set_spectrum(
    economic: Optional[int] = None,
    social: Optional[int] = None,
    diplomatic: Optional[int] = None,
    progressive: Optional[int] = None,
) -> dict:
    from ..models.ideology_model import get_or_create_spectrum, init_tables

    init_tables()
    spectrum = get_or_create_spectrum("global")

    if economic is not None:
        spectrum.economic = max(0, min(100, economic))
    if social is not None:
        spectrum.social = max(0, min(100, social))
    if diplomatic is not None:
        spectrum.diplomatic = max(0, min(100, diplomatic))
    if progressive is not None:
        spectrum.progressive = max(0, min(100, progressive))

    spectrum.updated_at = datetime.now()
    spectrum.save()

    return {
        "success": True,
        "spectrum": {
            "economic": spectrum.economic,
            "social": spectrum.social,
            "diplomatic": spectrum.diplomatic,
            "progressive": spectrum.progressive,
        },
    }
