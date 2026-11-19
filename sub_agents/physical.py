"""physical_consistency inspector."""

from __future__ import annotations

from sub_agents.base import run_visual_specialist

SKILL = "physical_consistency"


async def run(action: dict, **kwargs):
    return await run_visual_specialist(SKILL, action, **kwargs)
