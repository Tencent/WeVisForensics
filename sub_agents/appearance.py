"""visual_appearance inspector."""

from __future__ import annotations

from sub_agents.base import run_visual_specialist

SKILL = "visual_appearance"


async def run(action: dict, **kwargs):
    return await run_visual_specialist(SKILL, action, **kwargs)
