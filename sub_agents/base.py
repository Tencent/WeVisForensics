"""Shared runner for the three temporal visual inspectors."""

from __future__ import annotations

import asyncio

from main_agent.protocol import render_tool_result, specialist_timestamps, validate_visual_output
from sub_agents.prompts import visual_specialist_system, visual_specialist_user
from core.client import ChatEndpoint, user_with_images
from core.common import SpecialistOutputError, ToolError, format_seconds, try_parse_json
from core.media import FrameCache


async def run_visual_specialist(
    skill: str,
    action: dict,
    *,
    video_path: str,
    endpoint: ChatEndpoint,
    frames: FrameCache,
    frame_max_pixels: int,
    max_tokens: int,
) -> tuple[str, dict]:
    start_sec, end_sec = float(action["start_sec"]), float(action["end_sec"])
    requested = specialist_timestamps(start_sec, end_sec)
    paths, actual = await asyncio.to_thread(
        frames.extract, video_path, requested, max_pixels=frame_max_pixels,
    )
    if not paths:
        raise ToolError(f"{skill}: could not decode frames in {start_sec}-{end_sec}s")
    messages = [
        {"role": "system", "content": visual_specialist_system(skill)},
        user_with_images(visual_specialist_user(skill, actual), paths),
    ]
    result = await endpoint.chat(messages, max_tokens=max_tokens)
    parsed = try_parse_json(result.text)
    try:
        output = validate_visual_output(parsed if isinstance(parsed, dict) else {})
    except ValueError as exc:
        raise SpecialistOutputError(f"{skill}: {exc}") from exc
    text = render_tool_result(
        skill,
        f"Range: {format_seconds(start_sec)}s-{format_seconds(end_sec)}s",
        output["verdict"],
        output["confidence"],
        detail=f"over {len(paths)} frames",
        reason=output["reason"],
    )
    trace = {"frame_timestamps": actual, "raw_output": result.text, "parsed": output}
    return text, trace
