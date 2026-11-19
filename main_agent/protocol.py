"""Action schema, validation rules and sampling grids of the forensic agent."""

from __future__ import annotations

import math
from typing import Any, Final, Hashable

from core.common import DuplicateJSONKeyError, try_parse_json_strict


SKILL_ARGS: Final = {
    "physical_consistency": ("start_sec", "end_sec"),
    "visual_appearance": ("start_sec", "end_sec"),
    "common_sense": ("start_sec", "end_sec"),
    "ocr_consistency": ("at_sec",),
    "voice_profile": (),
    "asr_analysis": (),
}
SKILLS: Final = tuple(SKILL_ARGS)
WINDOW_SKILLS: Final = frozenset({"physical_consistency", "visual_appearance", "common_sense"})
AUDIO_SKILLS: Final = frozenset({"voice_profile", "asr_analysis"})

DEFAULT_MAX_STEPS: Final = 5
OVERVIEW_FPS: Final = 1.0
OVERVIEW_MAX_FRAMES: Final = 16
SPECIALIST_FPS: Final = 2.0
SPECIALIST_MAX_FRAMES: Final = 72
FRAME_MAX_PIXELS: Final = 384 * 32 * 32
OCR_CROP_MAX_PIXELS: Final = 768 * 32 * 32
OCR_CROP_MARGIN: Final = 0.15

VISUAL_VERDICTS: Final = frozenset({"issue", "no_issue", "inconclusive"})
OCR_VERDICTS: Final = frozenset({"text_ok", "text_suspicious", "no_text", "inconclusive"})
ASR_VERDICTS: Final = frozenset({"timing_anomaly", "timing_natural", "inconclusive"})
CONFIDENCES: Final = frozenset({"high", "medium", "low"})


class ActionError(ValueError):
    pass


def normalize_model_output(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"reasoning", "action"}:
        raise ActionError("invalid_top_level_keys")
    if not isinstance(value.get("reasoning"), str):
        raise ActionError("reasoning_not_string")
    action = value.get("action")
    if not isinstance(action, dict):
        raise ActionError("action_not_object")
    action_type = action.get("type")
    if action_type == "answer":
        content = str(action.get("content", "")).strip().lower()
        if set(action) != {"type", "content"} or content not in {"real", "fake"}:
            raise ActionError("invalid_answer")
        normalized = {"type": "answer", "content": content}
    elif action_type == "run_skill":
        skill = str(action.get("skill", ""))
        if skill not in SKILL_ARGS:
            raise ActionError(f"invalid_skill:{skill!r}")
        if set(action) != {"type", "skill", *SKILL_ARGS[skill]}:
            raise ActionError("invalid_skill_keys")
        normalized = dict(action)
    else:
        raise ActionError(f"invalid_action_type:{action_type!r}")
    return {"reasoning": value["reasoning"], "action": normalized}


def normalize_temporal_action(action: dict[str, Any], duration: float) -> dict[str, Any]:
    """Clamp timestamps into the media and round them to the public 0.1 s precision."""
    normalized = dict(action)
    try:
        duration = float(duration)
    except (TypeError, ValueError):
        return normalized
    if not math.isfinite(duration) or duration <= 0 or normalized.get("type") != "run_skill":
        return normalized
    skill = normalized.get("skill")

    if skill == "ocr_consistency":
        try:
            at_sec = float(normalized.get("at_sec", duration / 2.0))
        except (TypeError, ValueError):
            at_sec = duration / 2.0
        if not math.isfinite(at_sec):
            at_sec = duration / 2.0
        normalized["at_sec"] = round(min(max(0.0, at_sec), duration), 1)
        return normalized

    if skill not in WINDOW_SKILLS:
        return normalized
    try:
        start = float(normalized.get("start_sec", 0.0))
    except (TypeError, ValueError):
        start = 0.0
    try:
        end = float(normalized.get("end_sec", duration))
    except (TypeError, ValueError):
        end = duration
    if not math.isfinite(start):
        start = 0.0
    if not math.isfinite(end) or end <= 0:
        end = duration
    start = min(max(0.0, start), duration)
    end = min(max(0.0, end), duration)
    if end <= start:
        end = duration
        if start >= duration:
            start = max(0.0, duration - min(2.0, duration))
    rounded_start, rounded_end = round(start, 1), round(end, 1)
    # Keep sub-0.1 s clips from collapsing to an empty window.
    if end > start and rounded_end <= rounded_start:
        rounded_start, rounded_end = round(start, 6), round(end, 6)
    normalized["start_sec"] = rounded_start
    normalized["end_sec"] = rounded_end
    return normalized


def action_signature(action: dict[str, Any]) -> tuple[Hashable, ...]:
    skill = action.get("skill")
    if skill == "ocr_consistency":
        return ("run_skill", str(skill), action.get("at_sec"))
    if skill in WINDOW_SKILLS:
        return ("run_skill", str(skill), action.get("start_sec"), action.get("end_sec"))
    return ("run_skill", str(skill))


def parse_action(
    text: str,
    *,
    duration: float,
    has_audio: bool,
    turn: int,
    max_steps: int,
    previous_actions: list[dict[str, Any]],
    has_successful_tool: bool,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Parse one main-agent message. Any returned error ends the trajectory as invalid."""
    try:
        parsed = try_parse_json_strict(text)
    except DuplicateJSONKeyError as exc:
        return None, [f"duplicate_json_key:{exc.key!r}"]
    if not isinstance(parsed, dict) or "action" not in parsed:
        return None, ["unparseable_json"]
    try:
        parsed = normalize_model_output(parsed)
    except ActionError as exc:
        return None, [str(exc)]
    action = normalize_temporal_action(parsed["action"], duration)
    parsed["action"] = action
    errors: list[str] = []
    if action["type"] == "answer" and not has_successful_tool:
        errors.append("answer_before_successful_skill")
    if turn >= max_steps and action["type"] != "answer":
        errors.append("final_step_must_answer")
    if action["type"] == "run_skill":
        signature = action_signature(action)
        if any(action_signature(old) == signature for old in previous_actions):
            errors.append("repeated_action")
        if not has_audio and action["skill"] in AUDIO_SKILLS:
            errors.append("audio_skill_on_silent_video")
    return parsed, errors


def overview_timestamps(
    duration: float, *, fps: float = OVERVIEW_FPS, max_frames: int = OVERVIEW_MAX_FRAMES,
) -> list[float]:
    """The 0.0, 1.0, ... grid over the full video, uniformly subsampled to ``max_frames``."""
    duration = max(0.001, float(duration))
    fps = max(0.01, float(fps))
    natural_count = max(1, int(math.ceil(duration * fps)))
    count = min(max(1, int(max_frames)), natural_count)
    natural = [
        round(min(max(0.0, duration - 1e-3), index / fps), 1) for index in range(natural_count)
    ]
    if natural_count <= count:
        return natural
    if count == 1:
        return [natural[0]]
    indices = [round(index * (natural_count - 1) / (count - 1)) for index in range(count)]
    return [natural[index] for index in indices]


def specialist_timestamps(
    start_sec: float, end_sec: float, *, max_frames: int = SPECIALIST_MAX_FRAMES,
) -> list[float]:
    """The 2 FPS grid inside a window, uniformly capped while keeping full coverage."""
    if end_sec <= start_sec:
        raise ValueError(f"invalid window: {start_sec}..{end_sec}")
    first = math.ceil(float(start_sec) * SPECIALIST_FPS - 1e-9)
    last = math.ceil(float(end_sec) * SPECIALIST_FPS - 1e-9)
    values = [index / SPECIALIST_FPS for index in range(first, last)]
    if not values:
        values = [float(start_sec)]
    if len(values) > max_frames:
        selected = [
            round(index * (len(values) - 1) / (max_frames - 1)) for index in range(max_frames)
        ]
        values = [values[index] for index in selected]
    return [round(value, 6) for value in values]


def validate_visual_output(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != {"verdict", "confidence", "reason"}:
        raise ValueError("specialist output must contain only verdict/confidence/reason")
    verdict = str(value.get("verdict") or "")
    confidence = str(value.get("confidence") or "")
    reason = str(value.get("reason") or "").strip()
    if verdict not in VISUAL_VERDICTS:
        raise ValueError(f"invalid verdict: {verdict!r}")
    if confidence not in CONFIDENCES:
        raise ValueError(f"invalid confidence: {confidence!r}")
    if not reason:
        raise ValueError("empty reason")
    return {"verdict": verdict, "confidence": confidence, "reason": reason}


def render_tool_result(
    skill: str, scope: str, verdict: str, confidence: str, *, detail: str = "",
    issues: list[Any] | None = None, reason: str = "",
) -> str:
    """Observation text returned to the main agent after a specialist call."""
    summary = f"{skill} {verdict}({confidence}){(' ' + detail) if detail else ''}."
    if issues:
        summary += f" issues={issues}."
    if reason:
        summary += f" reason: {reason}"
    return f"Skill: {skill} | {scope} | Verdict: {verdict} ({confidence})\n{summary}"
