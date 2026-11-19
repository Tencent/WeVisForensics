"""Multi-turn main-agent loop."""

from __future__ import annotations

import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from main_agent.continuation import TokenHistory
from main_agent.prompts import MAIN_AGENT_SYSTEM, overview_user, step_notice
from main_agent.protocol import (
    ASR_VERDICTS,
    CONFIDENCES,
    DEFAULT_MAX_STEPS,
    FRAME_MAX_PIXELS,
    overview_timestamps,
    parse_action,
    render_tool_result,
)
from core.client import ChatEndpoint, user_with_images
from core.common import SpecialistOutputError, ToolError, try_parse_json
from core.media import FrameCache, probe_media

SUB_AGENT_ROLES = (
    "physical_consistency", "visual_appearance", "common_sense",
    "ocr_consistency", "asr_analysis",
)
ROLES = ("main", *SUB_AGENT_ROLES)


@dataclass
class AgentConfig:
    max_steps: int = DEFAULT_MAX_STEPS
    turn_max_tokens: int = 1024
    response_budget: int = 12288
    specialist_max_tokens: int = 500
    frame_max_pixels: int = FRAME_MAX_PIXELS
    cache_root: str | None = None


@dataclass
class Trajectory:
    video: str
    duration: float = 0.0
    has_audio: bool = False
    overview_timestamps: list[float] = field(default_factory=list)
    steps: list[dict[str, Any]] = field(default_factory=list)
    status: str = "running"
    answer: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ForensicAgent:
    def __init__(
        self,
        endpoints: dict[str, ChatEndpoint],
        *,
        main: Any,
        backend: str = "engine",
        ocr: Any = None,
        audio_config: Any = None,
        config: AgentConfig | None = None,
    ):
        if backend not in {"engine", "http"}:
            raise ValueError(f"unknown main backend: {backend!r}")
        missing = [role for role in SUB_AGENT_ROLES if role not in endpoints]
        if missing:
            raise ValueError(f"missing endpoints for roles: {missing}")
        if backend == "http" and not isinstance(main, ChatEndpoint):
            raise TypeError("http backend requires a ChatEndpoint for the main agent")
        self.endpoints = endpoints
        self.main = main
        self.backend = backend
        self.config = config or AgentConfig()
        self.frames = FrameCache(self.config.cache_root)
        if ocr is None:
            from sub_agents.ocr import PaddleOCRBackend

            ocr = PaddleOCRBackend()
        self.ocr = ocr
        if audio_config is None:
            from sub_agents.audio import AudioConfig

            audio_config = AudioConfig()
        self.audio_config = audio_config

    async def run(
        self, video_path: str, *, duration: float | None = None, has_audio: bool | None = None,
    ) -> Trajectory:
        """``duration`` / ``has_audio`` from a manifest take precedence over probing the file."""
        cfg = self.config
        video_path = str(Path(video_path).resolve())
        traj = Trajectory(video=video_path)
        if duration is None or has_audio is None:
            probed_duration, probed_audio = await asyncio.to_thread(probe_media, video_path)
            duration = probed_duration if duration is None else duration
            has_audio = probed_audio if has_audio is None else has_audio
        traj.duration, traj.has_audio = float(duration), bool(has_audio)
        paths, actual = await asyncio.to_thread(
            self.frames.extract, video_path, overview_timestamps(traj.duration),
            max_pixels=cfg.frame_max_pixels,
        )
        if not paths:
            raise ToolError(f"could not decode overview frames: {video_path}")
        traj.overview_timestamps = [round(value, 1) for value in actual]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": MAIN_AGENT_SYSTEM},
            user_with_images(
                overview_user(
                    duration=traj.duration, has_audio=traj.has_audio,
                    timestamps=traj.overview_timestamps,
                ),
                paths,
            ),
        ]
        if self.backend == "engine":
            await self._run_engine(traj, messages, paths)
        else:
            await self._run_http(traj, messages)
        return traj

    async def _run_engine(self, traj: Trajectory, messages: list[dict[str, Any]], paths: list[str]) -> None:
        from PIL import Image

        cfg = self.config
        images = []
        for path in paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
        history = TokenHistory(self.main.encode, turn_separator=[], response_budget=cfg.response_budget)
        try:
            history.start(messages, images)
        except Exception as exc:
            raise ToolError(f"main-agent prompt encoding failed: {exc}") from exc
        history.turn_separator = list(self.main.turn_separator)
        executed: list[dict[str, Any]] = []
        for turn in range(1, cfg.max_steps + 1):
            remaining = history.remaining()
            if remaining <= 0:
                traj.status = "response_budget_exhausted"
                break
            try:
                result = await self.main.generate(
                    history.runtime_ids, history.images, max_tokens=min(cfg.turn_max_tokens, remaining),
                )
            except Exception as exc:
                raise ToolError(f"main-agent generation failed: {exc}") from exc
            history.note_generation(result.token_ids)
            if not await self._consume_turn(traj, turn, result.text, result.finish_reason, executed):
                break
            status = history.push_observation(
                traj.steps[-1]["observation"] + "\n" + step_notice(turn, cfg.max_steps),
            )
            if status:
                traj.status = status
                break
        else:
            traj.status = "max_steps_exhausted"

    async def _run_http(self, traj: Trajectory, messages: list[dict[str, Any]]) -> None:
        """Chat-completions backend. It re-encodes assistant text and is not token-aligned."""
        cfg = self.config
        initial_prompt_tokens: int | None = None
        used_tokens = 0
        executed: list[dict[str, Any]] = []
        for turn in range(1, cfg.max_steps + 1):
            remaining = cfg.response_budget - used_tokens
            if remaining <= 0:
                traj.status = "response_budget_exhausted"
                break
            result = await self.main.chat(messages, max_tokens=min(cfg.turn_max_tokens, remaining))
            if initial_prompt_tokens is None:
                initial_prompt_tokens = result.prompt_tokens
            elif result.prompt_tokens - initial_prompt_tokens > cfg.response_budget:
                traj.status = "trajectory_length_exceeded"
                break
            used_tokens = (result.prompt_tokens - initial_prompt_tokens) + result.completion_tokens
            if not await self._consume_turn(traj, turn, result.text, result.finish_reason, executed):
                break
            observation = traj.steps[-1]["observation"]
            messages.append({"role": "assistant", "content": result.text})
            messages.append({"role": "user", "content": observation + "\n" + step_notice(turn, cfg.max_steps)})
        else:
            traj.status = "max_steps_exhausted"

    async def _consume_turn(
        self, traj: Trajectory, turn: int, text: str, finish_reason: str | None,
        executed: list[dict[str, Any]],
    ) -> bool:
        """Parse one generation. Return True when an observation was stored."""
        cfg = self.config
        parsed, errors = parse_action(
            text, duration=traj.duration, has_audio=traj.has_audio, turn=turn,
            max_steps=cfg.max_steps, previous_actions=executed, has_successful_tool=bool(executed),
        )
        step: dict[str, Any] = {
            "turn": turn,
            "raw": text,
            "action": parsed["action"] if parsed else None,
            "reasoning": parsed["reasoning"] if parsed else None,
            "errors": errors,
            "finish_reason": finish_reason,
        }
        traj.steps.append(step)
        if parsed is None or errors:
            traj.status = "invalid_output"
            return False
        action = parsed["action"]
        if action["type"] == "answer":
            traj.answer = action["content"]
            traj.status = "answered"
            return False
        executed.append(action)
        try:
            observation, tool_trace = await self._execute(action, traj)
        except SpecialistOutputError as exc:
            step["tool_error"] = str(exc)
            traj.status = "specialist_invalid_output"
            return False
        step["observation"] = observation
        step["tool_trace"] = tool_trace
        return True

    def _work_dir(self, video_path: str) -> Path:
        key = hashlib.sha256(video_path.encode()).hexdigest()[:16]
        path = self.frames.root / "audio" / key
        path.mkdir(parents=True, exist_ok=True)
        return path

    async def _execute(self, action: dict[str, Any], traj: Trajectory) -> tuple[str, dict[str, Any]]:
        skill = action["skill"]
        cfg = self.config
        kwargs = dict(
            video_path=traj.video, endpoint=self.endpoints.get(skill), frames=self.frames,
            frame_max_pixels=cfg.frame_max_pixels, max_tokens=cfg.specialist_max_tokens,
        )
        if skill == "visual_appearance":
            from sub_agents.appearance import run
            return await run(action, **kwargs)
        if skill == "physical_consistency":
            from sub_agents.physical import run
            return await run(action, **kwargs)
        if skill == "common_sense":
            from sub_agents.common_sense import run
            return await run(action, **kwargs)
        if skill == "ocr_consistency":
            from sub_agents.ocr import run_ocr_specialist
            return await run_ocr_specialist(
                action, video_path=traj.video, endpoint=self.endpoints[skill],
                frames=self.frames, ocr=self.ocr, max_tokens=cfg.specialist_max_tokens,
            )
        if skill == "voice_profile":
            from sub_agents.audio import run_voice_profile
            text = await asyncio.to_thread(
                run_voice_profile, traj.video, self._work_dir(traj.video), self.audio_config,
                end_sec=traj.duration,
            )
            return text, {}
        if skill == "asr_analysis":
            return await self._asr_analysis(traj)
        raise ToolError(f"unsupported skill: {skill}")

    async def _asr_analysis(self, traj: Trajectory) -> tuple[str, dict[str, Any]]:
        from sub_agents.audio import ASR_JUDGE_SYSTEM, build_asr_review_prompt, transcribe

        payload = await asyncio.to_thread(
            transcribe, traj.video, self._work_dir(traj.video), self.audio_config,
            start_sec=0.0, end_sec=traj.duration,
        )
        trace = {"transcript": payload.get("transcript", ""), "language": payload.get("language")}
        if not str(payload.get("transcript", "")).strip():
            return "Skill: asr_analysis | whole video | Verdict: inconclusive (low)\nASR found no speech.", trace
        messages = [
            {"role": "system", "content": ASR_JUDGE_SYSTEM},
            {"role": "user", "content": build_asr_review_prompt(payload, traj.duration)},
        ]
        result = await self.endpoints["asr_analysis"].chat(messages, max_tokens=self.config.specialist_max_tokens)
        parsed = try_parse_json(result.text)
        parsed = parsed if isinstance(parsed, dict) else {}
        verdict = str(parsed.get("verdict") or "inconclusive")
        if verdict not in ASR_VERDICTS:
            verdict = "inconclusive"
        confidence = str(parsed.get("confidence") or "low")
        if confidence not in CONFIDENCES:
            confidence = "low"
        trace["raw_output"] = result.text
        text = render_tool_result(
            "asr_analysis", "whole video", verdict, confidence,
            issues=parsed.get("issues") or [], reason=str(parsed.get("reason") or "").strip(),
        )
        return text, trace
