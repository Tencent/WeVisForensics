"""OpenAI-compatible chat client for vLLM endpoints."""

from __future__ import annotations

import base64
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.skill_card import Rule, with_card


Message = dict[str, Any]


def image_part(path: str) -> dict[str, str]:
    return {"type": "image", "path": str(path)}


def text_part(text: str) -> dict[str, str]:
    return {"type": "text", "text": text}


def user_with_images(text: str, image_paths: list[str]) -> Message:
    return {"role": "user", "content": [*(image_part(p) for p in image_paths), text_part(text)]}


@functools.lru_cache(maxsize=4096)
def _data_url(path: str) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str | None


class ChatEndpoint:
    """One served model (a full checkpoint or a LoRA adapter name on a vLLM server).

    ``image_transport="file"`` sends ``file://`` URLs and requires the server to be
    launched with ``--allowed-local-media-path``; ``"base64"`` works with remote servers.
    ``skill_card`` rules, if given, are appended to the last user text of every request.
    """

    def __init__(
        self, base_url: str, model: str, *, api_key: str = "EMPTY", timeout: float = 900.0,
        image_transport: str = "base64", max_retries: int = 3, skill_card: list[Rule] | None = None,
    ):
        from openai import AsyncOpenAI

        if image_transport not in {"base64", "file"}:
            raise ValueError(f"unknown image_transport: {image_transport!r}")
        self.model = model
        self.image_transport = image_transport
        self.skill_card = skill_card or []
        self._client = AsyncOpenAI(
            base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries,
        )

    def _image_url(self, path: str) -> str:
        if self.image_transport == "file":
            return Path(path).resolve().as_uri()
        return _data_url(str(Path(path).resolve()))

    def _convert(self, messages: list[Message]) -> list[dict[str, Any]]:
        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append({"role": message["role"], "content": content})
                continue
            parts = []
            for part in content:
                if part["type"] == "image":
                    parts.append({"type": "image_url", "image_url": {"url": self._image_url(part["path"])}})
                else:
                    parts.append({"type": "text", "text": part["text"]})
            converted.append({"role": message["role"], "content": parts})
        return converted

    async def chat(self, messages: list[Message], *, max_tokens: int) -> ChatResult:
        """Greedy decoding with thinking disabled, as in training and evaluation."""
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=self._convert(with_card(messages, self.skill_card)),
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            extra_body={
                "top_k": -1,
                "repetition_penalty": 1.0,
                "chat_template_kwargs": {"enable_thinking": False},
            },
        )
        if not response.choices:
            raise RuntimeError(f"{self.model}: empty response")
        choice = response.choices[0]
        usage = response.usage
        return ChatResult(
            text=choice.message.content or "",
            prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
            finish_reason=choice.finish_reason,
        )
