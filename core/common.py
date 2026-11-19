"""Small helpers shared by the agent loop and the tools."""

from __future__ import annotations

import json
import math
import re
from typing import Any, Optional


class ToolError(RuntimeError):
    """Infrastructure failure while executing a tool (decoding, OCR, ASR, HTTP)."""


class SpecialistOutputError(ValueError):
    """A specialist model returned output that violates its JSON contract."""


class DuplicateJSONKeyError(ValueError):
    def __init__(self, key: str):
        super().__init__(f"duplicate JSON key: {key}")
        self.key = key


def format_seconds(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.1f}"


def strip_code_fence(text: str) -> str:
    text = text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1:]
        if text.endswith("```"):
            text = text[:-3]
    return text.strip()


def try_parse_json(text: str) -> Optional[Any]:
    """Lenient parser used for specialist outputs."""
    if not text:
        return None
    cleaned = strip_code_fence(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start:end + 1])
            except json.JSONDecodeError:
                continue
    return None


def try_parse_json_strict(text: str) -> Optional[Any]:
    """Main-agent parser: duplicate keys at any depth are rejected, not overwritten."""
    if not text:
        return None
    cleaned = strip_code_fence(text)

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateJSONKeyError(key)
            result[key] = value
        return result

    candidates = [cleaned]
    for open_ch, close_ch in (("{", "}"), ("[", "]")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            candidate = cleaned[start:end + 1]
            if candidate not in candidates:
                candidates.append(candidate)
    for candidate in candidates:
        try:
            return json.loads(candidate, object_pairs_hook=reject_duplicates)
        except DuplicateJSONKeyError:
            raise
        except json.JSONDecodeError:
            continue
    return None
