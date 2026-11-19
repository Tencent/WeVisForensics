"""Skill cards learned during RL, appended to the end of each sub-agent's user text."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

CARD_HEADER = "\n\n### Skill card (guidance learned from previous investigations; apply only when relevant)\n"

Rule = dict[str, str]


def load_skill_card(path: str | Path) -> dict[str, list[Rule]]:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)["roles"]


def render_card(rules: list[Rule] | None) -> str:
    if not rules:
        return ""
    return CARD_HEADER + "\n".join(f"- [{rule['id']}] {rule['text']}" for rule in rules)


def with_card(messages: list[dict[str, Any]], rules: list[Rule] | None) -> list[dict[str, Any]]:
    text = render_card(rules)
    if not text:
        return messages
    out = copy.deepcopy(messages)
    user = next((m for m in reversed(out) if m.get("role") == "user"), None)
    if user is None:
        return out
    content = user["content"]
    if isinstance(content, str):
        user["content"] = content + text
    else:
        parts = [p for p in content if p.get("type") == "text"]
        if parts:
            parts[-1]["text"] += text
        else:
            content.append({"type": "text", "text": text.lstrip()})
    return out
