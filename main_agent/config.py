"""Build a ForensicAgent from a JSON deployment config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from main_agent.loop import SUB_AGENT_ROLES, AgentConfig, ForensicAgent
from core.client import ChatEndpoint
from core.skill_card import load_skill_card


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as stream:
        return json.load(stream)


def build_agent(
    config: dict[str, Any],
    *,
    cache_root: str | None = None,
    main_backend: str = "engine",
) -> ForensicAgent:
    from sub_agents.audio import AudioConfig
    from sub_agents.ocr import PaddleOCRBackend

    transport = config.get("image_transport", "base64")
    timeout = float(config.get("timeout", 900))
    cards = load_skill_card(config["skill_card"]) if config.get("skill_card") else {}
    endpoints = {}
    roles = list(SUB_AGENT_ROLES)
    if main_backend == "http":
        roles = ["main", *roles]
    for role in roles:
        spec = config["endpoints"][role]
        endpoints[role] = ChatEndpoint(
            spec["base_url"], spec["model"], api_key=spec.get("api_key", "EMPTY"),
            timeout=timeout, image_transport=spec.get("image_transport", transport),
            skill_card=cards.get(role),
        )
    if main_backend == "engine":
        if "engine" not in config:
            raise ValueError(
                "engine backend needs config['engine']; "
                "configs/lora_single_server.json is the http quick-trial config"
            )
        from main_agent.engine import MainEngine

        spec = dict(config["engine"])
        main = MainEngine(spec.pop("model"), **spec)
    else:
        main = endpoints["main"]
    agent_config = AgentConfig(**config.get("agent", {}))
    if cache_root:
        agent_config.cache_root = cache_root
    return ForensicAgent(
        endpoints,
        main=main,
        backend=main_backend,
        ocr=PaddleOCRBackend(**config.get("ocr", {})),
        audio_config=AudioConfig(**config.get("audio", {})),
        config=agent_config,
    )
