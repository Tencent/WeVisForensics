import asyncio
from pathlib import Path
from types import SimpleNamespace

from core.client import ChatEndpoint, text_part, user_with_images
from core.skill_card import CARD_HEADER, load_skill_card, render_card, with_card

CARD_PATH = Path(__file__).resolve().parents[1] / "configs" / "skill_card.json"


def _messages():
    return [{"role": "system", "content": "sys"}, user_with_images("inspect", ["a.jpg", "b.jpg"])]


def test_shipped_card_roles():
    cards = load_skill_card(CARD_PATH)
    assert [r["id"] for r in cards["visual_appearance"]] == ["R1", "R2", "R3", "R4", "R5"]
    assert [r["id"] for r in cards["common_sense"]] == ["R1", "R2", "R3"]
    assert cards["physical_consistency"] == [] and cards["ocr_consistency"] == []


def test_card_appended_to_last_user_text():
    rules = load_skill_card(CARD_PATH)["visual_appearance"]
    messages = _messages()
    out = with_card(messages, rules)
    assert out[0] == messages[0]
    assert out[1]["content"][:2] == messages[1]["content"][:2]
    text = out[1]["content"][-1]["text"]
    assert text == "inspect" + render_card(rules)
    assert text.startswith("inspect" + CARD_HEADER + "- [R1] Ignore: Motion blur")
    assert messages[1]["content"][-1]["text"] == "inspect"


def test_empty_rules_noop():
    messages = _messages()
    assert with_card(messages, []) is messages
    assert with_card(messages, None) is messages


def test_image_only_user_gets_text_part():
    rules = [{"id": "R1", "text": "x"}]
    out = with_card([{"role": "user", "content": [{"type": "image", "path": "a.jpg"}]}], rules)
    assert out[0]["content"][-1] == text_part(render_card(rules).lstrip())


def test_endpoint_sends_card():
    sent = {}

    async def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="ok")
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")], usage=None)

    rules = load_skill_card(CARD_PATH)["common_sense"]
    endpoint = ChatEndpoint("http://127.0.0.1:1/v1", "common_sense", skill_card=rules, image_transport="file")
    endpoint._client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    asyncio.run(endpoint.chat([{"role": "user", "content": [text_part("inspect")]}], max_tokens=8))
    assert sent["messages"][-1]["content"][-1]["text"] == "inspect" + render_card(rules)
