"""Action parsing: repeats, answering before a tool, and the last step."""

import json

from main_agent.protocol import parse_action


def _tool(skill="visual_appearance", start=1.0, end=4.0):
    return {"type": "run_skill", "skill": skill, "start_sec": start, "end_sec": end}


def _text(action):
    return json.dumps({"reasoning": "because", "action": action})


def test_answer_before_any_tool_is_invalid():
    parsed, errors = parse_action(
        _text({"type": "answer", "content": "fake"}),
        duration=8.0, has_audio=False, turn=1, max_steps=5,
        previous_actions=[], has_successful_tool=False,
    )
    assert parsed is not None
    assert errors == ["answer_before_successful_skill"]


def test_repeated_action_is_invalid():
    previous = _tool()
    parsed, errors = parse_action(
        _text(previous),
        duration=10.0, has_audio=True, turn=2, max_steps=5,
        previous_actions=[previous], has_successful_tool=True,
    )
    assert parsed is not None
    assert "repeated_action" in errors


def test_last_step_must_answer():
    parsed, errors = parse_action(
        _text(_tool("common_sense", 0.0, 2.0)),
        duration=10.0, has_audio=False, turn=5, max_steps=5,
        previous_actions=[_tool()], has_successful_tool=True,
    )
    assert "final_step_must_answer" in errors
    assert parsed["action"]["type"] == "run_skill"


def test_answer_after_a_tool_is_valid():
    parsed, errors = parse_action(
        _text({"type": "answer", "content": "real"}),
        duration=8.0, has_audio=False, turn=2, max_steps=5,
        previous_actions=[_tool()], has_successful_tool=True,
    )
    assert errors == []
    assert parsed["action"] == {"type": "answer", "content": "real"}
