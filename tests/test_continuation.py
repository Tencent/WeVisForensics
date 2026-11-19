"""Generated tokens are appended; assistant text is not re-encoded."""

from main_agent.continuation import TokenHistory, initialize_turn_separator


def test_history_appends_generation_then_observation():
    seen = []

    def encode(messages, images):
        seen.append([(message["role"], images) for message in messages])
        assert all(message["role"] != "assistant" for message in messages)
        if any(message["role"] == "system" for message in messages):
            return [1, 2, 3]
        return [8, 9]

    history = TokenHistory(encode, turn_separator=[7], response_budget=100)
    assert history.start(
        [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], ["frame.jpg"],
    ) == [1, 2, 3]
    history.note_generation([4, 5])
    assert history.push_observation("OBS") is None
    history.note_generation([6])
    assert history.runtime_ids == [1, 2, 3, 4, 5, 7, 8, 9, 6]
    assert seen[1] == [("user", None)]


def test_observation_is_counted_before_the_next_generation():
    def encode(messages, images):
        if any(message["role"] == "system" for message in messages):
            return [1]
        return [8, 9]

    tight = TokenHistory(encode, turn_separator=[7], response_budget=4)
    tight.start([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], None)
    tight.note_generation([4, 5])
    assert tight.push_observation("OBS") == "trajectory_length_exceeded"
    assert tight.runtime_ids == [1, 4, 5]

    exact = TokenHistory(encode, turn_separator=[7], response_budget=5)
    exact.start([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], None)
    exact.note_generation([4, 5])
    assert exact.push_observation("OBS") is None
    assert exact.runtime_ids == [1, 4, 5, 7, 8, 9]
    assert exact.remaining() == 0


class _Tokenizer:
    eos_token_id = 2

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        del tokenize, add_generation_prompt, kwargs
        ids = []
        for message in messages:
            body = message["content"]
            if isinstance(body, list):
                body = body[0]["text"]
            ids.extend([1, 9 if body else 0, 2, 3])
        return ids


def test_turn_separator_is_the_token_after_eos():
    assert initialize_turn_separator(_Tokenizer(), enable_thinking=False) == [3]
