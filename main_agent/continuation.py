"""Token-level history for the main agent.

Each turn appends the raw generated token ids, then a turn separator and the
encoded observation. Assistant text is never re-rendered through the chat template.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence


Encode = Callable[[list[dict[str, Any]], list[Any] | None], list[int]]


def normalize_token_ids(tokenized_output: Any) -> list[int]:
    token_ids = tokenized_output
    if isinstance(tokenized_output, dict):
        token_ids = tokenized_output["input_ids"]
    elif hasattr(tokenized_output, "input_ids"):
        token_ids = tokenized_output.input_ids
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if isinstance(token_ids, tuple):
        token_ids = list(token_ids)
    if isinstance(token_ids, list) and len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
        token_ids = list(token_ids[0])
    return [int(token_id.item() if hasattr(token_id, "item") else token_id) for token_id in token_ids]


def _apply(tokenizer: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    return tokenizer.apply_chat_template(messages, **kwargs)


def initialize_turn_separator(tokenizer: Any, **template_kwargs: Any) -> list[int]:
    """Tokens after the assistant close token that the model does not emit."""
    empty = filled = None
    for as_parts in (False, True):
        if as_parts:
            body_empty, body_filled = [{"type": "text", "text": ""}], [{"type": "text", "text": "x"}]
        else:
            body_empty, body_filled = "", "x"
        try:
            empty = normalize_token_ids(_apply(
                tokenizer, [{"role": "user", "content": body_empty}],
                add_generation_prompt=False, tokenize=True, **template_kwargs,
            ))
            filled = normalize_token_ids(_apply(
                tokenizer, [{"role": "user", "content": body_filled}],
                add_generation_prompt=False, tokenize=True, **template_kwargs,
            ))
            break
        except Exception:
            empty = filled = None
    if not empty or not filled:
        return []
    shared = 0
    while shared < len(empty) and shared < len(filled) and empty[-1 - shared] == filled[-1 - shared]:
        shared += 1
    suffix = empty[len(empty) - shared:]
    if not suffix:
        return []
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is None:
        eos_id = getattr(getattr(tokenizer, "tokenizer", None), "eos_token_id", None)
    eos_ids = {eos_id} if isinstance(eos_id, int) else set(eos_id or [])
    last_close = max((index for index, token_id in enumerate(suffix) if token_id in eos_ids), default=None)
    if last_close is not None:
        return suffix[last_close + 1:]
    return suffix[1:]


def initialize_system_prompt(tokenizer: Any, **template_kwargs: Any) -> list[int]:
    one = normalize_token_ids(_apply(
        tokenizer, [{"role": "user", "content": ""}],
        add_generation_prompt=False, tokenize=True, **template_kwargs,
    ))
    two = normalize_token_ids(_apply(
        tokenizer, [{"role": "user", "content": ""}] * 2,
        add_generation_prompt=False, tokenize=True, **template_kwargs,
    ))
    return one[: -(len(two) - len(one))]


class TokenHistory:
    """``runtime_ids`` is the next prompt. ``response_ids`` is the 12288 budget."""

    def __init__(self, encode: Encode, *, turn_separator: Sequence[int], response_budget: int):
        self.encode = encode
        self.turn_separator = list(turn_separator)
        self.response_budget = int(response_budget)
        self.runtime_ids: list[int] = []
        self.response_ids: list[int] = []
        self.images: list[Any] = []

    def start(self, messages: list[dict[str, Any]], images: list[Any] | None) -> list[int]:
        self.images = list(images or [])
        self.runtime_ids = list(self.encode(messages, self.images or None))
        self.response_ids = []
        return list(self.runtime_ids)

    def remaining(self) -> int:
        return self.response_budget - len(self.response_ids)

    def note_generation(self, token_ids: Sequence[int]) -> None:
        generated = [int(token_id) for token_id in token_ids]
        self.runtime_ids.extend(generated)
        self.response_ids.extend(generated)

    def push_observation(self, text: str) -> str | None:
        """Count observation tokens before the next generation. Drop them if they do not fit."""
        addition = list(self.encode([{"role": "user", "content": text}], None))
        complete = self.turn_separator + addition
        if len(self.response_ids) + len(complete) > self.response_budget:
            return "trajectory_length_exceeded"
        self.runtime_ids.extend(complete)
        self.response_ids.extend(complete)
        return None
