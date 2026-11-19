"""Prompt bytes are frozen to the prompts the models were trained with."""

import hashlib

from main_agent.prompts import MAIN_AGENT_SYSTEM, MAIN_AGENT_SYSTEM_SHA256
from sub_agents.audio import ASR_JUDGE_SYSTEM, ASR_VLM_PROMPT
from sub_agents.prompts import (
    OCR_STAGE1_PROMPT,
    OCR_STAGE1_SYSTEM,
    OCR_STAGE2_PROMPT,
    OCR_STAGE2_SYSTEM,
    visual_specialist_system,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_main_prompt_sha256():
    expected = "2daa19b036e597efc3571e9d6dacb4ca6d8c42bf9e8b6cbb30b3977778f5cf62"
    assert _sha(MAIN_AGENT_SYSTEM) == expected
    assert MAIN_AGENT_SYSTEM_SHA256 == expected


def test_inspector_prompt_sha256():
    assert _sha(visual_specialist_system("physical_consistency")) == (
        "18956e6d03b18b2bb2e7be463bdebd7d4b0248210c4fea3bce70c9be608a391d"
    )
    assert _sha(visual_specialist_system("visual_appearance")) == (
        "a58ed2229888e1a17d3981cd70609e4c4ccb184572e77a0c51eca37406b31120"
    )
    assert _sha(visual_specialist_system("common_sense")) == (
        "1f4ade54b9fb33a6262c07d515e0f6df85e377c55c41bcd794c4c8976939bf7a"
    )
    assert _sha(OCR_STAGE1_SYSTEM) == "89402f449576e9313312ffe351e48640f9a79003d57768716ffa9ea53636aaae"
    assert _sha(OCR_STAGE1_PROMPT) == "ef0668bfd2bd440bf01d77684075dfa227c58ddf089f6c9de0a1ddfd7bac76bd"
    assert _sha(OCR_STAGE2_SYSTEM) == "9e38e054c828bfe4385c13fa65d10cf0c7d081e873bb9cc54aa78eb3ca6ef146"
    assert _sha(OCR_STAGE2_PROMPT) == "206fffc443f67e39d60af459c185766978f9f89c2a82fd85e3832a7e5f5ef55f"
    assert _sha(ASR_JUDGE_SYSTEM) == "bf9ff0a18850e6af104b614a09b1c4af300e0e8150b3f551b5ddefaf29dcbca3"
    assert _sha(ASR_VLM_PROMPT) == "acb62a1af8a8f5db6ef685cbab74cf2cb1b88da8485c910b07450268fc5cb58c"
