"""In-process vLLM backend for the main agent.

Imports of vLLM happen on the first generate call so CPU tests can import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from main_agent.continuation import initialize_system_prompt, initialize_turn_separator, normalize_token_ids


DEFAULT_SEED = 20260716
DEFAULT_MAX_MODEL_LEN = 131072
DEFAULT_PROMPT_LENGTH = 131072
TEMPLATE_KWARGS = {"enable_thinking": False}


def dedup_image_tokens(prompt_ids: list[int], processor: Any) -> list[int]:
    """Collapse consecutive image-pad tokens. vLLM expands them from the image list."""
    image_processor = getattr(processor, "image_processor", None)
    name = type(image_processor).__name__ if image_processor is not None else ""
    if "Qwen2VLImageProcessor" not in name:
        return list(prompt_ids)
    import numpy as np

    ids = np.asarray(prompt_ids)
    mask = np.ones(len(ids), dtype=bool)
    image_id = int(processor.image_token_id)
    video_id = int(getattr(processor, "video_token_id", -1))
    is_pad = (ids == image_id) | (ids == video_id)
    mask[1:] &= ~(is_pad[1:] & is_pad[:-1])
    return ids[mask].tolist()


def _apply_template(processor: Any, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    try:
        return processor.apply_chat_template(messages, **kwargs)
    except Exception:
        dummy = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        prefix = processor.apply_chat_template(
            dummy, tokenize=kwargs.get("tokenize", True), add_generation_prompt=False, **TEMPLATE_KWARGS,
        )
        output = processor.apply_chat_template(dummy + messages, **kwargs)
        if kwargs.get("tokenize", True):
            prefix_ids = normalize_token_ids(prefix)
            output_ids = normalize_token_ids(output)
            return output_ids[len(prefix_ids):]
        return output[len(prefix):]


def encode_with_processor(
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Any] | None,
    *,
    system_prompt_ids: list[int],
    remove_system_prompt: bool,
) -> list[int]:
    # Text-only turns also go through the processor, so their tokens match the multimodal path.
    text = _apply_template(
        processor, messages, tokenize=False, add_generation_prompt=True, **TEMPLATE_KWARGS,
    )
    batch = processor(text=[text], images=list(images) if images else None, videos=None, return_tensors="pt")
    ids = normalize_token_ids(batch)
    if remove_system_prompt:
        ids = ids[len(system_prompt_ids):]
    return ids


@dataclass
class Generation:
    token_ids: list[int]
    text: str
    finish_reason: str | None


class MainEngine:
    """Default main-agent backend: one AsyncLLM, token-in token-out."""

    def __init__(
        self,
        model: str,
        *,
        gpu_memory_utilization: float = 0.60,
        max_model_len: int = DEFAULT_MAX_MODEL_LEN,
        max_num_seqs: int = 32,
        seed: int = DEFAULT_SEED,
        enforce_eager: bool = False,
        gdn_prefill_backend: str = "triton",
        chat_template: str | None = "configs/chat_template.jinja",
        allowed_local_media_path: str = "/",
        prompt_length: int = DEFAULT_PROMPT_LENGTH,
    ):
        self.model = model
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = int(max_model_len)
        self.max_num_seqs = int(max_num_seqs)
        self.seed = int(seed)
        self.enforce_eager = bool(enforce_eager)
        self.gdn_prefill_backend = gdn_prefill_backend
        self.chat_template = chat_template
        self.allowed_local_media_path = allowed_local_media_path
        self.prompt_length = int(prompt_length)
        self._engine = None
        self.processor = None
        self.tokenizer = None
        self.system_prompt_ids: list[int] = []
        self.turn_separator: list[int] = []

    def _read_template(self) -> str | None:
        if not self.chat_template:
            return None
        candidate = Path(self.chat_template)
        packaged = Path(__file__).resolve().parents[1] / self.chat_template
        path = candidate if candidate.is_file() else packaged
        if not path.is_file():
            raise FileNotFoundError(f"chat template not found: {self.chat_template}")
        return path.read_text(encoding="utf-8")

    def load(self) -> None:
        if self._engine is not None:
            return
        import os

        # DeepGEMM only serves FP8 kernels; importing it aborts on some CUDA builds.
        for key, value in (("VLLM_USE_DEEP_GEMM", "0"), ("VLLM_MOE_USE_DEEP_GEMM", "0"),
                           ("VLLM_WORKER_MULTIPROC_METHOD", "spawn")):
            os.environ.setdefault(key, value)
        from transformers import AutoProcessor
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM

        processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
        template = self._read_template()
        tokenizer = getattr(processor, "tokenizer", processor)
        if template:
            tokenizer.chat_template = template
        self.processor = processor
        self.tokenizer = tokenizer
        self.system_prompt_ids = initialize_system_prompt(processor, **TEMPLATE_KWARGS)
        self.turn_separator = initialize_turn_separator(processor, **TEMPLATE_KWARGS)
        self._engine = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=self.model,
            tokenizer=self.model,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enable_prefix_caching=True,
            enforce_eager=self.enforce_eager,
            limit_mm_per_prompt={"image": 120, "video": 0},
            allowed_local_media_path=self.allowed_local_media_path,
            gdn_prefill_backend=self.gdn_prefill_backend,
            seed=self.seed,
        ))

    def encode(self, messages: list[dict[str, Any]], images: list[Any] | None) -> list[int]:
        self.load()
        return encode_with_processor(
            self.processor, messages, images,
            system_prompt_ids=self.system_prompt_ids,
            remove_system_prompt=images is None,
        )

    async def generate(self, prompt_ids: list[int], images: list[Any], *, max_tokens: int) -> Generation:
        self.load()
        from vllm import SamplingParams

        if len(prompt_ids) > self.prompt_length and images:
            raise RuntimeError(
                f"initial prompt has {len(prompt_ids)} tokens, above prompt_length={self.prompt_length}"
            )
        token_ids = dedup_image_tokens(list(prompt_ids), self.processor)
        headroom = self.max_model_len - len(token_ids)
        if headroom < 1:
            raise RuntimeError(f"no context headroom: {len(token_ids)} tokens")
        maximum = max(1, min(int(max_tokens), headroom))
        prompt = {
            "prompt_token_ids": token_ids,
            "multi_modal_data": {"image": list(images)} if images else {},
        }
        sampling = SamplingParams(
            max_tokens=maximum, temperature=0.0, top_p=1.0, top_k=-1, repetition_penalty=1.0,
        )
        request_id = f"main-{uuid4().hex}"

        async def _pull(prompt_obj: Any) -> Any:
            collected = None
            async for collected in self._engine.generate(
                prompt=prompt_obj, sampling_params=sampling, request_id=request_id,
            ):
                pass
            return collected

        # Some vLLM builds accept only TokensPrompt.
        try:
            result = await _pull(prompt)
        except TypeError:
            from vllm.inputs import TokensPrompt
            request_id = f"main-{uuid4().hex}"
            result = await _pull(TokensPrompt(**prompt))
        if result is None or not result.outputs:
            raise RuntimeError(f"empty generation: {request_id}")
        item = result.outputs[0]
        if item.finish_reason not in {"stop", "length"}:
            raise RuntimeError(f"unexpected finish reason: {item.finish_reason}")
        generated = [int(token_id) for token_id in item.token_ids]
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        return Generation(token_ids=generated, text=text, finish_reason=item.finish_reason)
