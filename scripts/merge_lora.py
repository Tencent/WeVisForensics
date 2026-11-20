#!/usr/bin/env python3
"""Merge role adapters into standalone checkpoints (runs on CPU).

    python scripts/merge_lora.py --base Qwen/Qwen3.5-4B \
        --adapters ./weights/WeVisForensics-4B-LoRA --output ./weights/WeVisForensics-4B-LoRA-merged

The output directory can be served with scripts/serve_full.sh (MODEL_ROOT=<output>).
"""

from __future__ import annotations

import argparse
from pathlib import Path

ROLES = ("main", "physical_consistency", "visual_appearance", "common_sense", "ocr_consistency")


def adapter_path(adapters: Path, role: str) -> Path:
    """The main adapter sits at the release root, the inspectors in one sub-directory each."""
    return adapters if role == "main" else adapters / role


def merge(base: str, adapter: Path, output: Path) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    output.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="4GB")
    AutoProcessor.from_pretrained(base).save_pretrained(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapters", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--roles", nargs="+", default=list(ROLES))
    args = parser.parse_args()
    for role in args.roles:
        print(f"merging {role} ...", flush=True)
        merge(args.base, adapter_path(args.adapters, role), args.output / role)


if __name__ == "__main__":
    main()
