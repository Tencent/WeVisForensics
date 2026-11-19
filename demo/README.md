---
title: WeVisForensics
emoji: 🔍
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: 6.0.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Agentic detection of AI-generated video
---

# WeVisForensics demo

Upload a video, choose up to 16 s, and the investigator agent calls appearance, physics, commonsense, OCR and audio
inspectors on the spans it picks before giving a verdict. Code: <https://github.com/Tencent/WeVisForensics>.

## Running

The Space runs the whole agent in-process on one GPU (ZeroGPU or a dedicated GPU with at least 24 GB):

- the main agent and the four visual inspectors share one copy of `Qwen/Qwen3.5-4B`; before each call the
  role's adapter is merged into its weights, so generation runs without adapter overhead;
- speech analysis uses an untouched copy of the base model; OCR runs on CPU with oneDNN.

Settings (Space variables / secrets):

| Name | Default | Purpose |
| --- | --- | --- |
| `HF_TOKEN` | — | Secret. Needed for the gated `pyannote/speaker-diarization-community-1` |
| `WVF_ADAPTERS` | `tencent/WeVisForensics` | Adapter repo or local directory |
| `WVF_BASE` | `Qwen/Qwen3.5-4B` | Base model |
| `WVF_GPU_SECONDS` | `120` | Upper bound of the ZeroGPU time requested per analysis |
| `WVF_OCR_DEVICE` | `cpu` | PaddleOCR device; ZeroGPU only serves PyTorch, so keep `cpu` there |
| `WVF_OCR_THREADS` | available CPUs, at most 8 | CPU threads for OCR |
| `WVF_CONCURRENCY` | `1` | Parallel analyses |
| `WVF_LOCAL` | `1` | Set to `0` to skip model loading and only browse the recorded examples (no GPU) |

To deploy, upload the contents of this folder as the Space root; the `wevisforensics` package is installed from
GitHub through `requirements.txt`. From a repository checkout, `python demo/app.py` uses the local packages instead.
Third-party speech and speaker models keep their own licenses.
