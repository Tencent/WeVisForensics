# Deployment

WeVisForensics runs as one investigator process plus one vLLM server per inspector.

1. Download the base model and the LoRA adapters. See `checkpoints/README.md`.
2. Merge each adapter:

```bash
python scripts/merge_lora.py \
  --base Qwen/Qwen3.5-4B \
  --adapters weights/WeVisForensics-4B-LoRA \
  --output weights/WeVisForensics-4B-merged
```

3. Serve the inspectors:

```bash
MODEL_ROOT=weights/WeVisForensics-4B-merged ASR_MODEL=Qwen/Qwen3.5-4B \
  GPUS=0,1,2,3,4 bash scripts/serve_full.sh
```

4. Point `engine.model` in `configs/multi_server.json` at `weights/WeVisForensics-4B-merged/main`.
5. Run the agent:

```bash
python scripts/infer.py path/to/video.mp4 --config configs/multi_server.json
```

## Investigator backends

- `engine` (default): an in-process vLLM engine. Each turn appends the generated token ids and then the encoded observation, so the context is exactly what the model produced. Greedy decoding, seed 20260716, `max_model_len` 131072, CUDA Graph, `gdn_prefill_backend=triton`.
- `http`: the OpenAI-compatible chat API. Assistant turns are re-rendered through the chat template, which is convenient for debugging but not token-exact.

`scripts/serve_lora.sh` serves the base model with all adapters in one process for a quick try; pair it with `--main-backend http`.
