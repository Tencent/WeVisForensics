# Protocol

Each investigator message is one JSON object with `reasoning` then `action`.

```json
{"reasoning": "...", "action": {"type": "run_skill", "skill": "visual_appearance", "start_sec": 1.0, "end_sec": 4.0}}
{"reasoning": "...", "action": {"type": "answer", "content": "fake"}}
```

Invalid trajectories, counted wrong under strict scoring:

- an answer before any successful tool call
- a repeated tool call (same skill and timestamps)
- a tool call on the fifth step
- JSON that does not match the action schema

`tool_error` is an infrastructure failure (decode, OCR runtime, HTTP, or the main engine). Evaluation retries those rows and leaves the ones that still fail out of the accuracy denominator.

Frame grids:

- overview: 1 FPS over the full video, at most 16 frames, timestamps rounded to 0.1 s
- temporal inspectors: 2 FPS inside the requested window, at most 72 frames
- pixels per frame capped at 384 × 32 × 32, LANCZOS, JPEG quality 95, subsampling 0
- OCR uses the full frame (`max_pixels=0`) and crops padded by 15% up to 768 × 32 × 32 pixels

Decoding defaults: temperature 0, top_p 1, top_k -1, repetition penalty 1, thinking disabled, 1024 tokens per investigator step, 500 tokens per inspector call. The response budget is 12288 tokens of generated text plus observations. Observation tokens are counted before the next generation.
