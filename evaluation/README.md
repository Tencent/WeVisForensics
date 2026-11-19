# Evaluation

## Manifest

One JSON object per line:

```json
{"video": "videos/0001.mp4", "label": "fake", "source": "sora2"}
{"video": "videos/0002.mp4", "label": "real", "source": "pexels"}
```

`label` is `real` or `fake`. `source` is optional and only groups the report. Optional `duration_seconds` and `has_audio` are used as given; otherwise they are read from the file. Relative paths are resolved from `--video-root` or from the JSONL directory.

## Download

Obtain the public releases of GenBuster-Bench, GenBuster-Bench++, and MintVid, then point `video` at the local files.

## Scoring

```bash
python scripts/evaluate.py \
  --data bench.jsonl \
  --output outputs/bench \
  --config configs/multi_server.json \
  --main-backend engine \
  --mintvid-json-dir /path/to/mintvid/json
```

`--retry-errors` defaults to on. Rows that still end as `tool_error` are counted separately and left out of the strict denominator. Invalid model outputs stay in the denominator and count as wrong.

`--mintvid-json-dir` expands Real videos by the repeat count in the official JSON, then reports micro accuracy. That is the MintVid number. GenBuster and GenBuster-Bench++ stay one-row-per-video in the same `metrics.json` under `__all__` and per `source`.
