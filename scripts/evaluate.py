#!/usr/bin/env python3
"""Evaluate the forensic agent on a JSONL benchmark.

Each input line: {"video": "path/to/video.mp4", "label": "real"|"fake", "id": ..., "source": ...}
``id`` defaults to the video path; ``source`` is optional and used for per-subset metrics.
Relative video paths are resolved against --video-root (default: the JSONL directory).

``tool_error`` rows are retried by default and, if they still fail, are reported
separately. They are not part of the strict accuracy denominator. A model that
returns an invalid action is still counted wrong.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluation.mintvid import expand_mintvid  # noqa: E402
from core.common import ToolError  # noqa: E402
from core.metrics import split_tool_errors, summarize  # noqa: E402
from main_agent import build_agent, load_config  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


async def evaluate(args: argparse.Namespace) -> None:
    data_path = Path(args.data)
    video_root = Path(args.video_root) if args.video_root else data_path.parent
    rows = read_jsonl(data_path)
    for row in rows:
        row["label"] = str(row["label"]).strip().lower()
        if row["label"] not in {"real", "fake"}:
            raise ValueError(f"label must be real/fake: {row}")
        video = Path(row["video"])
        row["video"] = str(video if video.is_absolute() else video_root / video)
        row.setdefault("id", row["video"])

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    done = {}
    if predictions_path.exists():
        for record in read_jsonl(predictions_path):
            if args.retry_errors and record.get("status") == "tool_error":
                continue
            done[record["id"]] = record
    pending = [row for row in rows if row["id"] not in done]
    print(f"{len(rows)} videos, {len(done)} already evaluated, {len(pending)} to run", flush=True)

    agent = build_agent(
        load_config(args.config), cache_root=args.cache_root, main_backend=args.main_backend,
    )
    semaphore = asyncio.Semaphore(args.concurrency)
    lock = asyncio.Lock()
    started = time.monotonic()
    finished = 0

    async def run_one(row: dict) -> None:
        nonlocal finished
        async with semaphore:
            record = {key: row[key] for key in ("id", "video", "label") if key in row}
            if "source" in row:
                record["source"] = row["source"]
            try:
                trajectory = await agent.run(
                    row["video"], duration=row.get("duration_seconds"), has_audio=row.get("has_audio"),
                )
                record.update(trajectory.to_dict())
            except ToolError as exc:
                record.update(status="tool_error", answer=None, error=f"{type(exc).__name__}: {exc}")
            except Exception as exc:
                record.update(
                    status="tool_error", answer=None, error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(limit=5),
                )
            record["correct"] = record.get("answer") == row["label"] if record.get("status") != "tool_error" else None
            async with lock:
                with predictions_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                done[row["id"]] = record
                finished += 1
                if finished % args.log_every == 0 or finished == len(pending):
                    elapsed = time.monotonic() - started
                    print(f"[{finished}/{len(pending)}] {elapsed:.0f}s elapsed", flush=True)

    await asyncio.gather(*(run_one(row) for row in pending))

    records = [done[row["id"]] for row in rows if row["id"] in done]
    scored, errors = split_tool_errors(records)
    metrics = summarize(scored)
    metrics["__meta__"] = {
        "tool_errors": errors,
        "tool_errors_excluded_from_strict": True,
        "config": str(args.config),
        "data": str(data_path),
        "main_backend": args.main_backend,
    }
    if args.mintvid_json_dir:
        metrics["mintvid_official_repeated"] = expand_mintvid(records, args.mintvid_json_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n")
    overall = metrics["__all__"]
    summary = {
        "rows": overall["rows"],
        "invalid": overall["invalid"],
        "tool_errors": errors,
        "accuracy_strict": overall["strict"]["accuracy"],
        "accuracy_valid_only": overall["valid_only"]["accuracy"],
    }
    if args.mintvid_json_dir:
        summary["mintvid_micro_strict"] = metrics["mintvid_official_repeated"]["micro"]["strict"]["accuracy"]
        summary["mintvid_expanded_rows"] = metrics["mintvid_official_repeated"]["expanded_rows"]
    print(json.dumps(summary, indent=2))
    if errors:
        print(f"{errors} videos failed with tool errors and were excluded from accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/multi_server.json")
    parser.add_argument("--video-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--main-backend", choices=("engine", "http"), default="engine")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--retry-errors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mintvid-json-dir", default=None, help="official MintVid JSON directory")
    parser.add_argument("--log-every", type=int, default=20)
    asyncio.run(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()
