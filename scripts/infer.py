#!/usr/bin/env python3
"""Run the forensic agent on a single video and print the trajectory."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from main_agent import build_agent, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video")
    parser.add_argument("--config", default="configs/multi_server.json")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--main-backend", choices=("engine", "http"), default="engine")
    parser.add_argument("--json", action="store_true", help="print the full trajectory as JSON")
    args = parser.parse_args()

    agent = build_agent(load_config(args.config), cache_root=args.cache_root, main_backend=args.main_backend)
    trajectory = asyncio.run(agent.run(args.video))
    if args.json:
        print(json.dumps(trajectory.to_dict(), ensure_ascii=False, indent=2))
        return
    print(f"Video: {trajectory.video}  ({trajectory.duration:.1f}s, audio={'yes' if trajectory.has_audio else 'no'})")
    for step in trajectory.steps:
        action = step["action"] or {}
        print(f"\n[Step {step['turn']}] {json.dumps(action, ensure_ascii=False)}")
        if step.get("reasoning"):
            print(f"  reasoning: {step['reasoning']}")
        if step.get("observation"):
            print("  " + step["observation"].replace("\n", "\n  "))
        if step["errors"]:
            print(f"  errors: {step['errors']}")
    print(f"\nStatus: {trajectory.status}\nVerdict: {trajectory.answer or 'invalid'}")


if __name__ == "__main__":
    main()
