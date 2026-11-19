"""MintVid General scoring with the official repeated-real expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from core.metrics import binary_metrics


GENERAL_SUBSETS = ("hailuo", "jimeng", "kling", "seedance", "sora2", "wan25")


def mintvid_relative(path: str) -> str:
    normalized = str(path).replace("\\", "/")
    if normalized.startswith("MintVid/"):
        return normalized
    marker = "/MintVid/"
    if marker not in normalized:
        raise ValueError(f"not a MintVid path: {path}")
    return "MintVid/" + normalized.rsplit(marker, 1)[1]


def expand_mintvid(
    records: list[dict[str, Any]],
    official_json_dir: str | Path,
    *,
    subsets: tuple[str, ...] = GENERAL_SUBSETS,
) -> dict[str, Any]:
    """Expand Real references by their official repeat count, then score micro accuracy.

    ``tool_error`` rows are omitted from the denominator. A video with no prediction
    still counts as wrong. Model-invalid answers stay in the denominator.
    """
    root = Path(official_json_dir)
    predictions: dict[str, dict[str, Any]] = {}
    for record in records:
        key = mintvid_relative(str(record.get("video") or record.get("video_path")))
        predictions[key] = record
    repeated: list[dict[str, Any]] = []
    by_subset: dict[str, Any] = {}
    missing: list[str] = []
    excluded = 0
    for subset in subsets:
        official = json.loads((root / f"{subset}.json").read_text(encoding="utf-8"))
        expanded = []
        for item in official:
            label = "fake" if int(item["label"]) == 1 else "real"
            for video in item["videos"]:
                key = mintvid_relative(str(video))
                prediction = predictions.get(key)
                if prediction is None:
                    missing.append(key)
                    expanded.append({"label": label, "answer": None})
                    continue
                if prediction.get("status") == "tool_error":
                    excluded += 1
                    continue
                expanded.append({"label": label, "answer": prediction.get("answer")})
        by_subset[subset] = binary_metrics((row["label"], row["answer"]) for row in expanded)
        repeated.extend(expanded)
    micro = binary_metrics((row["label"], row["answer"]) for row in repeated)
    fake_recall = micro["strict"]["fake"]["recall"] or 0.0
    real_recall = micro["strict"]["real"]["recall"] or 0.0
    subset_accuracies = [
        entry["strict"]["accuracy"] for entry in by_subset.values() if entry["strict"]["accuracy"] is not None
    ]
    return {
        "protocol": "six General official JSONs; repeated Real references; invalid/missing as wrong",
        "expanded_rows": len(repeated),
        "missing_prediction_rows": len(missing),
        "tool_errors_excluded": excluded,
        "micro": micro,
        "macro_balanced_accuracy": (fake_recall + real_recall) / 2 if repeated else None,
        "mean_subset_accuracy": sum(subset_accuracies) / len(subset_accuracies) if subset_accuracies else None,
        "by_subset": by_subset,
    }
