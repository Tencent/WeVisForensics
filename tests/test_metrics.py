"""Strict scoring, tool_error exclusion, and MintVid official expansion."""

import json

from evaluation.mintvid import expand_mintvid
from sub_agents.ocr import parse_stage2
from core.metrics import split_tool_errors, summarize


def test_strict_counts_invalid_model_output():
    records = [
        {"label": "fake", "answer": "fake", "status": "answered"},
        {"label": "real", "answer": None, "status": "invalid_output"},
    ]
    metrics = summarize(records)["__all__"]
    assert metrics["rows"] == 2
    assert metrics["invalid"] == 1
    assert metrics["strict"]["accuracy"] == 0.5
    assert metrics["valid_only"]["accuracy"] == 1.0


def test_tool_error_is_excluded_from_strict():
    records = [
        {"label": "fake", "answer": "fake", "status": "answered"},
        {"label": "real", "answer": None, "status": "tool_error"},
        {"label": "fake", "answer": None, "status": "specialist_invalid_output"},
    ]
    scored, errors = split_tool_errors(records)
    assert errors == 1
    metrics = summarize(scored)["__all__"]
    assert metrics["rows"] == 2
    assert metrics["strict"]["accuracy"] == 0.5


def test_mintvid_repeats_real_and_skips_tool_error(tmp_path):
    official = [
        {"label": 0, "videos": ["MintVid/real/a.mp4", "MintVid/real/a.mp4"]},
        {"label": 1, "videos": ["MintVid/fake/b.mp4"]},
    ]
    for name in ("hailuo", "jimeng", "kling", "seedance", "sora2", "wan25"):
        payload = official if name == "hailuo" else []
        (tmp_path / f"{name}.json").write_text(json.dumps(payload), encoding="utf-8")
    records = [
        {"video": "/data/MintVid/real/a.mp4", "label": "real", "answer": "real", "status": "answered"},
        {"video": "/data/MintVid/fake/b.mp4", "label": "fake", "answer": None, "status": "tool_error"},
    ]
    report = expand_mintvid(records, tmp_path)
    assert report["expanded_rows"] == 2
    assert report["tool_errors_excluded"] == 1
    assert report["micro"]["strict"]["n"] == 2
    assert report["micro"]["strict"]["accuracy"] == 1.0
    assert report["by_subset"]["hailuo"]["strict"]["n"] == 2


def test_ocr_stage2_keeps_raw_text_when_json_fails():
    failed = parse_stage2("not json at all")
    assert failed["reason"] == "not json at all"
    assert failed["verdict"] == "inconclusive"
    omitted = parse_stage2('{"verdict":"text_ok","confidence":"high"}')
    assert omitted["reason"] == "OCR VLM omitted a separate reason."
