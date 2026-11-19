"""ocr_consistency: PP-OCR localization, text-presence routing and crop-level glyph check."""

from __future__ import annotations

import asyncio
import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from main_agent.protocol import CONFIDENCES, OCR_CROP_MARGIN, OCR_CROP_MAX_PIXELS, OCR_VERDICTS
from sub_agents.prompts import (
    OCR_MAX_CROP_REGIONS,
    OCR_STAGE1_PROMPT,
    OCR_STAGE1_SYSTEM,
    OCR_STAGE2_PROMPT,
    OCR_STAGE2_SYSTEM,
)
from core.client import ChatEndpoint, user_with_images
from core.common import SpecialistOutputError, ToolError, format_seconds, try_parse_json
from core.media import FrameCache


class PaddleOCRBackend:
    """Process-local PP-OCR pipeline returning per-line boxes and confidences."""

    def __init__(
        self, *, device: str = "auto", lang: str = "ch", ocr_version: str = "PP-OCRv5",
        det_model_dir: str | None = None, rec_model_dir: str | None = None,
    ):
        if bool(det_model_dir) != bool(rec_model_dir):
            raise ValueError("det_model_dir and rec_model_dir must be set together")
        self.device = device
        self.lang = lang
        self.ocr_version = ocr_version
        self.det_model_dir = det_model_dir
        self.rec_model_dir = rec_model_dir
        self._model: Any = None
        self._lock = threading.Lock()

    def _resolve_device(self) -> str:
        requested = self.device.strip().lower()
        if requested == "auto":
            import paddle

            has_gpu = paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0
            return "gpu:0" if has_gpu else "cpu"
        if requested in {"cuda", "gpu"}:
            return "gpu:0"
        if requested.startswith("cuda:"):
            return "gpu:" + requested.split(":", 1)[1]
        return requested

    def _get_model(self):
        if self._model is not None:
            return self._model
        from paddleocr import PaddleOCR

        kwargs: dict[str, Any] = {
            "device": self._resolve_device(),
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
        }
        if self.det_model_dir:
            for prefix, value in (("text_detection", self.det_model_dir), ("text_recognition", self.rec_model_dir)):
                path = Path(value).expanduser().resolve()
                header = (path / "inference.yml").read_text(encoding="utf-8")[:4096]
                match = re.search(r"(?m)^\s*model_name:\s*([^#\n]+?)\s*$", header)
                if not match:
                    raise ValueError(f"cannot resolve model_name from {path / 'inference.yml'}")
                kwargs[f"{prefix}_model_name"] = match.group(1).strip(" '\"")
                kwargs[f"{prefix}_model_dir"] = str(path)
        else:
            kwargs["lang"] = self.lang
            kwargs["ocr_version"] = self.ocr_version
        self._model = PaddleOCR(**kwargs)
        return self._model

    def __call__(self, image_path: str) -> list[dict[str, Any]]:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size
        with self._lock:
            results = list(self._get_model().predict(input=str(image_path), text_rec_score_thresh=0.0))
        if not results:
            return []
        return _regions(_result_dict(results[0]), width, height)


def _plain(value: Any) -> Any:
    return value.tolist() if hasattr(value, "tolist") else value


def _result_dict(result: Any) -> dict[str, Any]:
    candidates = [result]
    for name in ("json", "res", "to_dict"):
        if not hasattr(result, name):
            continue
        value = getattr(result, name)
        try:
            candidates.append(value() if callable(value) else value)
        except TypeError:
            continue
    for candidate in candidates:
        if isinstance(candidate, str):
            try:
                candidate = json.loads(candidate)
            except json.JSONDecodeError:
                continue
        if isinstance(candidate, dict):
            payload = candidate.get("res", candidate)
            if isinstance(payload, dict):
                return payload
    raise TypeError(f"unsupported PaddleOCR result type: {type(result).__name__}")


def _bbox(poly: Iterable[Iterable[float]], width: int, height: int) -> list[float]:
    points = [[float(x), float(y)] for x, y in poly]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return [
        max(0.0, min(1.0, min(xs) / width)),
        max(0.0, min(1.0, min(ys) / height)),
        max(0.0, min(1.0, max(xs) / width)),
        max(0.0, min(1.0, max(ys) / height)),
    ]


def _iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _regions(payload: dict[str, Any], width: int, height: int) -> list[dict[str, Any]]:
    texts = list(_plain(payload.get("rec_texts", [])) or [])
    rec_scores = list(_plain(payload.get("rec_scores", [])) or [])
    rec_polys = list(_plain(payload.get("rec_polys", [])) or [])
    dt_scores = list(_plain(payload.get("dt_scores", [])) or [])
    dt_boxes = [_bbox(poly, width, height) for poly in list(_plain(payload.get("dt_polys", [])) or [])]
    output = []
    for index in range(min(len(texts), len(rec_scores), len(rec_polys))):
        bbox = _bbox(rec_polys[index], width, height)
        detection_score = None
        if dt_boxes and dt_scores:
            best = max(range(min(len(dt_boxes), len(dt_scores))), key=lambda i: _iou(bbox, dt_boxes[i]))
            if _iou(bbox, dt_boxes[best]) > 0:
                detection_score = float(dt_scores[best])
        output.append({
            "index": index,
            "text": str(texts[index]),
            "bbox": bbox,
            "detection_score": detection_score,
            "recognition_score": float(rec_scores[index]),
        })
    return output


def filter_regions(regions: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop zero-score regions and re-index the rest from 0."""
    filtered = []
    for region in regions:
        if float(region.get("recognition_score") or 0.0) <= 0.0:
            continue
        filtered.append({**region, "index": len(filtered)})
    return filtered


def format_regions(regions: Iterable[dict[str, Any]]) -> str:
    lines = []
    for region in regions:
        bbox = ",".join(f"{value:.3f}" for value in region["bbox"])
        det, rec = region.get("detection_score"), region.get("recognition_score")
        det_text = "n/a" if det is None else f"{float(det):.3f}"
        rec_text = "n/a" if rec is None else f"{float(rec):.3f}"
        lines.append(
            f'[{region["index"]}] text={str(region.get("text") or "")!r} bbox=[{bbox}] '
            f"det_conf={det_text} rec_conf={rec_text}"
        )
    return "\n".join(lines)


def parse_stage1(value: Any, *, region_count: int) -> dict[str, Any]:
    expected = {"verdict", "inspect_region_indices"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"OCR stage 1 must return exactly {sorted(expected)}")
    verdict = str(value.get("verdict") or "").strip()
    if verdict not in {"no_text", "has_text"}:
        raise ValueError(f"invalid OCR stage 1 verdict {verdict!r}")
    if not isinstance(value.get("inspect_region_indices"), list):
        raise ValueError("inspect_region_indices must be a list")
    selected: list[int] = []
    for item in value["inspect_region_indices"]:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= index < region_count and index not in selected:
            selected.append(index)
        if len(selected) >= OCR_MAX_CROP_REGIONS:
            break
    if verdict == "no_text" and selected:
        raise ValueError("no_text requires an empty region list")
    if verdict == "has_text" and not selected:
        raise ValueError("has_text requires at least one valid region")
    return {"verdict": verdict, "inspect_region_indices": selected}


def parse_stage2(raw: str) -> dict[str, Any]:
    candidate = try_parse_json(raw)
    parsed = candidate if isinstance(candidate, dict) else {}
    verdict = str(parsed.get("verdict", "inconclusive"))
    if verdict not in {"text_ok", "text_suspicious", "inconclusive"}:
        verdict = "inconclusive"
    confidence = str(parsed.get("confidence", "low"))
    if confidence not in CONFIDENCES:
        confidence = "low"
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = []
    assessments = parsed.get("region_assessments")
    if parsed.get("reason"):
        reason = str(parsed["reason"])
    else:
        reasons = [
            str(item.get("reason")) for item in assessments
            if isinstance(item, dict) and item.get("reason")
        ] if isinstance(assessments, list) else []
        reason = "; ".join(reasons) or (raw.strip() if not parsed else "OCR VLM omitted a separate reason.")
    if verdict == "text_suspicious":
        degraded = isinstance(assessments, list) and any(
            isinstance(item, dict) and item.get("status") in {"warped", "generation_degraded"}
            for item in assessments
        )
        if not degraded:
            verdict, confidence = "inconclusive", "low"
            issues = [*issues, "text_suspicious rejected because no crop was assessed as warped or generation_degraded"]
    return {"verdict": verdict, "confidence": confidence, "issues": issues, "reason": reason}


def padded_box(region: dict[str, Any]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = region["bbox"]
    pad_x, pad_y = (x2 - x1) * OCR_CROP_MARGIN, (y2 - y1) * OCR_CROP_MARGIN
    x1, y1 = max(0.0, x1 - pad_x), max(0.0, y1 - pad_y)
    x2, y2 = min(1.0, x2 + pad_x), min(1.0, y2 + pad_y)
    return x1, y1, x2 - x1, y2 - y1


async def run_ocr_specialist(
    action: dict,
    *,
    video_path: str,
    endpoint: ChatEndpoint,
    frames: FrameCache,
    ocr: PaddleOCRBackend,
    max_tokens: int,
) -> tuple[str, dict]:
    at_sec = float(action["at_sec"])
    paths, actual = await asyncio.to_thread(frames.extract, video_path, [at_sec], max_pixels=0)
    if not paths:
        raise ToolError(f"ocr_consistency: could not decode a frame at {at_sec}s")
    frame, actual_sec = paths[0], actual[0]
    scope = f"@ {format_seconds(actual_sec)}s"
    try:
        regions = filter_regions(await asyncio.to_thread(ocr, frame))
    except Exception as exc:
        raise ToolError(f"PaddleOCR failed: {type(exc).__name__}: {exc}") from exc
    trace: dict[str, Any] = {"at_sec": actual_sec, "regions": regions}
    if not regions:
        return f"Skill: ocr_consistency | {scope} | Verdict: no_text (low)\nPP-OCR returned no localized text.", trace

    stage1_messages = [
        {"role": "system", "content": OCR_STAGE1_SYSTEM},
        user_with_images(OCR_STAGE1_PROMPT.format(regions=format_regions(regions)), [frame]),
    ]
    stage1_raw = await endpoint.chat(stage1_messages, max_tokens=max_tokens)
    trace["stage1_raw"] = stage1_raw.text
    stage1_parsed = try_parse_json(stage1_raw.text)
    try:
        stage1 = parse_stage1(stage1_parsed if isinstance(stage1_parsed, dict) else {}, region_count=len(regions))
    except ValueError as exc:
        raise SpecialistOutputError(f"ocr_consistency: {exc}") from exc
    if stage1["verdict"] == "no_text":
        return (
            f"Skill: ocr_consistency | {scope} | Verdict: no_text (low)\n"
            "The Stage-1 visual router judged every PP-OCR candidate to be a false positive."
        ), trace

    selected = stage1["inspect_region_indices"]
    crops, checked = [], []
    for index in selected:
        crop = await asyncio.to_thread(
            frames.crop, frame, padded_box(regions[index]), max_pixels=OCR_CROP_MAX_PIXELS,
        )
        crops.append(crop)
        checked.append(regions[index])
    stage2_messages = [
        {"role": "system", "content": OCR_STAGE2_SYSTEM},
        user_with_images(OCR_STAGE2_PROMPT.format(regions=format_regions(checked)), [frame, *crops]),
    ]
    stage2_raw = await endpoint.chat(stage2_messages, max_tokens=max_tokens)
    trace["stage2_raw"] = stage2_raw.text
    stage2 = parse_stage2(stage2_raw.text)
    verdict = stage2["verdict"] if stage2["verdict"] in OCR_VERDICTS else "inconclusive"
    summary = (
        f"ocr_consistency {verdict}({stage2['confidence']}) over {len(regions)} localized regions; "
        f"crop-double-checked regions {selected}."
    )
    if stage2["issues"]:
        summary += f" issues={stage2['issues']}."
    reason = stage2["reason"].strip()
    if reason:
        summary += f" reason: {reason}"
    text = f"Skill: ocr_consistency | {scope} | Verdict: {verdict} ({stage2['confidence']})\n{summary}"
    return text, trace
