"""Video probing, timestamped frame decoding and crop extraction."""

from __future__ import annotations

import fcntl
import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from PIL import Image

from .common import ToolError


MIN_PIXELS = 128 * 32 * 32
MAX_ASPECT_RATIO = 200
SEEK_PREROLL_SECONDS = 1.0
SEQUENTIAL_GAP_SECONDS = 2.5
_FORMAT_VERSION = 1


def default_cache_root() -> Path:
    return Path(tempfile.gettempdir()) / "wevisforensics_cache"


def probe_media(video_path: str) -> tuple[float, bool]:
    """Return (duration_seconds, has_audio) with a single PyAV open."""
    import av

    with av.open(video_path, mode="r") as container:
        has_audio = bool(container.streams.audio)
        if not container.streams.video:
            raise ToolError(f"no video stream: {video_path}")
        duration = None
        if container.duration is not None:
            duration = float(container.duration) / float(av.time_base)
        if duration is None or duration <= 0:
            stream = container.streams.video[0]
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise ToolError(f"cannot determine a positive duration: {video_path}")
        return duration, has_audio


def fit_within_pixels(width: int, height: int, max_pixels: int, factor: int = 32) -> tuple[int, int]:
    """Qwen-VL ``smart_resize`` applied before frames are written to disk."""
    if max_pixels <= 0:
        return width, height
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio must be smaller than {MAX_ASPECT_RATIO}")
    resized_h = max(factor, round(height / factor) * factor)
    resized_w = max(factor, round(width / factor) * factor)
    if resized_h * resized_w > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_h = max(factor, math.floor(height / beta / factor) * factor)
        resized_w = max(factor, math.floor(width / beta / factor) * factor)
    elif resized_h * resized_w < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        resized_h = max(factor, math.ceil(height * beta / factor) * factor)
        resized_w = max(factor, math.ceil(width * beta / factor) * factor)
    return resized_w, resized_h


def _resize(image: Image.Image, max_pixels: int) -> Image.Image:
    image = image.convert("RGB")
    if max_pixels <= 0:
        return image
    size = fit_within_pixels(image.width, image.height, max_pixels)
    if size == image.size:
        return image
    return image.resize(size, Image.Resampling.LANCZOS)


def _atomic_write(output: Path, writer) -> None:
    temporary = output.with_name(f".{output.name}.{os.getpid()}.{uuid4().hex}.tmp")
    try:
        writer(temporary)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _save_jpeg(image: Image.Image, output: Path) -> None:
    _atomic_write(output, lambda path: image.save(path, format="JPEG", quality=95, subsampling=0))


def _decode_group(container, stream, targets: list[float]) -> dict[float, tuple[Image.Image, float]]:
    """Return, for every target, the first decodable frame at or after it."""
    import av

    seek_sec = max(0.0, targets[0] - SEEK_PREROLL_SECONDS)
    container.seek(int(seek_sec * av.time_base), backward=True, any_frame=False)
    pending = list(targets)
    selected: dict[float, tuple[Image.Image, float]] = {}
    previous = None
    for frame in container.decode(stream):
        if frame.time is None:
            continue
        current_sec = float(frame.time)
        current = (frame.to_image(), current_sec)
        while pending and current_sec >= pending[0]:
            selected[pending.pop(0)] = current
        previous = current
        if not pending:
            break
    if pending and previous is not None:
        for target in pending:
            selected[target] = previous
    return selected


def _decode_pyav(video_path: str, timestamps: list[float]) -> dict[float, tuple[Image.Image, float]]:
    import av

    groups: list[list[float]] = []
    for timestamp in sorted(set(timestamps)):
        if not groups or timestamp - groups[-1][-1] > SEQUENTIAL_GAP_SECONDS:
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    decoded: dict[float, tuple[Image.Image, float]] = {}
    with av.open(video_path, mode="r") as container:
        stream = container.streams.video[0]
        stream.thread_count = 1
        stream.thread_type = "FRAME"
        for group in groups:
            decoded.update(_decode_group(container, stream, group))
    return decoded


def _decode_ffmpeg(video_path: str, timestamps: list[float]) -> dict[float, tuple[Image.Image, float]]:
    """Slow fallback for streams that PyAV cannot decode."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ToolError("PyAV failed to decode frames and ffmpeg is not installed")
    decoded: dict[float, tuple[Image.Image, float]] = {}
    for timestamp in sorted(set(timestamps)):
        command = [
            ffmpeg, "-hide_banner", "-nostdin", "-loglevel", "error", "-err_detect", "ignore_err",
            "-threads", "1", "-ss", f"{timestamp:.6f}", "-i", video_path, "-frames:v", "1",
            "-f", "image2pipe", "-vcodec", "png", "pipe:1",
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if not result.stdout:
            detail = result.stderr.decode("utf-8", errors="replace")[-1000:]
            raise ToolError(f"ffmpeg produced no frame at {timestamp:.6f}s: {detail}")
        with Image.open(io.BytesIO(result.stdout)) as image:
            decoded[timestamp] = (image.convert("RGB"), timestamp)
    return decoded


class FrameCache:
    """Deterministic on-disk frame cache, safe across threads and processes."""

    def __init__(self, root: str | Path | None = None):
        self.root = Path(root) if root else default_cache_root()

    def _video_dir(self, video_path: str) -> Path:
        path = Path(video_path).expanduser().resolve()
        stat = path.stat()
        key = hashlib.sha256(f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()[:24]
        return self.root / "frames" / key

    def extract(
        self, video_path: str, timestamps: Iterable[float], *, max_pixels: int,
    ) -> tuple[list[str], list[float]]:
        """Return frame paths and actual decoded timestamps, in request order."""
        requested = [round(float(value), 6) for value in timestamps]
        if not requested:
            return [], []
        directory = self._video_dir(video_path)
        directory.mkdir(parents=True, exist_ok=True)
        records: dict[float, dict[str, Any]] = {}
        with (directory / ".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            missing = []
            for timestamp in requested:
                stem = hashlib.sha256(
                    f"v{_FORMAT_VERSION}\0{timestamp:.6f}\0{int(max_pixels)}".encode()
                ).hexdigest()[:20]
                record = self._read_record(directory / f"{stem}.json")
                if record is None:
                    missing.append((timestamp, stem))
                else:
                    records[timestamp] = record
            if missing:
                targets = [timestamp for timestamp, _ in missing]
                try:
                    decoded = _decode_pyav(video_path, targets)
                except Exception:
                    decoded = {}
                unresolved = [t for t in targets if t not in decoded]
                if unresolved:
                    decoded.update(_decode_ffmpeg(video_path, unresolved))
                for timestamp, stem in missing:
                    item = decoded.get(timestamp)
                    if item is None:
                        continue
                    image, actual = item
                    frame_path = directory / f"{stem}.jpg"
                    _save_jpeg(_resize(image, max_pixels), frame_path)
                    record = {
                        "format_version": _FORMAT_VERSION,
                        "actual_at_sec": round(float(actual), 6),
                        "frame_path": str(frame_path),
                    }
                    _atomic_write(
                        directory / f"{stem}.json",
                        lambda path, payload=record: path.write_text(json.dumps(payload)),
                    )
                    records[timestamp] = record
        paths, actual = [], []
        for timestamp in requested:
            record = records.get(timestamp)
            if record is not None:
                paths.append(record["frame_path"])
                actual.append(float(record["actual_at_sec"]))
        return paths, actual

    @staticmethod
    def _read_record(path: Path) -> dict[str, Any] | None:
        try:
            record = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        if record.get("format_version") != _FORMAT_VERSION or not Path(record["frame_path"]).is_file():
            return None
        return record

    def crop(self, image_path: str, bbox: tuple[float, float, float, float], *, max_pixels: int) -> str:
        """Crop a decoded frame with a normalized (x, y, w, h) box."""
        source = Path(image_path).resolve()
        stat = source.stat()
        box = tuple(round(float(value), 6) for value in bbox)
        key = hashlib.sha256(
            f"{source}\0{stat.st_size}\0{stat.st_mtime_ns}\0{box}\0{max_pixels}".encode()
        ).hexdigest()[:32]
        directory = self.root / "crops"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{key}.jpg"
        with output.with_suffix(".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            if output.is_file() and output.stat().st_size > 0:
                return str(output)
            with Image.open(source) as image:
                image = image.convert("RGB")
                x, y, width, height = box
                left = max(0, min(image.width - 1, round(x * image.width)))
                top = max(0, min(image.height - 1, round(y * image.height)))
                right = max(left + 1, min(image.width, round((x + width) * image.width)))
                bottom = max(top + 1, min(image.height, round((y + height) * image.height)))
                _save_jpeg(_resize(image.crop((left, top, right, bottom)), max_pixels), output)
        return str(output)
