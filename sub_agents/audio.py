"""Audio extraction, speech recognition, and speaker-aware voice profiles."""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import uuid
import wave
from argparse import Namespace
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Any

from core.common import ToolError, format_seconds

# pyannote reads this when it is imported later.
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

_SR = 16000
_MIN_SEGMENT_SECONDS = 0.8

_WHISPER_CACHE: dict[str, Any] = {}
_WHISPER_LOCK = threading.Lock()

_PYANNOTE_CACHE: dict[tuple[str, str], Any] = {}
_PYANNOTE_MODEL_LOCK = threading.Lock()
_PYANNOTE_INFER_LOCK = threading.Lock()

_MOSS_CACHE: dict[tuple[str, str], tuple[Any, Any, Namespace, Any]] = {}
_MOSS_MODEL_LOCK = threading.Lock()
_MOSS_INFER_LOCK = threading.Lock()

_AUDEERING_CACHE: dict[tuple[str, str, str], tuple[Any, Any, str]] = {}
_AUDEERING_MODEL_LOCK = threading.Lock()
_AUDEERING_INFER_LOCK = threading.Lock()


@dataclass
class AudioConfig:
    whisper_model: str = "Systran/faster-whisper-small"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    pyannote_model: str = "pyannote/speaker-diarization-community-1"
    mossformer2_model: str = "alibabasglab/MossFormer2_SS_16K"
    audeering_emotion_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    audeering_age_gender_model: str = "audeering/wav2vec2-large-robust-24-ft-age-gender"
    device: str = "auto"


@dataclass
class _SkillReport:
    ok: bool
    verdict: str
    confidence: str
    summary: str
    evidence: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


ASR_VERDICTS: frozenset[str] = frozenset({
    "timing_anomaly",
    "timing_natural",
    "inconclusive",
})
ASR_JUDGE_SYSTEM = (
    "You are a conservative audio analyst checking only transcript and timing anomalies."
)
ASR_VLM_PROMPT = (
    "You are analyzing an ASR (automatic speech recognition) transcript from a "
    "video to check only for concrete transcript/timing anomalies.\n\n"
    "Transcript with timestamps:\n{transcript_block}\n\n"
    "Word-level timestamps:\n{word_timing_block}\n\n"
    "Audio duration: {duration}s, {n_segments} segments, detected language: {language}\n"
    "Flags from ASR engine: {flags}\n\n"
    "Potential timing/text anomalies:\n"
    "- Unnaturally regular pacing (segments are suspiciously evenly spaced)\n"
    "- Exact repetition of phrases or sentence structures\n"
    "- Missing natural disfluencies (no 'um', 'uh', false starts, self-corrections)\n"
    "- Abrupt silence gaps that don't match natural breathing patterns\n"
    "- All segments having nearly identical duration (robotic timing)\n\n"
    "Natural-looking timing/text patterns:\n"
    "- Natural variation in segment length and pacing\n"
    "- Disfluencies, hesitations, breathing pauses\n"
    "- Overlapping or interrupted speech patterns\n\n"
    "Important limits: you receive text, segment timestamps, and ASR word timestamps, not the waveform, so you "
    "cannot judge voice timbre, enunciation, breathing sounds, or background noise. Short "
    "speech, singing, scripted delivery, subtitles, ASR errors, and naturally repeated phrases "
    "are ambiguous. Return inconclusive unless timing/text contains a concrete strong pattern. "
    "Report only this transcript/timing dimension: timing_anomaly does not by itself prove "
    "synthetic audio, and timing_natural does not establish that either the audio or video is real.\n\n"
    "Reply STRICTLY in JSON:\n"
    '  {{"verdict": "timing_anomaly|timing_natural|inconclusive",\n'
    '   "confidence": "high|medium|low",\n'
    '   "issues": ["..."],\n'
    '   "reason": "within 200 words about the transcript timing patterns"}}'
)


def build_asr_review_prompt(payload: dict, duration: float) -> str:
    """Fill the ASR review prompt with transcript text and word timings."""
    segments = list(payload.get("segments") or [])
    transcript_block = "\n".join(
        f"[{format_seconds(seg['start'])}-{format_seconds(seg['end'])}s] {seg['text']}"
        for seg in segments
    )
    word_lines = []
    for segment_index, segment in enumerate(segments):
        words = list(segment.get("words") or [])
        if not words:
            word_lines.append(f"segment {segment_index}: <no word timestamps>")
            continue
        word_lines.append(
            f"segment {segment_index}: " + " | ".join(
                f"[{format_seconds(word['start'])}-{format_seconds(word['end'])}s] {word['word']}"
                for word in words
            )
        )
    return ASR_VLM_PROMPT.format(
        transcript_block=transcript_block,
        word_timing_block="\n".join(word_lines) or "none",
        duration=format_seconds(duration),
        n_segments=len(segments),
        language=payload.get("language", "unknown"),
        flags=", ".join(payload.get("flags") or []) or "none",
    )


def _ffmpeg_bin() -> str:
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("ffmpeg not found in PATH")
    return found


def _run_ffmpeg(args: list[str]) -> None:
    subprocess.check_call([_ffmpeg_bin(), "-y", "-loglevel", "error", *args])


def _pyav_media_info(video_path: str) -> tuple[float, bool]:
    """Duration in seconds and whether the container has an audio stream."""
    import av

    with av.open(video_path, mode="r") as container:
        has_audio = bool(container.streams.audio)
        duration: float | None = None
        if container.duration is not None:
            duration = float(container.duration) / float(av.time_base)
        if duration is None or duration <= 0:
            stream = next((item for item in container.streams if item.type == "video"), None)
            if (
                stream is not None
                and stream.duration is not None
                and stream.time_base is not None
            ):
                duration = float(stream.duration * stream.time_base)
        if duration is None or not math.isfinite(duration) or duration <= 0:
            raise RuntimeError(f"PyAV found no positive media duration: {video_path}")
        return duration, has_audio


def _extract_audio(
    video_path: str,
    work_dir: str | Path,
    *,
    start_sec: float = 0.0,
    end_sec: float = -1.0,
    sample_rate: int = 16000,
) -> dict[str, Any]:
    try:
        duration, source_has_audio = _pyav_media_info(video_path)
    except Exception as exc:
        raise ToolError(f"media probe failed: {exc}") from exc
    if not source_has_audio:
        raise ToolError("no audio stream")
    start_sec = max(0.0, float(start_sec))
    if end_sec is None or end_sec < 0:
        end_sec = duration if duration > 0 else 0.0
    else:
        end_sec = float(end_sec)
        if duration > 0:
            end_sec = min(end_sec, duration)
    if end_sec <= start_sec:
        raise ToolError("end<=start")

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    out = work / f"{Path(video_path).stem}_audio_{start_sec:.2f}_{end_sec:.2f}.wav"
    temporary = work / f".{out.stem}.{uuid.uuid4().hex}.tmp.wav"
    args = [
        "-i", str(video_path),
        "-ss", f"{start_sec:.3f}",
        "-to", f"{end_sec:.3f}",
        "-vn",
        "-ac", "1",
        "-ar", str(int(sample_rate)),
        "-c:a", "pcm_s16le",
        str(temporary),
    ]
    try:
        _run_ffmpeg(args)
        if not temporary.is_file() or temporary.stat().st_size <= 44:
            raise RuntimeError("ffmpeg produced an empty or truncated WAV")
        with wave.open(str(temporary), "rb") as reader:
            if reader.getnchannels() != 1:
                raise RuntimeError(f"unexpected WAV channels: {reader.getnchannels()}")
            if reader.getframerate() != int(sample_rate):
                raise RuntimeError(f"unexpected WAV sample rate: {reader.getframerate()}")
            if reader.getsampwidth() != 2 or reader.getnframes() <= 0:
                raise RuntimeError("WAV contains no valid PCM16 audio frames")
        os.replace(temporary, out)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise ToolError(f"invalid audio artifact: {exc}") from exc
    return {
        "audio_path": str(out),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sample_rate": int(sample_rate),
    }


def extract_audio(
    video_path: str,
    work_dir: str | Path,
    *,
    start_sec: float = 0.0,
    end_sec: float = -1.0,
    sample_rate: int = 16000,
) -> str:
    """Extract a mono PCM16 WAV. Returns the WAV path."""
    return _extract_audio(
        video_path,
        work_dir,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate=sample_rate,
    )["audio_path"]


def _whisper_compute_type(config: AudioConfig) -> str:
    if config.whisper_compute_type:
        return config.whisper_compute_type
    return "int8" if config.whisper_device == "cpu" else "float16"


def _load_whisper(config: AudioConfig) -> Any:
    compute = _whisper_compute_type(config)
    key = f"{config.whisper_model}|{config.whisper_device}|{compute}"
    cached = _WHISPER_CACHE.get(key)
    if cached is not None:
        return cached
    with _WHISPER_LOCK:
        cached = _WHISPER_CACHE.get(key)
        if cached is not None:
            return cached
        from faster_whisper import WhisperModel

        model = WhisperModel(
            config.whisper_model,
            device=config.whisper_device,
            compute_type=compute,
        )
        _WHISPER_CACHE[key] = model
        return model


def _asr_flags(segments: list[dict[str, Any]], transcript: str) -> list[str]:
    flags: list[str] = []
    if segments:
        texts = [segment["text"] for segment in segments if segment["text"]]
        if texts and len(set(texts)) == 1 and len(texts) >= 3:
            flags.append("all_segments_identical")
        elif texts and len(set(texts)) < len(texts) * 0.4:
            flags.append("high_repetition")
    if not transcript:
        flags.append("empty_transcript")
    return flags


def transcribe(
    video_path: str,
    work_dir: str | Path,
    config: AudioConfig,
    *,
    start_sec: float = 0.0,
    end_sec: float = -1.0,
) -> dict:
    """Transcribe speech. Timestamps are source-absolute and include words."""
    extracted = _extract_audio(
        video_path,
        work_dir,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate=_SR,
    )
    try:
        model = _load_whisper(config)
    except Exception as exc:
        raise ToolError(str(exc)) from exc
    try:
        segments_iter, info = model.transcribe(
            extracted["audio_path"],
            language=None,
            word_timestamps=True,
            vad_filter=True,
        )
        fw_segments = list(segments_iter)
    except Exception as exc:
        raise ToolError(str(exc)) from exc

    offset = float(extracted["start_sec"])
    segments = []
    for seg in fw_segments:
        words = []
        for word in (getattr(seg, "words", None) or []):
            words.append({
                "start": float(getattr(word, "start", 0.0) or 0.0) + offset,
                "end": float(getattr(word, "end", 0.0) or 0.0) + offset,
                "word": (getattr(word, "word", "") or "").strip(),
            })
        segments.append({
            "start": float(getattr(seg, "start", 0.0) or 0.0) + offset,
            "end": float(getattr(seg, "end", 0.0) or 0.0) + offset,
            "text": (getattr(seg, "text", "") or "").strip(),
            "words": words,
        })
    transcript = " ".join(segment["text"] for segment in segments).strip()
    return {
        "transcript": transcript,
        "language": getattr(info, "language", "") or "",
        "segments": segments,
        "flags": _asr_flags(segments, transcript),
        "audio_path": extracted["audio_path"],
    }


def _torch_device_name(requested: str) -> str:
    import torch

    name = requested.strip().lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"invalid device={requested!r}")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda but CUDA is unavailable")
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return "cuda"
    return "cpu"


def _device_key(requested: str) -> str:
    name = requested.strip().lower()
    if name not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"invalid device={requested!r}")
    return name


def _annotation_turns(output: Any) -> list[dict[str, Any]]:
    annotation = getattr(output, "speaker_diarization", output)
    turns: list[dict[str, Any]] = []
    if hasattr(annotation, "itertracks"):
        iterator = (
            (turn, speaker)
            for turn, _, speaker in annotation.itertracks(yield_label=True)
        )
    else:
        iterator = iter(annotation)
    for item in iterator:
        turn, speaker = item
        start, end = float(turn.start), float(turn.end)
        if end > start:
            turns.append({"start": start, "end": end, "speaker": str(speaker)})
    return sorted(turns, key=lambda row: (row["start"], row["end"], row["speaker"]))


def _single_speaker_segments(
    turns: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    boundaries = sorted({value for row in turns for value in (row["start"], row["end"])})
    raw: list[dict[str, Any]] = []
    overlap_seconds = 0.0
    for start, end in zip(boundaries, boundaries[1:]):
        if end <= start:
            continue
        midpoint = (start + end) / 2.0
        active = {
            row["speaker"]
            for row in turns
            if row["start"] <= midpoint < row["end"]
        }
        if len(active) > 1:
            overlap_seconds += end - start
        elif len(active) == 1:
            speaker = next(iter(active))
            if (
                raw
                and raw[-1]["speaker"] == speaker
                and abs(start - raw[-1]["end"]) <= 1e-6
            ):
                raw[-1]["end"] = end
            else:
                raw.append({"start": start, "end": end, "speaker": speaker})
    usable = [
        {**row, "duration": row["end"] - row["start"]}
        for row in raw
        if row["end"] - row["start"] >= _MIN_SEGMENT_SECONDS
    ]
    return usable, overlap_seconds


def _overlap_segments(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    boundaries = sorted({value for row in turns for value in (row["start"], row["end"])})
    regions: list[dict[str, Any]] = []
    for start, end in zip(boundaries, boundaries[1:]):
        midpoint = (start + end) / 2.0
        speakers = sorted({
            row["speaker"] for row in turns if row["start"] <= midpoint < row["end"]
        })
        if len(speakers) < 2:
            continue
        if (
            regions
            and regions[-1]["speakers"] == speakers
            and abs(regions[-1]["end"] - start) <= 1e-6
        ):
            regions[-1]["end"] = end
            regions[-1]["duration"] = end - regions[-1]["start"]
        else:
            regions.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "speakers": speakers,
            })
    return [row for row in regions if row["duration"] >= _MIN_SEGMENT_SECONDS]


def _get_pyannote(config: AudioConfig) -> Any:
    key = (config.pyannote_model, _device_key(config.device))
    cached = _PYANNOTE_CACHE.get(key)
    if cached is not None:
        return cached
    with _PYANNOTE_MODEL_LOCK:
        cached = _PYANNOTE_CACHE.get(key)
        if cached is not None:
            return cached
        import torch
        from pyannote.audio import Pipeline

        pipeline = Pipeline.from_pretrained(config.pyannote_model)
        if pipeline is None:
            raise RuntimeError(f"could not load pyannote pipeline: {config.pyannote_model}")
        pipeline.to(torch.device(_torch_device_name(config.device)))
        _PYANNOTE_CACHE[key] = pipeline
        return pipeline


def _diarize_speakers(wav_path: str, config: AudioConfig) -> dict[str, Any]:
    source = Path(wav_path)
    cache_key = sha256(str(source).encode("utf-8")).hexdigest()[:24]
    cache_path = source.with_name(f".pyannote-{cache_key}.json")
    source_stat = source.stat()
    if cache_path.is_file():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if (
                cached.get("source_size") == source_stat.st_size
                and cached.get("model_dir") == config.pyannote_model
            ):
                cached["overlap_segments"] = _overlap_segments(cached.get("turns") or [])
                return cached
        except (OSError, ValueError, TypeError):
            pass

    pipeline = _get_pyannote(config)
    # Feed samples directly so diarization does not depend on TorchCodec.
    import soundfile as sf
    import torch

    with _PYANNOTE_INFER_LOCK:
        waveform, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
        waveform_tensor = torch.from_numpy(waveform.T.copy())
        output = pipeline({"waveform": waveform_tensor, "sample_rate": int(sample_rate)})
    turns = _annotation_turns(output)
    segments, overlap_seconds = _single_speaker_segments(turns)
    result = {
        "model": config.pyannote_model,
        "model_dir": config.pyannote_model,
        "source_size": source_stat.st_size,
        "turns": turns,
        "segments": segments,
        "overlap_segments": _overlap_segments(turns),
        "speakers": sorted({row["speaker"] for row in turns}),
        "overlap_seconds": round(overlap_seconds, 3),
        "min_segment_seconds": _MIN_SEGMENT_SECONDS,
        "audio_input_backend": "waveform",
    }
    temporary = cache_path.with_suffix(
        cache_path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(json.dumps(result, ensure_ascii=False) + "\n", encoding="utf-8")
        temporary.replace(cache_path)
    finally:
        temporary.unlink(missing_ok=True)
    return result


def _get_mossformer(config: AudioConfig) -> tuple[Any, Any, Namespace, Any]:
    key = (config.mossformer2_model, _device_key(config.device))
    cached = _MOSS_CACHE.get(key)
    if cached is not None:
        return cached
    with _MOSS_MODEL_LOCK:
        cached = _MOSS_CACHE.get(key)
        if cached is not None:
            return cached
        import torch
        from clearvoice.models.mossformer2_ss.mossformer2 import MossFormer2_SS_16K
        from clearvoice.utils.decode_batch import decode_one_audio_batch

        model_dir = Path(config.mossformer2_model)
        if not model_dir.is_dir():
            from huggingface_hub import snapshot_download

            model_dir = Path(snapshot_download(config.mossformer2_model))
        checkpoint_file = model_dir / "last_best_checkpoint.pt"
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"MossFormer2 checkpoint missing: {checkpoint_file}")
        args = Namespace(
            network="MossFormer2_SS_16K",
            sampling_rate=_SR,
            num_spks=2,
            one_time_decode_length=2.0,
            decode_window=2.0,
            encoder_kernel_size=16,
            encoder_embedding_dim=512,
            mossformer_sequence_dim=512,
            num_mossformer_layer=24,
        )
        model = MossFormer2_SS_16K(args).model
        try:
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_file, map_location="cpu")
        pretrained = checkpoint.get("model", checkpoint)
        current = model.state_dict()
        loaded = 0
        for name, value in current.items():
            candidates = (name, name.removeprefix("module."), "module." + name)
            source = next(
                (
                    pretrained[candidate]
                    for candidate in candidates
                    if candidate in pretrained and pretrained[candidate].shape == value.shape
                ),
                None,
            )
            if source is not None:
                current[name] = source
                loaded += 1
        if loaded == 0:
            raise RuntimeError("MossFormer2 checkpoint did not match the model architecture")
        model.load_state_dict(current)
        device = torch.device(_torch_device_name(config.device))
        model.to(device).eval()
        cached = (model, device, args, decode_one_audio_batch)
        _MOSS_CACHE[key] = cached
        return cached


def _separate_overlap(
    signal: Any,
    *,
    source_wav: str,
    start: float,
    end: float,
    config: AudioConfig,
) -> list[dict[str, Any]]:
    import numpy as np
    import soundfile as sf

    source = Path(source_wav)
    first = max(0, int(round(start * _SR)))
    last = min(len(signal), int(round(end * _SR)))
    if last <= first:
        raise ValueError(f"empty overlap interval: {start:.3f}-{end:.3f}s")
    prefix = source.with_name(
        f"{source.stem}.mossformer2_{round(start * 1000):08d}_{round(end * 1000):08d}"
    )
    paths = [Path(f"{prefix}_spk1.wav"), Path(f"{prefix}_spk2.wav")]
    if all(path.is_file() for path in paths):
        return [
            {
                "speaker": f"MOSSFORMER2_SPK{index}",
                "signal": sf.read(path, dtype="float32", always_2d=False)[0],
                "audio_path": str(path),
            }
            for index, path in enumerate(paths, 1)
        ]

    model, device, args, decode = _get_mossformer(config)
    mixture = np.asarray(signal[first:last], dtype=np.float32)[None, :]
    with _MOSS_INFER_LOCK:
        separated = np.asarray(decode(model, device, mixture, args), dtype=np.float32)
    if separated.ndim == 3:
        separated = separated[:, 0, :]
    if separated.shape[0] != 2:
        raise RuntimeError(f"unexpected MossFormer2 output shape: {separated.shape}")
    outputs = []
    for index, (path, audio) in enumerate(zip(paths, separated), 1):
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
        try:
            sf.write(temporary, audio, _SR, format="WAV")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        outputs.append({
            "speaker": f"MOSSFORMER2_SPK{index}",
            "signal": audio,
            "audio_path": str(path),
        })
    return outputs


def _model_classes():
    import torch
    import torch.nn as nn
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2Model,
        Wav2Vec2PreTrainedModel,
    )

    class Head(nn.Module):
        def __init__(self, config, outputs: int):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, outputs)

        def forward(self, features):
            value = self.dropout(features)
            value = torch.tanh(self.dense(value))
            return self.out_proj(self.dropout(value))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = Head(config, config.num_labels)
            self.post_init()

        def forward(self, input_values):
            states = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
            return self.classifier(states)

    class AgeGenderModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.age = Head(config, 1)
            self.gender = Head(config, 3)
            self.post_init()

        def forward(self, input_values):
            states = self.wav2vec2(input_values).last_hidden_state.mean(dim=1)
            return self.age(states), torch.softmax(self.gender(states), dim=1)

    return EmotionModel, AgeGenderModel


def _audeering_id(kind: str, config: AudioConfig) -> str:
    if kind == "emotion":
        return config.audeering_emotion_model
    return config.audeering_age_gender_model


def _get_audeering(kind: str, config: AudioConfig) -> tuple[Any, Any, str]:
    model_id = _audeering_id(kind, config)
    key = (kind, model_id, _device_key(config.device))
    cached = _AUDEERING_CACHE.get(key)
    if cached is not None:
        return cached
    with _AUDEERING_MODEL_LOCK:
        cached = _AUDEERING_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            from transformers import Wav2Vec2FeatureExtractor
        except ImportError:
            from transformers.models.wav2vec2.feature_extraction_wav2vec2 import (
                Wav2Vec2FeatureExtractor,
            )

        emotion_cls, speaker_cls = _model_classes()
        model_cls = emotion_cls if kind == "emotion" else speaker_cls
        # These checkpoints ship an unused CTC tokenizer; the feature extractor is enough.
        processor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
        device = _torch_device_name(config.device)
        model = model_cls.from_pretrained(model_id).to(device).eval()
        cached = (model, processor, device)
        _AUDEERING_CACHE[key] = cached
        return cached


def _forward(kind: str, segment: Any, config: AudioConfig) -> list[float]:
    import torch

    model, processor, device = _get_audeering(kind, config)
    values = processor(segment, sampling_rate=_SR, return_tensors="pt").input_values.to(device)
    with torch.inference_mode():
        output = model(values)
    if kind == "emotion":
        return output[0].detach().float().cpu().tolist()
    age, gender = output
    return [float(age[0, 0].detach().cpu()), *gender[0].detach().float().cpu().tolist()]


def _weighted_profile(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> dict[str, float]:
    weights = [max(0.0, float(row["duration"])) for row in rows]
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * len(rows)
        total = float(len(rows))
    return {
        key: round(
            sum(float(row[key]) * weight for row, weight in zip(rows, weights)) / total,
            4,
        )
        for key in keys
    }


def _speaker_groups(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["speaker"], []).append(row)
    return groups


def _load_signal(
    video_path: str,
    work_dir: str | Path,
    *,
    end_sec: float,
    start_sec: float,
) -> tuple[str, Any]:
    wav_path = extract_audio(
        video_path,
        work_dir,
        start_sec=start_sec,
        end_sec=end_sec,
        sample_rate=_SR,
    )
    import soundfile as sf

    signal, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
    if getattr(signal, "ndim", 1) > 1:
        signal = signal.mean(axis=1)
    if sample_rate != _SR:
        raise ToolError(f"audio_extract returned {sample_rate}Hz, expected {_SR}Hz")
    return wav_path, signal


def _analysis_sources(
    wav_path: str,
    signal: Any,
    config: AudioConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any], _SkillReport | None]:
    try:
        result = _diarize_speakers(wav_path, config)
    except Exception as exc:
        return [], {}, _SkillReport(
            False,
            "diarization_unavailable",
            "low",
            f"speaker diarization failed: {exc}",
            error=str(exc),
        )

    sources = []
    for turn in result["segments"]:
        start, end = float(turn["start"]), float(turn["end"])
        sources.append({
            "speaker": turn["speaker"],
            "start": start,
            "end": end,
            "signal": signal[int(start * _SR):max(int(end * _SR), int(start * _SR) + 1)],
            "source_type": "clean_diarized",
            "audio_path": wav_path,
        })

    separated_paths: list[str] = []
    separation_errors: list[str] = []
    for overlap_index, overlap in enumerate(result.get("overlap_segments") or [], 1):
        start, end = float(overlap["start"]), float(overlap["end"])
        try:
            outputs = _separate_overlap(
                signal,
                source_wav=wav_path,
                start=start,
                end=end,
                config=config,
            )
            for source_index, output in enumerate(outputs, 1):
                sources.append({
                    "speaker": f"OVERLAP_{overlap_index:02d}_SOURCE_{source_index}",
                    "start": start,
                    "end": end,
                    "signal": output["signal"],
                    "source_type": "mossformer2_separated",
                    "audio_path": output["audio_path"],
                })
                separated_paths.append(output["audio_path"])
        except Exception as exc:
            separation_errors.append(f"{start:.3f}-{end:.3f}s: {type(exc).__name__}: {exc}")

    metadata = {
        "diarization": result,
        "mossformer2_used": bool(separated_paths),
        "separated_audio_paths": separated_paths,
        "separation_errors": separation_errors,
    }
    if not sources:
        if separation_errors:
            message = "MossFormer2 overlap separation failed: " + "; ".join(separation_errors)
            return [], metadata, _SkillReport(
                False,
                "separation_failed",
                "low",
                message,
                evidence=metadata,
                error=message,
            )
        return [], metadata, _SkillReport(
            True,
            "insufficient_speech",
            "low",
            "speaker diarization found no usable clean or overlapping speech segment of at least 0.8s.",
            evidence=metadata,
        )
    return sources, metadata, None


def _voice_profile(
    video_path: str,
    work_dir: str | Path,
    config: AudioConfig,
    *,
    end_sec: float,
) -> _SkillReport:
    wav_path, signal = _load_signal(video_path, work_dir, end_sec=float(end_sec), start_sec=0.0)
    sources, source_metadata, error = _analysis_sources(wav_path, signal, config)
    if error is not None:
        return error

    rows = []
    try:
        with _AUDEERING_INFER_LOCK:
            for turn in sources:
                start, end = float(turn["start"]), float(turn["end"])
                arousal, dominance, valence = _forward("emotion", turn["signal"], config)
                age, female, male, child = _forward("speaker", turn["signal"], config)
                rows.append({
                    "speaker": turn["speaker"],
                    "start": start,
                    "end": end,
                    "duration": round(end - start, 3),
                    "source_type": turn["source_type"],
                    "arousal": round(arousal, 4),
                    "dominance": round(dominance, 4),
                    "valence": round(valence, 4),
                    "age_years": round(age * 100.0, 2),
                    "female": round(female, 4),
                    "male": round(male, 4),
                    "child": round(child, 4),
                })
    except Exception as exc:
        return _SkillReport(
            False, "error", "low", f"voice_profile failed: {exc}", error=str(exc),
        )
    if not rows:
        return _SkillReport(
            True,
            "insufficient_speech",
            "low",
            "voice_profile: insufficient voiced audio.",
            evidence={"segments": []},
        )

    parts = []
    for row in rows:
        parts.append(
            f"{row['speaker']}@{format_seconds(row['start'])}-{format_seconds(row['end'])}s:"
            f"arousal={row['arousal']:.2f},dominance={row['dominance']:.2f},"
            f"valence={row['valence']:.2f};estimated_age_years={row['age_years']:.0f},"
            f"female_probability={row['female']:.2f},male_probability={row['male']:.2f},"
            f"child_probability={row['child']:.2f}"
        )

    speaker_keys = (
        "arousal", "dominance", "valence", "age_years", "female", "male", "child",
    )
    speakers = {
        speaker: {
            "n_segments": len(group),
            "speech_seconds": round(sum(row["duration"] for row in group), 3),
            **_weighted_profile(group, speaker_keys),
        }
        for speaker, group in _speaker_groups(rows).items()
    }
    notices = []
    if source_metadata["mossformer2_used"]:
        notices.append(
            "MossFormer2 was used only on overlapping intervals; separated-source estimates "
            "may contain separation artifacts."
        )
    if source_metadata["separation_errors"]:
        notices.append("Some overlapping intervals could not be separated and were omitted.")
    summary = " ".join(notices + [
        "voice_profile (speaker-aware and descriptive; profile_complete means the analysis "
        "succeeded, not that the media is real or fake; emotion dimensions and category "
        "probabilities are on 0-1 scales): " + "; ".join(parts)
    ])
    return _SkillReport(
        True,
        "profile_complete",
        "info",
        summary,
        evidence={
            "segments": rows,
            "speakers": speakers,
            "overlap_seconds": source_metadata["diarization"]["overlap_seconds"],
            "mossformer2_used": source_metadata["mossformer2_used"],
            "separated_audio_paths": source_metadata["separated_audio_paths"],
            "separation_errors": source_metadata["separation_errors"],
            "n_segments": len(rows),
        },
    )


def _render_voice_profile(report: _SkillReport) -> str:
    lines = [
        "Skill: voice_profile | whole video | "
        f"Verdict: {report.verdict} ({report.confidence})"
    ]
    if report.summary:
        lines.append(report.summary)
    for key, value in (report.evidence or {}).items():
        if key == "reason" and isinstance(value, str) and value:
            lines.append(f"Reason: {value}")
        elif key == "transcript" and isinstance(value, str) and value:
            lines.append(f'Transcript: "{value}"')
        elif key == "issues" and isinstance(value, list) and value:
            lines.append(f"Issues: {'; '.join(str(item) for item in value)}")
        elif isinstance(value, (int, float)):
            # bool is a subclass of int, so flags such as mossformer2_used are included.
            lines.append(f"{key}: {value}")
    return "\n".join(lines)


def run_voice_profile(
    video_path: str,
    work_dir: str | Path,
    config: AudioConfig,
    *,
    end_sec: float,
) -> str:
    """Run voice_profile from the start of the media through ``end_sec`` (-1 keeps the full file)."""
    try:
        report = _voice_profile(video_path, work_dir, config, end_sec=end_sec)
    except ToolError:
        raise
    except Exception as exc:
        raise ToolError(str(exc)) from exc
    if report.error:
        raise ToolError(report.error)
    return _render_voice_profile(report)
