"""In-process backend for the Space: the WeVisForensics agent on one GPU with transformers.

The main agent and the four visual inspectors share one working copy of Qwen3.5-4B whose weights
hold ``W + scale * B @ A`` for the active role (the same merge as scripts/merge_lora.py), so every
generation runs on plain linear layers; speech analysis uses an untouched copy of the base model.
The agent loop, prompts, frame sampling and parsing come unchanged from the WeVisForensics package.
"""
try:
    import spaces  # must be imported before torch on ZeroGPU
except ImportError:
    spaces = None
import asyncio
import contextvars
import copy
import inspect
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
# Inside a repository checkout, use its packages instead of the pip-installed copy.
if (HERE.parent / 'main_agent').is_dir():
    sys.path.insert(0, str(HERE.parent))

from core import media as core_media
from core.client import ChatResult
from core.skill_card import load_skill_card, with_card
from main_agent.continuation import initialize_system_prompt, initialize_turn_separator
from main_agent.engine import TEMPLATE_KWARGS, Generation, MainEngine, dedup_image_tokens
from main_agent.loop import AgentConfig, ForensicAgent
from main_agent.protocol import overview_timestamps

BASE = os.environ.get('WVF_BASE', 'Qwen/Qwen3.5-4B')
ADAPTERS = os.environ.get('WVF_ADAPTERS', 'tencent/WeVisForensics')
TEMPLATE = Path(os.environ.get('WVF_CHAT_TEMPLATE', HERE / 'configs' / 'chat_template.jinja'))
WORK_DIR = Path(os.environ.get('WVF_WORK_DIR', '/tmp/wevisforensics_space'))
OCR_DEVICE = os.environ.get('WVF_OCR_DEVICE', 'cpu')
GPU_SECONDS = int(os.environ.get('WVF_GPU_SECONDS', 120))
INSPECTORS = ('physical_consistency', 'visual_appearance', 'common_sense', 'ocr_consistency')
ROLES = ('main',) + INSPECTORS
MAX_SECONDS = 16.
PYAV_THREADS = int(os.environ.get('WVF_PYAV_THREADS', 4))


def _decode_pyav_threaded(video_path, timestamps):
    """core.media._decode_pyav with frame-threaded decoding."""
    import av

    groups = []
    for timestamp in sorted(set(timestamps)):
        if not groups or timestamp - groups[-1][-1] > core_media.SEQUENTIAL_GAP_SECONDS:
            groups.append([timestamp])
        else:
            groups[-1].append(timestamp)
    decoded = {}
    with av.open(video_path, mode='r') as container:
        stream = container.streams.video[0]
        stream.thread_count = PYAV_THREADS
        stream.thread_type = 'FRAME'
        for group in groups:
            decoded.update(core_media._decode_group(container, stream, group))
    return decoded


core_media._decode_pyav = _decode_pyav_threaded


def _adapter_root():
    if Path(ADAPTERS).is_dir():
        return Path(ADAPTERS)
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(ADAPTERS))


def _base_model():
    from transformers import AutoModelForImageTextToText
    return AutoModelForImageTextToText.from_pretrained(BASE, torch_dtype=torch.bfloat16, device_map='cpu')


class Runtime:
    """Loads the base model, one working copy and every role's LoRA factors once; serializes the GPU."""

    def __init__(self):
        from peft import PeftModel
        from peft.tuners.lora.layer import LoraLayer
        from transformers import AutoProcessor

        if os.environ.get('WVF_GPU_MEMORY_GB'):
            total = torch.cuda.get_device_properties(0).total_memory / 2 ** 30
            torch.cuda.set_per_process_memory_fraction(min(1., float(os.environ['WVF_GPU_MEMORY_GB']) / total))
        self.root = _adapter_root()
        self.template = TEMPLATE.read_text(encoding='utf-8')
        self.processor = AutoProcessor.from_pretrained(BASE, trust_remote_code=True)
        self.tokenizer = getattr(self.processor, 'tokenizer', self.processor)
        self.tokenizer.chat_template = self.template
        self.base = _base_model().eval()
        peft = PeftModel.from_pretrained(copy.deepcopy(self.base), str(self.root),
                                         adapter_name=ROLES[0], is_trainable=False)
        for role in ROLES[1:]:
            peft.load_adapter(str(self.root / role), adapter_name=role, is_trainable=False)
        factors = {}
        for name, module in peft.base_model.model.named_modules():
            if isinstance(module, LoraLayer):
                factors[name] = {role: (module.lora_A[role].weight.detach().clone(),
                                        module.lora_B[role].weight.detach().clone(), float(module.scaling[role]))
                                 for role in ROLES if role in module.lora_A}
        self.work = peft.unload().eval()
        del peft
        self.base.to('cuda')
        self.work.to('cuda')
        base_modules, work_modules = dict(self.base.named_modules()), dict(self.work.named_modules())
        self.layers = [(work_modules[name].weight.data, base_modules[name].weight.data,
                        {role: (a.to('cuda'), b.to('cuda'), s) for role, (a, b, s) in per_role.items()})
                       for name, per_role in factors.items()]
        self.active = None
        self.cards = load_skill_card(self.root / 'skill_card.json')
        self.image_token_id = int(self.processor.image_token_id)
        self.needs_type_ids = 'mm_token_type_ids' in inspect.signature(self.work.forward).parameters
        eos = set()
        for value in (self.work.generation_config.eos_token_id, self.tokenizer.eos_token_id):
            eos.update(value if isinstance(value, (list, tuple)) else [value] if value is not None else [])
        self.eos = eos
        self.lock = threading.Lock()

    def _activate(self, role):
        """Rebuilds each adapted layer from the read-only base so repeated switches never accumulate rounding."""
        if self.active == role:
            return
        with torch.no_grad():
            for work, base, per_role in self.layers:
                if role in per_role:
                    a, b, scale = per_role[role]
                    work.copy_(base + ((b.float() @ a.float()) * scale).to(base.dtype))
                else:
                    work.copy_(base)
        self.active = role

    def _generate(self, model, inputs, max_new_tokens):
        with torch.inference_mode():
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False, temperature=None,
                                 top_p=None, top_k=None, repetition_penalty=1.0, eos_token_id=sorted(self.eos))
        ids = out[0, inputs['input_ids'].shape[1]:].tolist()
        return ids, 'stop' if ids and ids[-1] in self.eos else 'length'

    def main_generate(self, prompt_ids, images, max_new_tokens):
        """Token-in token-out turn of the main agent on the full, image-pad-expanded history."""
        ids = torch.tensor([prompt_ids], dtype=torch.long)
        inputs = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        if images:
            pixels = self.processor.image_processor(images=list(images), return_tensors='pt')
            inputs.update(pixel_values=pixels['pixel_values'], image_grid_thw=pixels['image_grid_thw'])
            if self.needs_type_ids:
                inputs['mm_token_type_ids'] = (ids == self.image_token_id).int()
        inputs = {k: v.to('cuda') for k, v in inputs.items()}
        with self.lock:
            self._activate('main')
            return self._generate(self.work, inputs, max_new_tokens)

    def chat(self, adapter, messages, max_new_tokens):
        """Chat request rendered with the served chat template, as vLLM does for the inspectors."""
        images, converted = [], []
        for message in messages:
            content = message['content']
            if isinstance(content, str):
                converted.append(message)
                continue
            parts = []
            for part in content:
                if part['type'] == 'image':
                    with Image.open(part['path']) as image:
                        images.append(image.convert('RGB'))
                    parts.append({'type': 'image'})
                else:
                    parts.append({'type': 'text', 'text': part['text']})
            converted.append({'role': message['role'], 'content': parts})
        text = self.processor.apply_chat_template(converted, chat_template=self.template, tokenize=False,
                                                  add_generation_prompt=True, **TEMPLATE_KWARGS)
        inputs = self.processor(text=[text], images=images or None, return_tensors='pt').to('cuda')
        with self.lock:
            if adapter is None:
                ids, finish = self._generate(self.base, inputs, max_new_tokens)
            else:
                self._activate(adapter)
                ids, finish = self._generate(self.work, inputs, max_new_tokens)
        return ChatResult(text=self.tokenizer.decode(ids, skip_special_tokens=True),
                          prompt_tokens=int(inputs['input_ids'].shape[1]), completion_tokens=len(ids),
                          finish_reason=finish)


class LocalEndpoint:
    """Drop-in for core.client.ChatEndpoint. ``adapter=None`` is the plain base model (asr_analysis)."""

    def __init__(self, runtime, adapter, skill_card=None):
        self.runtime, self.adapter, self.skill_card = runtime, adapter, skill_card or []

    async def chat(self, messages, *, max_tokens):
        return await asyncio.to_thread(self.runtime.chat, self.adapter, with_card(messages, self.skill_card),
                                       max_tokens)


class LocalMainEngine(MainEngine):
    """MainEngine with the same encoding and limits; generation runs on the main-role working copy."""

    def __init__(self, runtime):
        super().__init__(BASE, chat_template=str(TEMPLATE))
        self.runtime = runtime

    def load(self):
        if self.processor is not None:
            return
        self.processor, self.tokenizer = self.runtime.processor, self.runtime.tokenizer
        self.system_prompt_ids = initialize_system_prompt(self.processor, **TEMPLATE_KWARGS)
        self.turn_separator = initialize_turn_separator(self.processor, **TEMPLATE_KWARGS)

    async def generate(self, prompt_ids, images, *, max_tokens):
        self.load()
        if len(prompt_ids) > self.prompt_length and images:
            raise RuntimeError(f'initial prompt has {len(prompt_ids)} tokens, above prompt_length={self.prompt_length}')
        headroom = self.max_model_len - len(dedup_image_tokens(list(prompt_ids), self.processor))
        # vLLM also rejects prompts whose image-expanded length exceeds max_model_len.
        if headroom < 1 or len(prompt_ids) >= self.max_model_len:
            raise RuntimeError(f'no context headroom: {len(prompt_ids)} tokens')
        ids, finish = await asyncio.to_thread(self.runtime.main_generate, list(prompt_ids), images,
                                              max(1, min(int(max_tokens), headroom)))
        return Generation(token_ids=ids, text=self.tokenizer.decode(ids, skip_special_tokens=True),
                          finish_reason=finish)


EVENTS = contextvars.ContextVar('wvf_events')


class DemoAgent(ForensicAgent):
    """Reports each inspection to the request's own queue; actions and results are untouched."""

    async def run_reporting(self, events, *args, **kwargs):
        EVENTS.set(events)
        return await self.run(*args, **kwargs)

    async def _execute(self, action, traj):
        events = EVENTS.get(None)
        if events is not None:
            events.put(('inspect', action.get('skill')))
        try:
            return await super()._execute(action, traj)
        finally:
            if events is not None:
                events.put(('thinking', None))


def build_agent(runtime):
    from sub_agents.audio import AudioConfig
    from sub_agents.ocr import PaddleOCRBackend
    tools_path = os.environ.get('WVF_TOOLS_CONFIG')
    tools = json.loads(Path(tools_path).read_text()) if tools_path else {}
    ocr_options = dict(device=OCR_DEVICE, lang='ch', ocr_version='PP-OCRv5')
    ocr_options.update(tools.get('ocr', {}))
    if ocr_options['device'].lower() == 'cpu':
        from wvf_cpu_ocr import CpuOCRBackend
        ocr = CpuOCRBackend(work_dir=WORK_DIR/'cpu_ocr', **ocr_options)
    else:
        ocr = PaddleOCRBackend(**ocr_options)

    endpoints = {role: LocalEndpoint(runtime, role, runtime.cards.get(role)) for role in INSPECTORS}
    endpoints['asr_analysis'] = LocalEndpoint(runtime, None, runtime.cards.get('asr_analysis'))
    return DemoAgent(
        endpoints, main=LocalMainEngine(runtime), backend='engine',
        ocr=ocr,
        audio_config=AudioConfig(**tools.get('audio', {})),
        config=AgentConfig(max_steps=5, turn_max_tokens=1024, response_budget=12288, specialist_max_tokens=500,
                           cache_root=str(WORK_DIR / 'cache')),
    )


def probe(path):
    result = subprocess.run(['ffprobe', '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(path)],
                            capture_output=True, text=True, check=True, timeout=30)
    info = json.loads(result.stdout)
    videos = [s for s in info['streams'] if s['codec_type'] == 'video']
    if not videos:
        raise ValueError('no decodable video stream')
    duration = float(videos[0].get('duration') or info['format']['duration'])
    if duration <= 0:
        raise ValueError('invalid video duration')
    return duration, any(s['codec_type'] == 'audio' for s in info['streams'])


def cut_window(source, work, start, length):
    """Frame-accurate lossless cut of at most 16 s; returns the clip and its analysis metadata."""
    original, _ = probe(source)
    start = min(max(float(start), 0.), max(original - 1., 0.))
    length = min(max(float(length), 1.), MAX_SECONDS, original - start)
    work.mkdir(parents=True, exist_ok=True)
    clip = work / 'window.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(source),
                    '-ss', f'{start:.3f}', '-t', f'{length:.3f}', '-map', '0:v:0', '-map', '0:a:0?',
                    '-c:v', 'libx264', '-crf', '0', '-preset', 'ultrafast', '-vsync', '0', '-c:a', 'flac', str(clip)],
                   capture_output=True, check=True, timeout=300)
    duration, has_audio = probe(clip)
    # Container timestamps can overshoot the cut by one frame; the analysis window stays capped.
    return dict(path=str(clip), duration=min(length, duration), has_audio=has_audio,
                original=original, start=start)


def to_backend_steps(traj):
    """Trajectory steps in the layout the Gradio renderer reads."""
    return [dict(raw=s['raw'], parsed=dict(action=s['action'], reasoning=s['reasoning']),
                 environment=dict(text=s.get('observation') or ''), errors=s['errors']) for s in traj.steps]


def prefetch(agent):
    """Download tool weights at startup so the first request does not spend GPU time on it."""
    from huggingface_hub import snapshot_download

    cfg = agent.audio_config
    for repo in (cfg.whisper_model, cfg.pyannote_model, cfg.mossformer2_model,
                 cfg.audeering_emotion_model, cfg.audeering_age_gender_model):
        if Path(repo).is_dir():
            continue
        try:
            snapshot_download(repo)
        except Exception as error:  # gated or offline; the tool reports its own error when called
            print(f'prefetch skipped {repo}: {error}', flush=True)
    agent.ocr._get_model()


RUNTIME = Runtime()
AGENT = build_agent(RUNTIME)
prefetch(AGENT)
def gpu_seconds(media):
    """ZeroGPU refuses requests above the visitor's remaining quota, so ask only for what the clip needs."""
    return min(GPU_SECONDS, int(45 + 3 * media['duration'] + (20 if media['has_audio'] else 0)))


gpu = spaces.GPU(duration=gpu_seconds) if spaces is not None else (lambda fn: fn)


def prepare(video_path, start, length, job):
    """CPU step before the GPU is requested; also fills the frame cache the agent reads its overview from."""
    media = cut_window(video_path, WORK_DIR / 'requests' / job, start, length)
    AGENT.frames.extract(media['path'], overview_timestamps(media['duration']),
                         max_pixels=AGENT.config.frame_max_pixels)
    return media


@gpu
def investigate(media):
    """Yields progress dicts, then a final dict with ``result``."""
    yield dict(stage='overview', history=[])
    events, box = queue.Queue(), {}

    def worker():
        try:
            box['traj'] = asyncio.run(AGENT.run_reporting(events, media['path'], duration=media['duration'],
                                                          has_audio=media['has_audio']))
        except BaseException as error:  # surfaced to the UI below
            box['error'] = error
        finally:
            events.put(None)

    threading.Thread(target=worker, daemon=True).start()
    history, state = [], dict(stage='overview', history=[])
    while True:
        try:
            item = events.get(timeout=1)
        except queue.Empty:
            yield state  # heartbeat so the elapsed timer keeps moving
            continue
        if item is None:
            break
        stage, skill = item
        if skill:
            history.append(skill)
        state = dict(stage=stage, history=list(history))
        yield state
    if 'error' in box:
        raise box['error']
    traj = box['traj']
    yield dict(stage='done', history=history, result=dict(
        analyzed_seconds=media['duration'], original_seconds=media['original'], window_start=media['start'],
        trajectory=dict(status=traj.status, answer=traj.answer, steps=to_backend_steps(traj))))
