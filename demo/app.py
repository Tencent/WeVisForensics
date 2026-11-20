"""WeVisForensics Gradio demo: upload a video, pick a window of up to 16 s, get a verdict and the full trace."""
import ast
import base64
import concurrent.futures
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

HERE = Path(__file__).resolve().parent
BACKEND = os.environ.get('WVF_BACKEND', '').rstrip('/')
WORK_DIR = Path(os.environ.get('WVF_WORK_DIR', '/tmp/wevisforensics_space'))
MOCK = os.environ.get('WVF_MOCK', '')
EXAMPLES_DIR = Path(os.environ.get('WVF_EXAMPLES_DIR', HERE / 'examples'))
GITHUB = os.environ.get('WVF_GITHUB', 'https://github.com/Tencent/WeVisForensics')
PAPER = os.environ.get('WVF_PAPER', 'https://arxiv.org/abs/2611.21373')
PROJECT = os.environ.get('WVF_PROJECT', 'https://tencent.github.io/WeVisForensics')
MAX_SECONDS = 16.
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# The backend only accepts videos under WORK_DIR/uploads, so Gradio must cache uploads there.
(WORK_DIR / 'uploads').mkdir(parents=True, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = str(WORK_DIR / 'uploads')
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')
LOCAL = None
if not BACKEND and not MOCK and os.environ.get('WVF_LOCAL', '1') != '0':
    try:
        import wvf_local as LOCAL  # noqa: E402  (loads the models; must precede gradio on ZeroGPU)
    except Exception:
        import traceback
        traceback.print_exc()
import gradio as gr  # noqa: E402

TOOLS = {
    'visual_appearance': ('Appearance', 'Rendering, texture and lighting artifacts'),
    'physical_consistency': ('Physics', 'Motion, contact and object permanence'),
    'common_sense': ('Commonsense', 'Plausibility of the depicted event'),
    'ocr_consistency': ('OCR', 'Glyph structure of visible text'),
    'voice_profile': ('Voice profile', 'Speaker age, gender and affect from audio'),
    'asr_analysis': ('Speech', 'Transcript timing and delivery'),
}
# Inspector findings are evidence for the investigator, so they never use the red/green verdict colors.
FLAG = {'issue', 'text_suspicious', 'timing_anomaly'}
OK = {'no_issue', 'text_ok', 'timing_natural'}
VERDICT_TEXT = {'issue': 'anomaly found', 'no_issue': 'nothing found'}

CITATION = '''@misc{du2026wevisforensicsselfevolvingagentomni,
      title={WeVisForensics: Self-Evolving Agent for Omni AI-Generated Video Detection}, 
      author={Yipeng Du and Yuxi Li and Ying Zhang and Chong Sun and Chen Li and Jing Lyu and Ying Tai},
      year={2026},
      eprint={2611.21373},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2611.21373}, 
}'''

LOGO = '''<svg viewBox="0 0 48 48" width="46" height="46" aria-hidden="true">
<defs><linearGradient id="wvf-g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#2fd383"/><stop offset="1" stop-color="#07a456"/></linearGradient></defs>
<rect x="1" y="1" width="46" height="46" rx="13" fill="url(#wvf-g)"/>
<path d="M10 15 L15.5 32 L21 20 L26.5 32 L32 15" fill="none" stroke="#fff" stroke-width="3.4"
 stroke-linecap="round" stroke-linejoin="round"/>
<circle cx="33.5" cy="30.5" r="5.2" fill="none" stroke="#fff" stroke-width="2.6"/>
<path d="M37.4 34.4 L41 38" stroke="#fff" stroke-width="2.8" stroke-linecap="round"/></svg>'''

GITHUB_ICON = ('<svg viewBox="0 0 16 16" width="15" height="15" fill="currentColor" aria-hidden="true"><path d="M8 0C3.58 '
               '0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94'
               '-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87'
               '.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 '
               '1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 '
               '3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0'
               '-4.42-3.58-8-8-8z"/></svg>')

CSS = '''
.gradio-container {width: 100% !important; max-width: 1260px !important; margin: auto !important}
#wvf-hero {padding: 18px 4px 4px}
.wvf-brand {display: flex; align-items: center; gap: 14px}
.wvf-brand svg {flex: none; filter: drop-shadow(0 4px 10px rgba(7,164,86,.28))}
#wvf-hero h1 {font-size: 32px; font-weight: 780; letter-spacing: -0.025em; margin: 0; line-height: 1.1}
#wvf-hero .tag {font-size: 13.5px; font-weight: 600; color: #07a456; margin-top: 3px}
#wvf-hero .sub {font-size: 15px; line-height: 1.62; opacity: .82; margin: 12px 0 0; max-width: 820px}
.wvf-links {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; align-items: center}
.wvf-btn {display: inline-flex; align-items: center; gap: 7px; font-size: 13px; font-weight: 650;
  padding: 6px 13px; border-radius: 99px; text-decoration: none !important; color: var(--body-text-color) !important;
  border: 1px solid var(--border-color-primary); background: var(--block-background-fill); transition: all .15s}
.wvf-btn:hover {border-color: #07c160; color: #07a456 !important; transform: translateY(-1px)}
.wvf-btn.dark {background: #111827; color: #fff !important; border-color: #111827}
.wvf-btn.dark:hover {background: #07a456; border-color: #07a456; color: #fff !important}
.wvf-pill {display: inline-flex; align-items: center; gap: 7px; font-size: 12.5px; font-weight: 600;
  padding: 6px 12px; border-radius: 99px; margin-left: 6px; opacity: .85}
.wvf-dot {width: 8px; height: 8px; border-radius: 50%; background: #9ca3af}
.wvf-dot.on {background: #10b981; box-shadow: 0 0 0 3px rgba(16,185,129,.18)}
.wvf-dot.wait {background: #f59e0b; animation: wvf-blink 1.6s ease-in-out infinite}
@keyframes wvf-blink {50% {opacity: .35}}
.wvf-panel {border: 1px solid var(--border-color-primary); border-radius: 16px; padding: 22px 24px;
  background: var(--block-background-fill); color: var(--body-text-color)}
.wvf-empty {text-align: center; padding: 46px 24px; opacity: .7; font-size: 15px; line-height: 1.7}
.wvf-verdict {display: flex; align-items: center; gap: 18px; flex-wrap: wrap}
.wvf-label {font-size: 30px; font-weight: 800; letter-spacing: -0.01em; line-height: 1.1}
.wvf-label.fake {color: #dc2626} .wvf-label.real {color: #059669} .wvf-label.unknown {opacity: .7}
.wvf-kicker {font-size: 12px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; opacity: .6}
.wvf-kicker .src {text-transform: none; letter-spacing: 0; font-weight: 600; margin-left: 8px; padding: 2px 8px;
  border-radius: 6px; background: rgba(7,193,96,.12); color: #07a456; opacity: 1}
.wvf-meta {font-size: 13.5px; opacity: .75; margin-top: 6px}
.wvf-chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px}
.wvf-chip {display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 600;
  padding: 4px 10px; border-radius: 8px; border: 1px solid transparent}
.wvf-chip.flag {background: rgba(217,119,6,.12); color: #b45309; border-color: rgba(217,119,6,.32)}
.dark .wvf-chip.flag {background: rgba(245,158,11,.14); color: #fbbf24; border-color: rgba(245,158,11,.35)}
.wvf-chip.ok {background: rgba(107,114,128,.12); color: var(--body-text-color); border-color: var(--border-color-primary)}
.wvf-chip.info {background: transparent; color: var(--body-text-color); border: 1px dashed var(--border-color-primary);
  opacity: .8}
.wvf-ex-head {display: flex; align-items: baseline; justify-content: space-between; gap: 12px; flex-wrap: wrap;
  margin: 18px 0 12px}
.wvf-ex-head h2 {font-size: 22px; margin: 0; letter-spacing: -.02em}
.wvf-ex-head p {font-size: 12.5px; opacity: .62; margin: 0}
.wvf-ex-grid {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px}
.wvf-ex {position: relative; aspect-ratio: 16/10; padding: 0; border: 2px solid transparent; border-radius: 12px;
  overflow: hidden; background: #0b1410; cursor: pointer; transition: border-color .15s, transform .15s}
.wvf-ex:hover {border-color: #6bcb99; transform: translateY(-2px)}
.wvf-ex.sel {border-color: #07c160; outline: 3px solid rgba(7,193,96,.18); outline-offset: 2px}
.wvf-ex video {display: block; width: 100%; height: 100%; object-fit: contain; pointer-events: none}
.wvf-bridge {display: none !important}
@media (max-width: 900px) {.wvf-ex-grid {grid-template-columns: repeat(3, minmax(0, 1fr))}}
@media (max-width: 560px) {.wvf-ex-grid {grid-template-columns: repeat(2, minmax(0, 1fr))}}
.wvf-steps {position: relative}
.wvf-step {position: relative; border: 1px solid var(--border-color-primary); border-left-width: 4px;
  border-radius: 12px; padding: 16px 18px; margin: 10px 0 0; background: var(--block-background-fill);
  color: var(--body-text-color); animation: wvf-in .35s ease both}
@keyframes wvf-in {from {opacity: 0; transform: translateY(6px)}}
.wvf-step.flag {border-left-color: #f59e0b} .wvf-step.ok {border-left-color: #9ca3af}
.wvf-step.info {border-left-color: rgba(156,163,175,.35)} .wvf-step.final {border-left-color: #07c160}
.wvf-head {display: flex; align-items: baseline; gap: 10px; flex-wrap: wrap}
.wvf-num {font-size: 12px; font-weight: 700; opacity: .55; font-variant-numeric: tabular-nums}
.wvf-tool {font-size: 16px; font-weight: 700}
.wvf-scope {font-size: 12px; font-weight: 600; padding: 2px 8px; border-radius: 6px;
  background: rgba(107,114,128,.12); font-variant-numeric: tabular-nums}
.wvf-desc {font-size: 12.5px; opacity: .6}
.wvf-step details {margin-top: 10px}
.wvf-step summary {cursor: pointer; font-size: 12.5px; font-weight: 700; letter-spacing: .04em;
  text-transform: uppercase; opacity: .65; list-style: none}
.wvf-step summary::-webkit-details-marker {display: none}
.wvf-step summary::before {content: '▸ '} .wvf-step details[open] summary::before {content: '▾ '}
.wvf-text {font-size: 14px; line-height: 1.68; margin: 8px 0 0; white-space: pre-wrap; overflow-wrap: anywhere}
.wvf-finding {margin-top: 12px; padding: 12px 14px; border-radius: 10px; background: rgba(107,114,128,.07)}
.wvf-finding .wvf-text {margin-top: 6px}
.wvf-finding ul {margin: 8px 0 0 18px; padding: 0; font-size: 14px; line-height: 1.6}
.wvf-table {width: 100%; border-collapse: collapse; margin-top: 8px; font-size: 13px; font-variant-numeric: tabular-nums}
.wvf-table th, .wvf-table td {text-align: left; padding: 5px 8px; border-bottom: 1px solid var(--border-color-primary)}
.wvf-table th {font-weight: 700; opacity: .7}
.wvf-note {font-size: 12.5px; opacity: .62; margin-top: 8px}
.wvf-win {border: 1px solid var(--border-color-primary); border-radius: 12px; padding: 12px 14px 10px;
  background: var(--block-background-fill)}
.wvf-win-head {display: flex; justify-content: space-between; font-size: 13px; font-weight: 650}
.wvf-win-val {font-variant-numeric: tabular-nums; color: #07a456}
.wvf-rng {position: relative; height: 34px; margin: 10px 9px 0; touch-action: none; user-select: none}
.wvf-rng::before {content: ""; position: absolute; left: 0; right: 0; top: 12px; height: 10px; border-radius: 5px;
  background: rgba(107,114,128,.16)}
.wvf-rng-sel {position: absolute; top: 12px; height: 10px; border-radius: 5px; cursor: grab;
  background: linear-gradient(90deg, #2fd383, #07a456); box-shadow: 0 0 0 3px rgba(7,193,96,.16)}
.wvf-h {position: absolute; top: 5px; width: 18px; height: 24px; margin-left: -9px; border-radius: 7px; cursor: ew-resize;
  background: #fff; border: 2px solid #07a456; box-shadow: 0 2px 6px rgba(0,0,0,.18)}
.wvf-h::after {content: ""; position: absolute; left: 6px; top: 6px; width: 2px; height: 8px;
  border-left: 1.5px solid #07a456; border-right: 1.5px solid #07a456}
.wvf-win-axis {display: flex; justify-content: space-between; font-size: 11px; opacity: .55; margin: 2px 0 0;
  font-variant-numeric: tabular-nums}
.wvf-win-tip {font-size: 11.5px; opacity: .6; margin-top: 4px}
.wvf-film {position: relative; overflow: hidden; border-radius: 4px;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 7%, #000 93%, transparent);
  mask-image: linear-gradient(90deg, transparent, #000 7%, #000 93%, transparent)}
.wvf-film-track {position: relative; display: flex; width: max-content; padding: 16px 0; background: #fff;
  animation: wvf-roll 14s linear infinite}
.wvf-film-track::before, .wvf-film-track::after {content: ""; position: absolute; left: 0; right: 0; height: 14px;
  background: repeating-linear-gradient(90deg, #fff 0 10px, #000 10px 16px) 3px 50% / 100% 7px no-repeat, #000}
.wvf-film-track::before {top: 0} .wvf-film-track::after {bottom: 0}
.wvf-film-track img {display: block; width: 126px; height: 80px; margin-right: 2px; object-fit: cover}
@keyframes wvf-roll {to {transform: translateX(-50%)}}
.wvf-film-gate {position: absolute; top: 12px; bottom: 12px; left: 50%; width: 132px; margin-left: -66px;
  border: 2.5px solid #2fd383; border-radius: 3px; pointer-events: none;
  box-shadow: 0 0 0 2000px rgba(0,0,0,.28), 0 0 14px rgba(47,211,131,.75), inset 0 0 10px rgba(47,211,131,.45);
  animation: wvf-gate 1.6s ease-in-out infinite}
@keyframes wvf-gate {50% {border-color: #9af0c4; box-shadow: 0 0 0 2000px rgba(0,0,0,.28), 0 0 22px rgba(47,211,131,.95),
  inset 0 0 14px rgba(47,211,131,.6)}}
.wvf-stages {list-style: none; margin: 16px 0 0; padding: 0}
.wvf-stages li {position: relative; display: flex; align-items: center; gap: 12px; padding: 7px 0 7px 0;
  font-size: 14px; opacity: .45}
.wvf-stages li.done {opacity: .8} .wvf-stages li.now {opacity: 1; font-weight: 650}
.wvf-stages li .ic {flex: none; width: 20px; height: 20px; border-radius: 50%; display: grid; place-items: center;
  font-size: 11px; font-weight: 800; border: 2px solid var(--border-color-primary); color: transparent}
.wvf-stages li.done .ic {background: #07c160; border-color: #07c160; color: #fff}
.wvf-stages li.now .ic {border-color: #07c160; position: relative}
.wvf-stages li.now .ic::after {content: ""; position: absolute; inset: 3px; border-radius: 50%; background: #07c160;
  animation: wvf-pulse 1.3s ease-in-out infinite}
@keyframes wvf-pulse {0%,100% {transform: scale(.55); opacity: .6} 50% {transform: scale(1); opacity: 1}}
.wvf-stages li .t {margin-left: auto; font-size: 12px; opacity: .6; font-variant-numeric: tabular-nums; font-weight: 500}
.wvf-cite pre {font-size: 12.5px !important}
#wvf-run {font-size: 16px; font-weight: 700}
footer {display: none !important}
'''

HEAD = '''<script>
const wvfVideo = () => document.querySelector('#wvf-video video');
// Keep the preview player inside the selected analysis window.
setInterval(() => {
  const v = wvfVideo(), w = document.querySelector('.wvf-win');
  if (!v || !w || v.paused || w.dataset.drag) return;
  const a = +w.dataset.s, b = +w.dataset.e;
  if (v.currentTime < a - 0.3 || v.currentTime > b) v.currentTime = a;
}, 200);
// autoplay is not reliable for videos injected through gr.HTML.
setInterval(() => document.querySelectorAll('.wvf-ex video').forEach(v => v.paused && v.play().catch(() => {})), 1000);
function wvfPaint(w) {
  const d = +w.dataset.d, s = +w.dataset.s, e = +w.dataset.e;
  w.querySelector('.wvf-rng-sel').style.left = (s / d * 100) + '%';
  w.querySelector('.wvf-rng-sel').style.width = ((e - s) / d * 100) + '%';
  w.querySelector('[data-h=s]').style.left = (s / d * 100) + '%';
  w.querySelector('[data-h=e]').style.left = (e / d * 100) + '%';
  w.querySelector('.wvf-win-val').textContent = `${s.toFixed(1)} – ${e.toFixed(1)} s · ${(e - s).toFixed(1)} s`;
}
document.addEventListener('pointerdown', ev => {
  const w = ev.target.closest('.wvf-win');
  if (!w || !ev.target.closest('.wvf-rng')) return;
  ev.preventDefault();
  const rng = w.querySelector('.wvf-rng'), d = +w.dataset.d, max = +w.dataset.max, min = 1;
  const at = x => { const r = rng.getBoundingClientRect(); return Math.min(Math.max((x - r.left) / r.width, 0), 1) * d; };
  const h = ev.target.closest('.wvf-h')?.dataset.h;
  const mode = h || (ev.target.closest('.wvf-rng-sel') ? 'move' : 'jump');
  const s0 = +w.dataset.s, e0 = +w.dataset.e, t0 = at(ev.clientX);
  w.dataset.drag = '1';
  const move = e => {
    const t = at(e.clientX);
    let s = s0, en = e0;
    if (mode === 's') { s = Math.min(t, en - min); if (en - s > max) en = s + max; }
    else if (mode === 'e') { en = Math.max(t, s + min); if (en - s > max) s = en - max; }
    else { const len = e0 - s0; s = mode === 'move' ? s0 + t - t0 : t - len / 2;
           s = Math.min(Math.max(s, 0), d - len); en = s + len; }
    s = Math.max(0, Math.round(s * 10) / 10); en = Math.min(d, Math.round(en * 10) / 10);
    w.dataset.s = s; w.dataset.e = en; wvfPaint(w);
    const v = wvfVideo(); if (v) { v.pause(); v.currentTime = mode === 'e' ? en : s; }
  };
  const up = () => {
    document.removeEventListener('pointermove', move);
    document.removeEventListener('pointerup', up);
    delete w.dataset.drag;
    const v = wvfVideo(); if (v) v.currentTime = +w.dataset.s;
    const f = document.querySelector('#wvf-range textarea, #wvf-range input');
    if (f) { f.value = `${w.dataset.s},${w.dataset.e}`; f.dispatchEvent(new Event('input', {bubbles: true})); }
  };
  if (mode === 'jump') move(ev);
  document.addEventListener('pointermove', move);
  document.addEventListener('pointerup', up);
});
document.addEventListener('click', e => {
  const card = e.target.closest('.wvf-ex');
  if (!card) return;
  document.querySelectorAll('.wvf-ex').forEach(x => x.classList.toggle('sel', x === card));
  const field = document.querySelector('#wvf-key textarea, #wvf-key input');
  if (!field) return;
  field.value = card.dataset.id;
  field.dispatchEvent(new Event('input', {bubbles: true}));
  setTimeout(() => {
    document.querySelector('#wvf-load')?.click();
    document.querySelector('#wvf-video')?.scrollIntoView({behavior: 'smooth', block: 'start'});
  }, 80);
});
</script>'''

def esc(value):
    return html.escape(str(value))


def fmt(t):
    return f'{t:.1f}'.rstrip('0').rstrip('.') if t == int(t) else f'{t:.1f}'


def load_cases():
    path = EXAMPLES_DIR / 'cases.json'
    if not path.is_file():
        return {}
    return {c['id']: c for c in json.loads(path.read_text(encoding='utf-8'))
            if (EXAMPLES_DIR / c['video']).is_file()}


CASES = load_cases()


def backend_health():
    if MOCK or LOCAL:
        return dict(ready=True, window=True, progress=True)
    if not BACKEND:
        return None
    state_file = WORK_DIR / 'STATUS.json'
    if state_file.exists() and json.loads(state_file.read_text()).get('state') != 'READY':
        return dict(ready=False)
    try:
        with OPENER.open(f'{BACKEND}/health', timeout=3) as response:
            return json.load(response)
    except (OSError, ValueError):
        return dict(ready=False)


def status_pill():
    health = backend_health()
    if MOCK:
        dot, text = 'on', 'Preview mode · replaying a recorded trace'
    elif health is None:
        dot, text = '', 'Live analysis is offline · browse the recorded examples below'
    elif health.get('ready'):
        dot, text = 'on', 'Model ready'
    else:
        dot, text = 'wait', 'Model is loading · this can take a few minutes'
    return f'<span class="wvf-pill"><span class="wvf-dot {dot}"></span>{esc(text)}</span>'


def hero_html():
    links = [f'<a class="wvf-btn dark" href="{esc(GITHUB)}" target="_blank" rel="noopener">{GITHUB_ICON}GitHub</a>']
    if PAPER:
        links.append(f'<a class="wvf-btn" href="{esc(PAPER)}" target="_blank" rel="noopener">📄 Paper</a>')
    if PROJECT:
        links.append(f'<a class="wvf-btn" href="{esc(PROJECT)}" target="_blank" rel="noopener">🌐 Project page</a>')
    links.append('<a class="wvf-btn" href="#wvf-citation">❝ Citation</a>')
    links.append(status_pill())
    return (f'<div id="wvf-hero"><div class="wvf-brand">{LOGO}<div><h1>WeVisForensics</h1>'
            '<div class="tag">Self-Evolving Agent for Omni AI-Generated Video Detection</div></div></div>'
            '<p class="sub">An investigator agent plans each inspection, calls specialist inspectors for appearance, '
            'physics, commonsense, on-screen text and audio on the time span it chooses, and reasons over their '
            'evidence before giving a verdict.</p>'
            f'<div class="wvf-links">{"".join(links)}</div></div>')


def kind_of(verdict):
    if verdict in FLAG:
        return 'flag'
    if verdict in OK:
        return 'ok'
    return 'info'


def verdict_text(verdict):
    return VERDICT_TEXT.get(verdict, verdict.replace('_', ' '))


def chip(kind, text):
    return f'<span class="wvf-chip {kind}">{esc(text)}</span>'


def parse_observation(text):
    """Split an inspector reply into verdict, confidence, coverage, issues, reason and leftovers."""
    lines = text.strip().splitlines()
    head = re.match(r'Skill:\s*(\S+)\s*\|\s*(.+?)\s*\|\s*Verdict:\s*(.+?)\s*\((\w+)\)', lines[0]) if lines else None
    if not head:
        return dict(verdict=None, confidence=None, body=text.strip())
    body = '\n'.join(lines[1:]).strip()
    verdict = head.group(3).strip().replace(' ', '_')
    out = dict(verdict=verdict, confidence=head.group(4), body=body)
    cover = re.search(r'over (\d+) (frames|localized regions)', body)
    if cover:
        out['coverage'] = f'{cover.group(1)} {cover.group(2)}'
    issues = re.search(r'issues=(\[.*?\])\.\s', body, re.S)
    if issues:
        try:
            out['issues'] = [str(x) for x in ast.literal_eval(issues.group(1))]
        except (ValueError, SyntaxError):
            pass
    reason = re.search(r'reason:\s*(.*)', body, re.S)
    if reason:
        out['reason'] = reason.group(1).strip()
    return out


def voice_table(body):
    rows = []
    for seg in re.split(r';\s*(?=SPEAKER_)', body.split('\n')[0].split(':', 1)[-1]):
        m = re.match(r'\s*(SPEAKER_\d+)@([\d.]+)-([\d.]+)s:(.*)', seg)
        if not m:
            continue
        kv = dict(re.findall(r'(\w+)=([\d.]+)', m.group(4)))
        rows.append((m.group(1).replace('SPEAKER_', 'Speaker '), f'{m.group(2)}–{m.group(3)} s',
                     kv.get('estimated_age_years', '–'), kv.get('female_probability', '–'),
                     kv.get('male_probability', '–')))
    if not rows:
        return ''
    head = ''.join(f'<th>{h}</th>' for h in ('Speaker', 'Segment', 'Est. age', 'P(female)', 'P(male)'))
    body_rows = ''.join('<tr>' + ''.join(f'<td>{esc(c)}</td>' for c in r) + '</tr>' for r in rows)
    return f'<table class="wvf-table"><tr>{head}</tr>{body_rows}</table>'


def from_backend(result):
    """Normalize a live backend result into the record used by the renderer."""
    trace = result['trajectory']
    steps = []
    for i, step in enumerate(trace.get('steps', []), 1):
        parsed = step.get('parsed') or {}
        action = parsed.get('action') or {}
        reasoning = parsed.get('reasoning') or step.get('raw') or ''
        env = step.get('environment') or {}
        env_text = env.get('text', '') if isinstance(env, dict) else str(env)
        skill = action.get('skill')
        if action.get('type') == 'answer' or (not skill and i == len(trace.get('steps', []))):
            steps.append(dict(final=True, reasoning=reasoning))
            continue
        if 'start_sec' in action:
            span = (float(action['start_sec']), float(action.get('end_sec', 0)))
        elif 'at_sec' in action:
            span = (float(action['at_sec']),)
        else:
            span = None
        obs = parse_observation(env_text) if env_text else dict(verdict=None, body='')
        steps.append(dict(skill=skill, span=span, reasoning=reasoning, obs=obs, raw=env_text))
    start = float(result.get('window_start', 0.))
    return dict(answer=trace.get('answer') if trace.get('status') == 'answered' else None,
                start=start, analyzed=float(result['analyzed_seconds']), original=float(result['original_seconds']),
                elapsed=result.get('elapsed_seconds'), steps=steps, source='live')


def from_case(case):
    steps = []
    for step in case['steps']:
        if not step.get('skill'):
            steps.append(dict(final=True, reasoning=step['reasoning']['en']))
            continue
        w = step.get('window') or {}
        span = (w['at'],) if w.get('type') == 'point' else (w['start'], w['end']) if 'start' in w else None
        rep = step['report']
        obs = dict(verdict=rep['verdict'], confidence=rep['confidence'], body=rep['en']['text'],
                   reason=rep['en']['text'], issues=rep['en'].get('issues') or [])
        steps.append(dict(skill=step['skill'], span=span, reasoning=step['reasoning']['en'], obs=obs))
    return dict(answer=case['answer'], start=0., analyzed=float(case['duration']), original=float(case['duration']),
                elapsed=None, steps=steps, source='example')


def scope_text(span):
    if not span:
        return 'whole clip'
    if len(span) == 1:
        return f'@ {span[0]:.1f} s'
    return f'{span[0]:.1f}–{span[1]:.1f} s'


def finding_html(skill, obs):
    if obs.get('verdict') is None:
        return f'<div class="wvf-finding"><div class="wvf-text">{esc(obs["body"])}</div></div>'
    kind = kind_of(obs['verdict'])
    parts = [chip(kind, f"{verdict_text(obs['verdict'])} · {obs['confidence']}")]
    if obs.get('coverage'):
        parts.append(f'<span class="wvf-desc">over {esc(obs["coverage"])}</span>')
    out = f'<div class="wvf-finding"><div class="wvf-head">{" ".join(parts)}</div>'
    table = voice_table(obs['body']) if skill == 'voice_profile' else ''
    if table:
        out += table + ('<div class="wvf-note">A complete profile means the analysis succeeded; the investigator '
                        'compares the estimated speaker with who is visible.</div>')
    else:
        if obs.get('issues'):
            out += '<ul>' + ''.join(f'<li>{esc(i)}</li>' for i in obs['issues']) + '</ul>'
        out += f'<div class="wvf-text">{esc(obs.get("reason") or obs["body"])}</div>'
    return out + '</div>'


def render(record):
    answer = record['answer']
    kind = answer if answer in ('real', 'fake') else 'unknown'
    label = {'real': 'Real', 'fake': 'AI-generated', 'unknown': 'No verdict'}[kind]
    blocks, chips = [], []
    for i, step in enumerate(record['steps'], 1):
        if step.get('final'):
            blocks.append(f'<section class="wvf-step final" style="animation-delay:{i * .06:.2f}s">'
                          f'<div class="wvf-head"><span class="wvf-num">{i:02d}</span>'
                          f'<span class="wvf-tool">Verdict · {esc(label)}</span></div>'
                          f'<div class="wvf-text">{esc(step["reasoning"])}</div></section>')
            continue
        skill, obs = step['skill'], step['obs']
        name, desc = TOOLS.get(skill, (skill or 'Model output', ''))
        step_kind = kind_of(obs['verdict']) if obs.get('verdict') else 'info'
        if obs.get('verdict'):
            chips.append(chip(step_kind, f"{name} · {verdict_text(obs['verdict'])}"))
        block = (f'<section class="wvf-step {step_kind}" style="animation-delay:{i * .06:.2f}s">'
                 f'<div class="wvf-head"><span class="wvf-num">{i:02d}</span>'
                 f'<span class="wvf-tool">{esc(name)}</span><span class="wvf-scope">{esc(scope_text(step["span"]))}</span>'
                 f'<span class="wvf-desc">{esc(desc)}</span></div>'
                 f'<details open><summary>Investigator</summary><div class="wvf-text">{esc(step["reasoning"])}</div></details>')
        if obs.get('verdict') or obs.get('body'):
            block += finding_html(skill, obs)
        blocks.append(block + '</section>')

    start, analyzed, original = record['start'], record['analyzed'], record['original']
    if original <= analyzed + 0.05:
        scope = f'Analyzed the full {analyzed:.1f} s clip'
    else:
        scope = f'Analyzed {fmt(start)}–{fmt(start + analyzed)} s of a {original:.0f} s video'
    n = len(chips)
    meta = f'{scope} · {n} inspection{"s" if n != 1 else ""}'
    if record.get('elapsed'):
        meta += f' · {record["elapsed"]:.1f} s'
    src = '<span class="src">Recorded example</span>' if record['source'] == 'example' else ''
    verdict = (f'<div class="wvf-panel"><div class="wvf-kicker">Verdict{src}</div>'
               f'<div class="wvf-verdict"><div class="wvf-label {kind}">{label}</div></div>'
               f'<div class="wvf-meta">{esc(meta)}</div><div class="wvf-chips">{"".join(chips)}</div></div>')
    return verdict + '<div class="wvf-steps">' + ''.join(blocks) + '</div>'


def clean_trace(record):
    """Public trace for download: no token ids, server paths or internal model names."""
    steps = []
    for i, step in enumerate(record['steps'], 1):
        if step.get('final'):
            steps.append(dict(step=i, action='answer', verdict=record['answer'], reasoning=step['reasoning']))
            continue
        span = step['span']
        scope = None if not span else dict(at_sec=span[0]) if len(span) == 1 else dict(start_sec=span[0], end_sec=span[1])
        obs = step['obs']
        steps.append(dict(step=i, action='run_skill', skill=step['skill'], scope=scope, reasoning=step['reasoning'],
                          report=dict(verdict=obs.get('verdict'), confidence=obs.get('confidence'),
                                      issues=obs.get('issues') or [], text=obs.get('reason') or obs.get('body'))))
    return dict(verdict=record['answer'],
                window=dict(start_sec=round(record['start'], 2), end_sec=round(record['start'] + record['analyzed'], 2),
                            video_seconds=round(record['original'], 2)),
                elapsed_seconds=record.get('elapsed'), steps=steps)


EMPTY_VERDICT = ('<div class="wvf-panel wvf-empty">Upload a video, choose up to <b>16 s</b> to analyze, '
                 'and press <b>Analyze</b>.<br>Or click an example below to see a recorded investigation.</div>')


def probe(path):
    try:
        out = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', str(path)],
                             capture_output=True, text=True, check=True, timeout=20)
        return float(json.loads(out.stdout)['format']['duration'])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError):
        return None


def filmstrip(path, start, length, n=8):
    imgs = []
    with tempfile.TemporaryDirectory() as tmp:
        for k in range(n):
            t = start + length * (k + .5) / n
            out = Path(tmp) / f'{k}.jpg'
            try:
                subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-ss', f'{t:.3f}', '-i', str(path),
                                '-frames:v', '1', '-vf', 'scale=-2:120', '-q:v', '6', str(out)],
                               capture_output=True, check=True, timeout=15)
                imgs.append(base64.b64encode(out.read_bytes()).decode())
            except (OSError, subprocess.SubprocessError):
                pass
    if not imgs:
        return ''
    # Frames are repeated so the -50% roll loops seamlessly; 128 px per frame keeps the 16 px hole pitch aligned.
    frames = ''.join(f'<img src="data:image/jpeg;base64,{b}">' for b in imgs * 2)
    return f'<div class="wvf-film"><div class="wvf-film-track">{frames}</div><div class="wvf-film-gate"></div></div>'


def window_html(duration):
    end = round(min(duration, MAX_SECONDS), 1)
    tip = f'Drag the handles to set the start and end (up to {MAX_SECONDS:.0f} s), or drag the green bar to move it.'
    return (f'<div class="wvf-win" data-d="{duration:.3f}" data-s="0" data-e="{end}" '
            f'data-max="{MAX_SECONDS:.0f}"><div class="wvf-win-head"><span>Analysis window</span>'
            f'<span class="wvf-win-val">0.0 – {end:.1f} s · {end:.1f} s</span></div>'
            f'<div class="wvf-rng"><div class="wvf-rng-sel" style="left:0;width:{end / duration * 100:.2f}%"></div>'
            f'<span class="wvf-h" data-h="s" style="left:0"></span>'
            f'<span class="wvf-h" data-h="e" style="left:{end / duration * 100:.2f}%"></span></div>'
            f'<div class="wvf-win-axis"><span>0 s</span><span>{duration:.1f} s</span></div>'
            f'<div class="wvf-win-tip">{tip}</div></div>')


def on_video(path):
    duration = probe(path) if path else None
    # Example clicks render their trace before this fires; only a new upload clears the result.
    reset = gr.update() if example_of(path) else EMPTY_VERDICT
    if not duration:
        return None, '', '', reset
    # Keep the HTML component mounted and update its value directly. Gradio
    # 6.0 can drop the initial HTML when value+visible are updated together.
    return (duration, window_html(duration),
            f'0,{min(duration, MAX_SECONDS):.1f}', reset)


def parse_range(duration, value):
    try:
        start, end = (float(x) for x in str(value).split(','))
    except ValueError:
        start, end = 0., MAX_SECONDS
    duration = duration or MAX_SECONDS
    start = min(max(start, 0.), max(duration - 1., 0.))
    return start, max(min(end, duration, start + MAX_SECONDS) - start, 1.)


def stages_html(job, elapsed):
    stage = job.get('stage', 'preparing')
    order = ['queued', 'preparing', 'overview', 'inspect', 'thinking']
    cur = order.index(stage) if stage in order else 1
    items = []
    if stage == 'queued':
        items.append(('now', 'Waiting for the GPU · another analysis is running', ''))
    items.append(('done' if cur > 1 else 'now' if cur == 1 else '', 'Cutting the analysis window', ''))
    items.append(('done' if cur > 2 else 'now' if cur == 2 else '', 'Investigator reads a 1 fps overview', ''))
    for k, name in enumerate(job.get('history', []), 1):
        state = 'now' if stage == 'inspect' and k == len(job['history']) else 'done'
        items.append((state, f'Inspection {k} · {TOOLS.get(name, (name,))[0]} inspector', ''))
    items.append(('now' if stage in ('verdict', 'thinking') else '', 'Investigator weighs the evidence', ''))
    lis = ''.join(f'<li class="{c}"><span class="ic">✓</span>{esc(t)}<span class="t">{esc(x)}</span></li>'
                  for c, t, x in items)
    return (f'<div class="wvf-panel"><div class="wvf-kicker">Investigating · {elapsed} s</div>'
            f'<ul class="wvf-stages">{lis}</ul></div>')


def mock_progress(elapsed, record):
    skills = [s['skill'] for s in record['steps'] if not s.get('final')]
    t = elapsed / 1.2
    if t < 1:
        return dict(stage='preparing')
    if t < 2:
        return dict(stage='overview')
    k = int(t - 2) // 2 + 1
    if k <= len(skills):
        return dict(stage='inspect', history=skills[:k])
    return dict(stage='thinking', history=skills)


def fetch_progress(job):
    try:
        with OPENER.open(f'{BACKEND}/progress?job={job}', timeout=2) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def call_backend(path, start, length, job):
    if MOCK:
        time.sleep(1.2 * 12)
        return json.loads(Path(MOCK).read_text())
    payload = dict(video_path=str(path), start_sec=start, length_sec=length, job_id=job)
    request = urllib.request.Request(f'{BACKEND}/infer', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with OPENER.open(request, timeout=1200) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors='replace')
        try:
            detail = json.loads(detail).get('detail', detail)
        except ValueError:
            pass
        raise RuntimeError(detail) from error


def run_local(path, start, length, job, started):
    """In-process analysis; the GPU generator runs in this request thread so ZeroGPU can bill the visitor."""
    result = None
    try:
        media = LOCAL.prepare(path, start, length, job)
        for item in LOCAL.investigate(media):
            if 'result' in item:
                result = item['result']
                continue
            yield gr.update(), stages_html(item, int(time.monotonic() - started)), gr.update()
    except gr.Error:
        raise
    except Exception as error:
        raise gr.Error(f'Analysis failed: {error}') from error
    if result is None:
        raise gr.Error('Analysis failed: the agent returned no result.')
    result['elapsed_seconds'] = time.monotonic() - started
    return result


def example_of(path):
    return CASES.get(Path(path).stem) if path else None


def show_example(path):
    case = example_of(path)
    if not case:
        return EMPTY_VERDICT, gr.update(visible=False)
    return render(from_case(case)), gr.update(value=None, visible=False)


def load_example(key):
    case = CASES.get(key)
    if not case:
        return gr.update(), gr.update(), gr.update(), gr.update()
    path = str(EXAMPLES_DIR / case['video'])
    return (path, '', *show_example(path))


def file_url(path):
    return '/gradio_api/file=' + urllib.parse.quote(str(path), safe='/')


def examples_html():
    cards = ''.join(f'<button type="button" class="wvf-ex" data-id="{esc(cid)}">'
                    f'<video src="{esc(file_url(EXAMPLES_DIR / c["video"]))}" '
                    f'poster="{esc(file_url(EXAMPLES_DIR / c["poster"]))}" muted loop autoplay playsinline '
                    f'preload="metadata" tabindex="-1"></video></button>' for cid, c in CASES.items())
    return f'<div class="wvf-ex-head"><h2>Examples</h2></div><div class="wvf-ex-grid">{cards}</div>'


def analyze(path, duration, window_range):
    if not path:
        raise gr.Error('Please upload a video or pick an example first.')
    health = backend_health()
    if health is None:
        if example_of(path):
            yield '', *show_example(path)
            return
        raise gr.Error('Live analysis is offline in this Space. Click an example to see a recorded investigation, '
                       f'or run the full pipeline locally from {GITHUB}.')
    if not health.get('ready'):
        raise gr.Error('The model is still loading. Please try again in a few minutes.')
    start, length = parse_range(duration, window_range)
    if start > 0.05 and not health.get('window'):
        raise gr.Error('This deployment only analyzes the first 16 s. Move the window back to 0 s.')
    job = uuid.uuid4().hex
    if not Path(path).resolve().is_relative_to((WORK_DIR / 'uploads').resolve()):
        copy = WORK_DIR / 'uploads' / job / Path(path).name
        copy.parent.mkdir(parents=True)
        shutil.copyfile(path, copy)
        path = str(copy)
    strip = filmstrip(path, start, length)
    mock_record = from_backend(json.loads(Path(MOCK).read_text())) if MOCK else None
    started = time.monotonic()
    yield strip, stages_html(dict(stage='preparing'), 0), gr.update(visible=False)
    job_state = dict(stage='preparing')
    if LOCAL:
        result = yield from run_local(path, start, length, job, started)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(call_backend, path, start, length, job)
            while not future.done():
                elapsed = time.monotonic() - started
                if MOCK:
                    job_state = mock_progress(elapsed, mock_record)
                elif health.get('progress'):
                    job_state = fetch_progress(job) or job_state
                yield gr.update(), stages_html(job_state, int(elapsed)), gr.update()
                time.sleep(1)
            try:
                result = future.result()
            except (OSError, ValueError, RuntimeError) as error:
                raise gr.Error(f'Analysis failed: {error}') from error
    result.setdefault('window_start', start)
    record = from_backend(result)
    out = WORK_DIR / 'requests' / job / 'wevisforensics_trace.json'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(clean_trace(record), ensure_ascii=False, indent=2))
    yield '', render(record), gr.update(value=str(out), visible=True)


GRADIO6 = int(gr.__version__.split('.')[0]) >= 6


def style():
    theme = gr.themes.Soft(primary_hue=gr.themes.colors.green, neutral_hue=gr.themes.colors.slate,
                           font=['Arial', 'ui-sans-serif', 'system-ui', 'sans-serif'],
                           font_mono=['Consolas', 'ui-monospace', 'monospace'])
    return dict(theme=theme, css=CSS, head=HEAD)


def build():
    # Gradio 6 takes theme/css/head in launch(); Gradio 5 only in Blocks().
    with gr.Blocks(title='WeVisForensics · AI-generated video forensics',
                   **({} if GRADIO6 else style())) as demo:
        hero = gr.HTML(hero_html())
        duration = gr.State(None)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=340):
                video = gr.Video(label='Video', sources=['upload'], height=360, elem_id='wvf-video')
                window = gr.HTML(value='')
                window_range = gr.Textbox(elem_id='wvf-range', elem_classes='wvf-bridge', show_label=False)
                with gr.Row():
                    clear = gr.Button('Clear', variant='secondary', scale=1)
                    run = gr.Button('Analyze', variant='primary', scale=3, elem_id='wvf-run')
                download = gr.File(label='Investigation trace (JSON)', interactive=False, visible=False)
            with gr.Column(scale=7, min_width=380):
                strip = gr.HTML(value='')
                verdict = gr.HTML(EMPTY_VERDICT)
        if CASES:
            gr.HTML(examples_html())
        key = gr.Textbox(elem_id='wvf-key', elem_classes='wvf-bridge', show_label=False)
        load = gr.Button('load', elem_id='wvf-load', elem_classes='wvf-bridge')
        with gr.Accordion('How it works', open=False):
            gr.Markdown(
                '1. The **investigator** watches a 1 fps overview of the selected window and decides what to inspect, '
                'where, and when.\n'
                '2. Each call goes to a specialist **inspector** (appearance, physics, commonsense, OCR, voice, speech) '
                'that returns a verdict, a confidence, and its visual or acoustic reasoning.\n'
                '3. After up to four inspections the investigator weighs the evidence and answers **real** or '
                '**AI-generated**.')
        gr.HTML(f'<h2 id="wvf-citation" style="margin:8px 0 0">Citation</h2>'
                f'<p style="opacity:.75;margin:6px 0 0">If you find WeVisForensics useful, please consider citing our '
                f'paper and starring the <a href="{esc(GITHUB)}" target="_blank" rel="noopener">GitHub repository</a>.</p>')
        gr.Code(CITATION, language=None, interactive=False, show_label=False, elem_classes='wvf-cite')

        video.change(on_video, video, [duration, window, window_range, verdict], queue=False)
        live = run.click(analyze, [video, duration, window_range], [strip, verdict, download],
                         concurrency_limit=int(os.environ.get('WVF_CONCURRENCY', 1)), show_progress='hidden')
        load.click(load_example, key, [video, strip, verdict, download], queue=False,
                   show_progress='hidden', cancels=[live])
        clear.click(lambda: (None, '', EMPTY_VERDICT, gr.update(value=None, visible=False)),
                    outputs=[video, strip, verdict, download], queue=False)
        demo.load(hero_html, outputs=hero, queue=False)
    return demo


if __name__ == '__main__':
    extra = dict(style(), footer_links=[]) if GRADIO6 else dict(show_api=False)
    build().queue(max_size=16, default_concurrency_limit=1).launch(
        server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'),
        server_port=int(os.environ.get('PORT', 7860)),
        allowed_paths=[str(WORK_DIR / 'uploads'), str(WORK_DIR / 'requests'), str(EXAMPLES_DIR)],
        show_error=True, **extra)
