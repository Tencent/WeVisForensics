"""CPU Paddle worker with oneDNN, isolated from PyTorch/CUDA native libraries."""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
import uuid


def ocr_threads():
    if os.environ.get('WVF_OCR_THREADS'):
        return max(1, int(os.environ['WVF_OCR_THREADS']))
    return max(1, min(8, len(os.sched_getaffinity(0))))


class CpuOCRBackend:
    def __init__(self, *, work_dir, **options):
        self.work = Path(work_dir)
        self.work.mkdir(parents=True, exist_ok=True)
        self.options = dict(options, device='cpu')
        self.lock = threading.Lock()
        self.process = None
        self.url = None
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        atexit.register(self.close)

    def close(self):
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.url = None

    def _start(self):
        if self.process is not None and self.process.poll() is None:
            return
        run = self.work / uuid.uuid4().hex
        run.mkdir()
        config = run/'config.json'
        config.write_text(json.dumps(self.options))
        threads = str(ocr_threads())
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS=threads, MKL_NUM_THREADS=threads,
                   OPENBLAS_NUM_THREADS=threads, FLAGS_use_mkldnn='1', WVF_OCR_THREADS=threads, WVF_LOCAL='0')
        env['PYTHONPATH'] = os.pathsep.join(str(p) for p in sys.path if p)
        self.process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), '--worker', str(config)],
            env=env, stdin=subprocess.DEVNULL, stdout=(run/'worker.log').open('a'), stderr=subprocess.STDOUT)
        ready = run/'ready.json'
        deadline = time.monotonic()+300
        while time.monotonic()<deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f'CPU OCR worker exited {self.process.returncode}; see {run / "worker.log"}')
            if ready.exists():
                self.url = 'http://127.0.0.1:'+str(json.loads(ready.read_text())['port'])
                return
            time.sleep(.2)
        self.close()
        raise TimeoutError('CPU OCR worker startup exceeded 300 seconds')

    def _get_model(self):
        with self.lock:
            self._start()
        return self

    def __call__(self, image_path):
        with self.lock:
            for attempt in range(2):
                self._start()
                request = urllib.request.Request(self.url+'/detect',
                    data=json.dumps(dict(path=str(image_path))).encode(), headers={'Content-Type':'application/json'})
                try:
                    with self.opener.open(request, timeout=120) as response:
                        result = json.loads(response.read())
                    if 'error' in result:
                        raise RuntimeError(result['error'])
                    return result['regions']
                except Exception:
                    self.close()
                    if attempt:
                        raise


def worker(config):
    import resource
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    from functools import partial
    import paddleocr
    paddleocr.PaddleOCR = partial(paddleocr.PaddleOCR, enable_mkldnn=True, cpu_threads=ocr_threads())
    from sub_agents.ocr import PaddleOCRBackend
    model = PaddleOCRBackend(**json.loads(config.read_text()))
    model._get_model()
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            try:
                payload=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                result=dict(regions=model(payload['path']))
            except Exception as error:
                result=dict(error=f'{type(error).__name__}: {error}')
            data=json.dumps(result).encode()
            self.send_response(200);self.send_header('Content-Length',str(len(data)))
            self.end_headers();self.wfile.write(data)
        def log_message(self,*args):
            pass
    server=HTTPServer(('127.0.0.1',0),Handler)
    (config.parent/'ready.json').write_text(json.dumps(dict(port=server.server_port,pid=os.getpid())))
    server.serve_forever()


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--worker',type=Path,required=True)
    worker(parser.parse_args().worker)
