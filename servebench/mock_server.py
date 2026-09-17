"""MOCK OpenAI-compatible streaming server (stdlib only). CPU-runnable.

Implements POST /v1/completions with stream=true. Token timing comes from
the analytical mock model (serving_model.evaluate), then compressed by
TIME_SCALE so the whole thing runs on CPU without GPUs. Every response
carries X-Mock: 1 and the simulated (unscaled) TTFT/ITL so the client's
measurement path can be validated honestly.

This exists to prove the BENCH CLIENT works end-to-end. It is not a
substitute for real hardware numbers.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from .configs import ServingConfig
from .gpu_catalog import GPU, get as get_gpu
from .serving_model import evaluate, LLAMA_8B

TIME_SCALE = 0.01  # 1 simulated second -> 10 ms of wall time


class Handler(BaseHTTPRequestHandler):
    server_version = "ServeBenchMock/1.0"

    def log_message(self, *a):  # quiet
        pass

    def _send_json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlparse(self.path).path == "/health":
            self._send_json(200, {"status": "ok", "mock": True})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if urlparse(self.path).path != "/v1/completions":
            self._send_json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(length) or b"{}")
        prompt = req.get("prompt", "")
        max_tokens = int(req.get("max_tokens", 16))
        prompt_tokens = max(1, len(str(prompt).split()))

        cfg: ServingConfig = self.server.cfg
        m = evaluate(cfg, self.server.gpu, LLAMA_8B,
                     workload=[(prompt_tokens, max_tokens, 1.0)])
        if not m.feasible:
            self._send_json(422, {"error": "config infeasible in mock model"})
            return

        # SSE stream: one chunk per token, timed by the mock model.
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Mock", "1")
        self.send_header("X-Mock-TTFT-S", f"{m.ttft_s:.6f}")
        self.send_header("X-Mock-ITL-S", f"{m.itl_s:.6f}")
        self.end_headers()

        def chunk(text, finish=None):
            payload = {"choices": [{"text": text,
                                    "finish_reason": finish,
                                    "index": 0}]}
            return f"data: {json.dumps(payload)}\n\n".encode()

        time.sleep(m.ttft_s * TIME_SCALE)          # prefill
        self.wfile.write(chunk(" tok0"))
        for i in range(1, max_tokens):             # decode
            time.sleep(m.itl_s * TIME_SCALE)
            self.wfile.write(chunk(f" tok{i}"))
        time.sleep(m.itl_s * TIME_SCALE)
        self.wfile.write(chunk("", finish="length"))
        self.wfile.write(b"data: [DONE]\n\n")


def run_server(cfg: ServingConfig, gpu: GPU, port: int = 0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.cfg = cfg
    srv.gpu = gpu
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv
