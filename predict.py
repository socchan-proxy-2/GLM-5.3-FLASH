"""
Replicate Cog predictor for GLM-5.3-Flash (UD-IQ1_M GGUF, ~93GB, baked into
the image at build time) via llama.cpp's llama-server, on 2x A100 80GB
(160GB VRAM total).

IMPORTANT CAVEATS (read before trusting this in production):
- glm5_next is a brand-new architecture (2026-08-26). This uses Unsloth's
  fork branch (glm5next/upstream), NOT mainline llama.cpp, because the
  upstream PRs (#27752, #27754) are unmerged as of this writing.
- No one has published A100 (Ampere, sm_80) benchmarks for this model yet.
  H100/H200/B200/GB300 are the only hardware with published numbers.
- UD-IQ1_M (1-bit dynamic quant) retains ~71% of top-1 accuracy vs BF16 per
  Unsloth's own measurements - a real, noticeable quality drop, traded for
  fitting comfortably in 160GB VRAM with room to spare and no per-cold-start
  download (weights are baked into the image, see cog.yaml).
- --jinja is required for GLM-5.3-Flash's chat template to apply correctly.
- --parallel 1 --no-kv-unified works around a known crash: GLM-5.3-Flash's
  pooled indexer cache is incompatible with unified KV cache + >1 parallel
  sequence (see https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/discussions/1).
  Omitting this causes an immediate startup crash with no clean error
  message - this was very likely the root cause of earlier
  "RemoteProtocolError: peer closed connection ... setup failed" errors.
"""

import os
import subprocess
import time
import atexit

import requests
from cog import BasePredictor, Input

LLAMA_SERVER_BIN = "/src/bin/llama-server"
MODEL_DIR = "/src/model"  # baked into the image at build time - see cog.yaml
MODEL_GLOB_HINT = "UD-IQ1_M"  # actual filename(s) live under MODEL_DIR; (exact size/accuracy per Unsloth model card)
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
N_GPU_LAYERS = 999  # push everything onto GPU; llama.cpp will cap at what fits
CTX_SIZE = 32768    # start conservative; GLM-5.3-Flash supports up to 1,048,576


def _find_gguf_first_shard(model_dir: str) -> str:
    """UD-IQ1_M ships as split GGUF shards; llama.cpp wants the first one."""
    candidates = []
    for root, _, files in os.walk(model_dir):
        for f in files:
            if f.endswith(".gguf") and MODEL_GLOB_HINT in f:
                candidates.append(os.path.join(root, f))
    if not candidates:
        raise FileNotFoundError(
            f"No {MODEL_GLOB_HINT} GGUF files found under {model_dir}. "
            "Did the build-time `hf download` step complete?"
        )
    candidates.sort()
    # Split GGUFs are named like ...-00001-of-00006.gguf; llama.cpp only
    # needs the first shard path and auto-discovers the rest.
    for c in candidates:
        if "00001-of-" in c or "-of-" not in c:
            return c
    return candidates[0]


class Predictor(BasePredictor):
    def setup(self):
        model_path = _find_gguf_first_shard(MODEL_DIR)

        cmd = [
            LLAMA_SERVER_BIN,
            "--model", model_path,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--n-gpu-layers", str(N_GPU_LAYERS),
            "--ctx-size", str(CTX_SIZE),
            "--jinja",  # required: applies the GLM-5.3-Flash chat template
            "--flash-attn",
            "--parallel", "1",
            "--no-kv-unified",
        ]

        self.server_proc = subprocess.Popen(cmd)
        atexit.register(self._shutdown)

        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        # Weights are already on local disk (baked into image), so this
        # should mostly be model-load time, not download time. Still generous
        # since it's a ~93GB load from disk into GPU/CPU memory.
        self._wait_for_server(base_url, timeout_s=3600)  # 60 min
        self.base_url = base_url

    def _wait_for_server(self, base_url: str, timeout_s: int):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.server_proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self.server_proc.returncode}. "
                    "Check logs - this is often a CUDA/kernel incompatibility "
                    "on unverified hardware like A100, or a missing --no-kv-unified "
                    "type flag mismatch."
                )
            try:
                r = requests.get(f"{base_url}/health", timeout=5)
                if r.status_code == 200:
                    return
            except requests.exceptions.ConnectionError:
                pass
            time.sleep(2)
        raise TimeoutError("llama-server did not become healthy within timeout")

    def _shutdown(self):
        if getattr(self, "server_proc", None) and self.server_proc.poll() is None:
            self.server_proc.terminate()

    def predict(
        self,
        prompt: str = Input(description="User prompt"),
        system_prompt: str = Input(description="Optional system prompt", default=""),
        max_tokens: int = Input(description="Max tokens to generate", default=1024, ge=1, le=8192),
        temperature: float = Input(description="Sampling temperature", default=1.0, ge=0.0, le=2.0),
        top_p: float = Input(description="Top-p nucleus sampling", default=0.95, ge=0.0, le=1.0),
        reasoning_effort: str = Input(
            description="Thinking budget", default="max", choices=["low", "high", "max"]
        ),
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": "glm-5.3-flash",
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "reasoning_effort": reasoning_effort,
        }

        resp = requests.post(
            f"{self.base_url}/v1/chat/completions", json=payload, timeout=600
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
