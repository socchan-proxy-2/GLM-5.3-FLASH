"""
Replicate Cog predictor for GLM-5.3-Flash (UD-IQ3_XXS GGUF, ~120GB) via
llama.cpp's llama-server, on 2x A100 80GB (160GB VRAM total).

IMPORTANT CAVEATS (read before trusting this in production):
- glm5_next is a brand-new architecture (2026-08-26). This uses Unsloth's
  fork branch (glm5next/upstream), NOT mainline llama.cpp, because the
  upstream PRs (#27752, #27754) are unmerged as of this writing.
- No one has published A100 (Ampere, sm_80) benchmarks for this model yet.
  H100/H200/B200/GB300 are the only hardware with published numbers.
  Loading may simply work (CUDA is CUDA), but treat first-run behavior as
  unverified territory - watch VRAM usage and generation quality closely.
- UD-IQ3_XXS retains ~82% of top-1 accuracy vs BF16 per Unsloth's own
  measurements - a real quality tradeoff, not just a memory one.
- --jinja is required for GLM-5.3-Flash's chat template to apply correctly.
"""

import os
import subprocess
import time
import atexit
from typing import Optional

import requests
from huggingface_hub import snapshot_download
from cog import BasePredictor, Input

LLAMA_SERVER_BIN = "/src/bin/llama-server"
MODEL_DIR = "/src/model"
HF_REPO = "unsloth/GLM-5.3-Flash-GGUF"
MODEL_GLOB_HINT = "UD-IQ3_XXS"  # actual filename(s) live under MODEL_DIR; ~120GB, ~82% top-1 accuracy retained
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8080
N_GPU_LAYERS = 999  # push everything onto GPU; llama.cpp will cap at what fits
CTX_SIZE = 32768    # start conservative; GLM-5.3-Flash supports up to 1,048,576


def _ensure_model_downloaded(repo_id: str, local_dir: str, glob_hint: str):
    """
    Downloads the quantized GGUF from Hugging Face at setup() time rather than
    build time. This keeps local `cog build` lightweight (no ~120GB needed on
    the machine running `cog push`), at the cost of every cold start on
    Replicate re-downloading ~120GB before it can serve a request.
    """
    if os.path.isdir(local_dir) and any(
        glob_hint in f for _, _, files in os.walk(local_dir) for f in files
    ):
        return  # already present (e.g. warm container reusing disk)
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[f"*{glob_hint}*"],
    )


def _find_gguf_first_shard(model_dir: str) -> str:
    """UD-IQ3_XXS ships as split GGUF shards; llama.cpp wants the first one."""
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
        _ensure_model_downloaded(HF_REPO, MODEL_DIR, MODEL_GLOB_HINT)
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
        ]

        self.server_proc = subprocess.Popen(cmd)
        atexit.register(self._shutdown)

        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        self._wait_for_server(base_url, timeout_s=10800)  # ~120GB download + load from scratch each cold start (180 min)
        self.base_url = base_url

    def _wait_for_server(self, base_url: str, timeout_s: int):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.server_proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server exited early with code {self.server_proc.returncode}. "
                    "Check build logs - this is often a CUDA/kernel incompatibility "
                    "on unverified hardware like A100."
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

