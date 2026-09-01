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
import sys
import subprocess
import time
import atexit

import requests

# IMPORTANT: force unbuffered stdout/stderr as early as possible. Cog's
# runtime captures our print() output as logs, but Python buffers stdout
# when it's not attached to a real terminal (which is always true inside a
# container). Combined with a known Cog issue where a crashed/hung predictor
# can leave "no useful logging" behind (see Replicate's own changelog on the
# Go-based Cog runtime rewrite), this is very likely why prior setup
# failures showed literally zero log output, even the debug prints we
# added before llama-server was even started.
sys.stdout = os.fdopen(sys.stdout.fileno(), "w", buffering=1)
sys.stderr = os.fdopen(sys.stderr.fileno(), "w", buffering=1)
from cog import BasePredictor, Input

LLAMA_SERVER_BIN = os.environ.get("LLAMA_SERVER_BIN_OVERRIDE", "/src/bin/llama-server")
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
    # needs the first shard path and auto-discovers the rest. Prefer an
    # explicit "00001-of-" match; only fall back to a lone non-split file if
    # none exists. If multiple "00001-of-" matches exist (e.g. an extra
    # unrelated GGUF got pulled in), warn loudly instead of silently picking
    # one - this is likely the actual bug if shard counts don't match.
    first_shard_matches = [c for c in candidates if "00001-of-" in c]
    if len(first_shard_matches) > 1:
        print(f"WARNING: multiple '00001-of-' candidates found: {first_shard_matches}")
        print("This likely means an unexpected extra GGUF was downloaded. "
              "Picking the first one, but verify allow_patterns in cog.yaml.")
    if first_shard_matches:
        return first_shard_matches[0]
    non_split = [c for c in candidates if "-of-" not in c]
    if non_split:
        return non_split[0]
    return candidates[0]


class Predictor(BasePredictor):
    def setup(self):
        # DEBUG: log every file actually present under MODEL_DIR before doing
        # anything else, since we've hit a mismatch between expected shard
        # count and what got downloaded. Remove once shard detection is
        # confirmed working.
        print(f"--- Contents of {MODEL_DIR} ---")
        for root, _, files in os.walk(MODEL_DIR):
            for f in files:
                full = os.path.join(root, f)
                size = os.path.getsize(full)
                print(f"{full}  ({size / 1e9:.2f} GB)")
        print("--- end listing ---")

        model_path = _find_gguf_first_shard(MODEL_DIR)
        print(f"Selected first shard: {model_path}")

        cmd = [
            "stdbuf", "-oL", "-eL",  # force line-buffered stdout/stderr on
                                       # llama-server itself - many CLI tools
                                       # switch to full buffering when their
                                       # output isn't a real terminal, which
                                       # would otherwise delay/hide its logs
                                       # the same way our own were hidden.
            LLAMA_SERVER_BIN,
            "--model", model_path,
            "--host", SERVER_HOST,
            "--port", str(SERVER_PORT),
            "--n-gpu-layers", str(N_GPU_LAYERS),
            "--ctx-size", str(CTX_SIZE),
            "--jinja",  # required: applies the GLM-5.3-Flash chat template
            "--flash-attn", "on",  # this build's arg parser requires an
                                     # explicit value (on/off/auto) - passing
                                     # bare --flash-attn causes it to
                                     # swallow the next argument as its
                                     # value and fail immediately with
                                     # "unknown value for --flash-attn:
                                     # '--parallel'" (confirmed via direct
                                     # CLI test of this exact binary).
            "--parallel", "1",
            "--no-kv-unified",
        ]

        # IMPORTANT: capture llama-server's stdout/stderr instead of letting
        # them vanish. Without this, if llama-server hangs or crashes during
        # model load, there is zero visibility into why - which is very
        # likely why prior failures showed no useful error at all.
        self.server_proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        atexit.register(self._shutdown)

        # Stream llama-server's output to our own stdout in a background
        # thread so it shows up in Replicate's logs in real time, rather
        # than only being visible if/when we read it after the fact.
        import threading

        def _pump_output(proc):
            for line in proc.stdout:
                print(f"[llama-server] {line}", end="")

        self._log_thread = threading.Thread(
            target=_pump_output, args=(self.server_proc,), daemon=True
        )
        self._log_thread.start()

        base_url = f"http://{SERVER_HOST}:{SERVER_PORT}"
        # Weights are already on local disk (baked into image), so this
        # should mostly be model-load time, not download time. Still generous
        # since it's a ~93GB load from disk into GPU/CPU memory.
        self._wait_for_server(base_url, timeout_s=3600)  # 60 min
        self.base_url = base_url

    def _wait_for_server(self, base_url: str, timeout_s: int):
        deadline = time.time() + timeout_s
        last_status = None
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
                if r.status_code != last_status:
                    # llama-server's /health can return non-200 (e.g. 503)
                    # while still loading - log status changes so we can see
                    # it's making progress rather than assuming a silent hang.
                    print(f"[health] status={r.status_code} body={r.text[:200]}", flush=True)
                    last_status = r.status_code
            except requests.exceptions.ConnectionError:
                # server not listening yet at all - expected early on
                pass
            except requests.exceptions.Timeout:
                # IMPORTANT: previously uncaught - a /health request that
                # itself hangs (e.g. because the server's event loop is
                # blocked by synchronous model-loading work on the same
                # thread) would raise here and crash setup() with an
                # unrelated-looking traceback instead of a clear timeout.
                print("[health] request to /health timed out, retrying...", flush=True)
            time.sleep(2)
        raise TimeoutError(
            f"llama-server did not become healthy within {timeout_s}s "
            f"(last /health status: {last_status})"
        )

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
