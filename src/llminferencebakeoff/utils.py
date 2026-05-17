"""Shared configuration, images, and helpers for the LLM inference bake-off.

Import as: from llminferencebakeoff.utils import ...
"""

import subprocess
import time

import modal

from llminferencebakeoff.config import (
    GPU_CONFIG,
    MODEL_NAME,
)

MINUTES = 60

# Sampling Parameters
DEFAULT_MAX_TOKENS = 512
MAX_TOKENS_LIMIT = 2048
TEMPERATURE = 0
TOP_P = 1.0
SEED = 42

# Server Configuration
SGLANG_PORT = 8000
VLLM_PORT = 8001
SERVER_STARTUP_TIMEOUT = 600
SERVER_HEALTH_CHECK_INTERVAL = 1

# Modal Configuration
MODAL_TIMEOUT = 10 * MINUTES
SCALEDOWN_WINDOW = 5 * MINUTES

# ----------------------------------------------------------------------
# Volumes
# ----------------------------------------------------------------------

hf_cache_vol = modal.Volume.from_name(
    "llminferencebakeoff-huggingface-cache", create_if_missing=True
)
vllm_cache_vol = modal.Volume.from_name("llminferencebakeoff-vllm-cache", create_if_missing=True)
sglang_cache_vol = modal.Volume.from_name(
    "llminferencebakeoff-sglang-cache", create_if_missing=True
)

# ----------------------------------------------------------------------
# Images
# ----------------------------------------------------------------------

# L4 image: torch 2.11 on cuda 12.9 (matching cublas versions, no CUBLAS issue).
# flash-linear-attention alone (triton-based, no compilation) gives ~20 tok/s on L4.
# causal-conv1d is omitted - no prebuilt wheel exists for torch 2.11.
transformers_image_l4 = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "transformers==5.8.0",
        "torch==2.11.0",
        "accelerate==1.13.0",
        "hf-transfer==0.1.9",
        "flash-linear-attention",
        extra_options="--torch-backend=cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("llminferencebakeoff")
)

# Prebuilt wheel avoids compiling causal-conv1d from source, which OOMs Modal's image builder.
# torch2.10 + cxx11abiTRUE is the only cu12/cp312 variant released; no FALSE variant exists.
_causal_conv1d_wheel = (
    "https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2.post1/"
    "causal_conv1d-1.6.2.post1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
)

# H100 image: torch 2.10 on cuda 12.8 (matching cublas 12.8.4.1 pinned by torch 2.10).
# Using cuda:12.9 caused CUBLAS_STATUS_INVALID_VALUE at runtime (pytorch#174949).
# Qwen3.5 has hybrid linear-attention layers; without fla it falls back to
# a slow pure-torch implementation (~5 tok/s on H100 vs ~50+ with fla).
transformers_image_h100 = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "transformers==5.8.0",
        "torch==2.10.0",
        "accelerate==1.13.0",
        "hf-transfer==0.1.9",
        "flash-linear-attention",
        extra_options="--torch-backend=cu128",
    )
    .uv_pip_install(_causal_conv1d_wheel)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("llminferencebakeoff")
)

transformers_image = transformers_image_l4 if GPU_CONFIG == "L4" else transformers_image_h100

sglang_image = (
    modal.Image.from_registry("lmsysorg/sglang:v0.5.11-cu129", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("requests", "hf-transfer==0.1.9", "numpy")
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "SGLANG_DISABLE_CUDNN_CHECK": "1",
            "HF_HOME": "/modal_cache/hf",
        }
    )
    .add_local_python_source("llminferencebakeoff")
)

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.20.1",
        "hf-transfer==0.1.8",
        extra_options="--torch-backend=cu130",
    )
    .env(
        {
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "TRANSFORMERS_VERBOSITY": "error",
            "VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS": "1",
        }
    )
    .add_local_python_source("llminferencebakeoff")
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _compute_metrics(token_timestamps: list, start_time: float, token_count: int) -> dict:
    if not token_timestamps:
        return {
            "time_to_first_token_ms": 0,
            "decode_throughput": 0,
            "avg_inter_token_latency_ms": 0,
            "total_time_ms": 0,
        }
    first_token_time = token_timestamps[0]
    last_token_time = token_timestamps[-1]
    ttft_ms = int((first_token_time - start_time) * 1000)
    if token_count > 1:
        decode_time = last_token_time - first_token_time
        decode_throughput = (token_count - 1) / decode_time if decode_time > 0 else 0
        avg_itl_ms = int(decode_time / (token_count - 1) * 1000)
    else:
        decode_throughput = 0
        avg_itl_ms = 0
    return {
        "time_to_first_token_ms": ttft_ms,
        "decode_throughput": round(decode_throughput, 2),
        "avg_inter_token_latency_ms": avg_itl_ms,
        "total_time_ms": int((last_token_time - start_time) * 1000),
    }


def _wait_for_server(process: subprocess.Popen, port: int, label: str) -> None:
    import requests

    print(f"Waiting for {label} server...")
    deadline = time.time() + SERVER_STARTUP_TIMEOUT
    time.sleep(10)
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{label} process exited with code {process.returncode}")
        elapsed = int(time.time() - (deadline - SERVER_STARTUP_TIMEOUT))
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=5)
            if r.status_code == 200:
                print(f"{label} ready after {elapsed}s")
                return
            if r.status_code == 503 and elapsed % 20 < SERVER_HEALTH_CHECK_INTERVAL:
                print(f"{label} initializing... ({elapsed}s elapsed)")
        except requests.exceptions.RequestException:
            if elapsed % 20 < SERVER_HEALTH_CHECK_INTERVAL:
                print(f"Waiting for {label}... ({elapsed}s elapsed)")
        time.sleep(SERVER_HEALTH_CHECK_INTERVAL)
    raise RuntimeError(f"{label} failed to start within {SERVER_STARTUP_TIMEOUT}s")


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def _stream_openai(port: int, backend: str, messages: list, max_tokens: int, cache_key: str = None):
    import json

    import requests

    start_time = time.time()
    request_data = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "seed": SEED,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    if cache_key:
        request_data["extra_body"] = {"cache_salt": cache_key}

    response = requests.post(
        f"http://localhost:{port}/v1/chat/completions",
        json=request_data,
        stream=True,
    )
    token_timestamps = []
    token_count = 0
    usage_stats = None

    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if not decoded.startswith("data: "):
            continue
        data = decoded[6:]
        if data == "[DONE]":
            continue
        try:
            chunk = json.loads(data)
            if "usage" in chunk:
                usage_stats = chunk["usage"]
            if chunk.get("choices"):
                token_text = chunk["choices"][0].get("delta", {}).get("content", "")
                if token_text:
                    current_time = time.time()
                    token_timestamps.append(current_time)
                    token_count += 1
                    yield {
                        "backend": backend,
                        "token": token_text,
                        "token_count": token_count,
                        "elapsed_ms": int((current_time - start_time) * 1000),
                    }
        except json.JSONDecodeError:
            pass

    if token_timestamps:
        timing_count = len(token_timestamps)
        reported_count = (
            usage_stats.get("completion_tokens", timing_count) if usage_stats else timing_count
        )
        yield {
            "backend": backend,
            "final": True,
            "token_count": reported_count,
            **_compute_metrics(token_timestamps, start_time, reported_count),
        }
