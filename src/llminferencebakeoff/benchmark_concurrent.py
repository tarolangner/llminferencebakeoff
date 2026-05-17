"""Experiment 4: Concurrent Requests Benchmark.

Measures continuous batching and chunked prefill benefits by sending N
staggered requests to one backend at a time. Requests are sent with a small
delay (STAGGER_DELAY) between each one to simulate distinct users arriving
at different timestamps, rather than a single pre-batched bundle.

Each backend runs independently to save cost. Only one GPU container is
active at any point.

Key demonstration:
  - vLLM/SGLang slot each new arrival into the ongoing decode batch
    (real continuous batching behavior)
  - HF Transformers queues each arrival and processes sequentially
    (no continuous batching)

Usage:
    uv run modal run src/llminferencebakeoff/benchmark_concurrent.py

Expected results:
    - vLLM/SGLang: aggregate throughput scales with batch size
    - HF Transformers: flat throughput regardless of batch size
    - The divergence is the key finding for the blog post
"""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from threading import Thread

import modal

from llminferencebakeoff.config import (
    DISABLE_PREFIX_CACHING,
    ENABLE_MTP,
    GPU_MEMORY_UTILIZATION,
    MAX_CONTEXT_LENGTH,
)
from llminferencebakeoff.utils import (
    GPU_CONFIG,
    MODAL_TIMEOUT,
    MODEL_NAME,
    SCALEDOWN_WINDOW,
    SEED,
    SGLANG_PORT,
    TEMPERATURE,
    TOP_P,
    VLLM_PORT,
    _stop_server,
    _wait_for_server,
    hf_cache_vol,
    sglang_cache_vol,
    sglang_image,
    transformers_image,
    vllm_cache_vol,
    vllm_image,
)

# Number of concurrent requests. Repeat size of 1 for backend warm-up
BATCH_SIZES = [1, 1, 1, 4, 8, 16, 32, 64, 128]

STAGGER_DELAY = 0.1
PROMPT = "Write a short paragraph about LLM inference serving optimizations."
MAX_TOKENS = 512

app = modal.App("LlmInferenceBakeOffExp4")


def _send_one(port: int, backend: str, prompt: str, max_tokens: int) -> dict | None:
    """Send one streaming request via OpenAI API.

    Returns the final metrics chunk containing token_count and TTFT,
    or None if the request failed (server not ready, transient error, etc.).
    """
    import requests

    try:
        start_time = time.monotonic()
        response = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "stream": True,
                "stream_options": {"include_usage": True},
            },
            stream=True,
            timeout=(30, 300),  # connect timeout, read timeout
        )
        response.raise_for_status()
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
                        token_timestamps.append(time.monotonic())
                        token_count += 1
            except json.JSONDecodeError:
                pass

        if token_timestamps:
            timing_count = len(token_timestamps)
            reported_count = (
                usage_stats.get("completion_tokens", timing_count) if usage_stats else timing_count
            )
            return {
                "token_count": reported_count,
                "time_to_first_token_ms": int((token_timestamps[0] - start_time) * 1000),
            }
    except (
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.HTTPError,
    ) as exc:
        print(f"  [!] Request to {backend} failed: {exc}")
    return None


def _run_benchmark(port: int, backend_label: str):
    """Staggered batch benchmark loop for HTTP-based backends (SGLang, vLLM).

    Yields (batch_size_str, summary) for every entry in BATCH_SIZES, including
    duplicate batch sizes (e.g. repeated 1s used as warmup passes).
    """
    for n in BATCH_SIZES:
        batch_start = time.time()
        per_request = []

        with ThreadPoolExecutor(max_workers=n) as pool:
            futures = []
            for _ in range(n):
                futures.append(pool.submit(_send_one, port, backend_label, PROMPT, MAX_TOKENS))
                time.sleep(STAGGER_DELAY)
            for f in as_completed(futures):
                chunk = f.result()
                if chunk:
                    per_request.append(chunk)

        batch_end = time.time()
        yield str(n), _summarize(per_request, n, batch_start, batch_end)


def _summarize(results: list[dict], n: int, batch_start: float, batch_end: float) -> dict:
    """Aggregate per-request results into benchmark summary for one batch size."""
    import numpy as np

    total_tokens = sum(r.get("token_count", 0) for r in results if r)
    ttfts = [r.get("time_to_first_token_ms", 0) for r in results if r]
    elapsed = batch_end - batch_start if n > 0 else 1
    return {
        "aggregate_throughput": round(total_tokens / elapsed, 2),
        "p50_ttft_ms": round(float(np.quantile(ttfts, 0.50)), 0) if ttfts else 0,
        "p95_ttft_ms": round(float(np.quantile(ttfts, 0.95)), 0) if ttfts else 0,
        "total_tokens": total_tokens,
        "wall_time_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# HF Transformers backend (no HTTP server, direct model calls)
# ---------------------------------------------------------------------------


@app.cls(
    image=transformers_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    min_containers=0,
)
class HFTransformersBench:
    @modal.enter()
    def load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    @modal.method()
    def run_benchmark(self):
        import torch
        from transformers import TextIteratorStreamer

        torch.manual_seed(SEED)

        messages = [{"role": "user", "content": PROMPT}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        def process_one() -> dict:
            t0 = time.time()
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            thread = Thread(
                target=self.model.generate,
                kwargs={
                    "input_ids": inputs["input_ids"],
                    "attention_mask": inputs["attention_mask"],
                    "streamer": streamer,
                    "max_new_tokens": MAX_TOKENS,
                    "do_sample": False,
                },
            )
            thread.start()
            first = True
            output_text = ""
            for chunk in streamer:
                if first:
                    ttft = int((time.time() - t0) * 1000)
                    first = False
                output_text += chunk
            thread.join()
            token_count = len(self.tokenizer.encode(output_text, add_special_tokens=False))
            return {"token_count": token_count, "time_to_first_token_ms": ttft}

        for n in BATCH_SIZES:
            batch_start = time.time()
            per_request = []

            with ThreadPoolExecutor(max_workers=1) as pool:
                futures = []
                for _ in range(n):
                    futures.append(pool.submit(process_one))
                    time.sleep(STAGGER_DELAY)
                for f in as_completed(futures):
                    chunk = f.result()
                    if chunk:
                        per_request.append(chunk)

            batch_end = time.time()
            yield str(n), _summarize(per_request, n, batch_start, batch_end)


# ---------------------------------------------------------------------------
# SGLang backend (HTTP server, concurrent requests via OpenAI API)
# ---------------------------------------------------------------------------


@app.cls(
    image=sglang_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/modal_cache/hf": hf_cache_vol,
        "/modal_cache/sglang": sglang_cache_vol,
    },
    min_containers=0,
)
class SGLangBench:
    @modal.enter()
    def start_server(self):
        import subprocess

        self.port = SGLANG_PORT
        cmd = [
            "sglang",
            "serve",
            "--model-path",
            MODEL_NAME,
            "--port",
            str(self.port),
            "--host",
            "0.0.0.0",
            "--dtype",
            "bfloat16",
            "--mem-fraction-static",
            str(GPU_MEMORY_UTILIZATION),
            "--context-length",
            str(MAX_CONTEXT_LENGTH),
        ]
        if DISABLE_PREFIX_CACHING:
            cmd += ["--disable-radix-cache"]
        if ENABLE_MTP:
            cmd += [
                "--speculative-algo",
                "NEXTN",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "1",
            ]
        self.process = subprocess.Popen(cmd)
        _wait_for_server(self.process, self.port, "SGLang")

    @modal.exit()
    def stop_server(self):
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def cleanup(self):
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def run_benchmark(self):
        yield from _run_benchmark(self.port, "sglang")


@app.cls(
    image=vllm_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=0,
)
class VLLMBench:
    @modal.enter()
    def start_server(self):
        import subprocess

        self.port = VLLM_PORT
        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--port",
            str(self.port),
            "--host",
            "0.0.0.0",
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            str(GPU_MEMORY_UTILIZATION),
            "--max-model-len",
            str(MAX_CONTEXT_LENGTH),
        ]
        if DISABLE_PREFIX_CACHING:
            cmd += ["--no-enable-prefix-caching"]
        if ENABLE_MTP:
            cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}']

        self.process = subprocess.Popen(cmd)
        _wait_for_server(self.process, self.port, "vLLM")

    @modal.exit()
    def stop_server(self):
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def cleanup(self):
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def run_benchmark(self):
        yield from _run_benchmark(self.port, "vllm")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def main():
    print("Starting benchmarking...")
    backends = [
        ("vLLM", VLLMBench),
        ("SGLang", SGLangBench),
        # ("HF Transformers", HFTransformersBench), # Will take a long time, lower counts recommended
    ]

    print(f"{MODEL_NAME} is model")
    print(f"{GPU_CONFIG} is GPU")

    for label, cls in backends:
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {label}")
        print(f"{'=' * 60}")
        print()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_label = label.replace(" ", "_")
        csv_path = f"experiment4_{safe_label}_{timestamp}.csv"
        fieldnames = [
            "batch_size",
            "aggregate_throughput",
            "p50_ttft_ms",
            "p95_ttft_ms",
            "wall_time_s",
        ]
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        print(f"Writing results to {csv_path}")

        print(
            f"{'Batch Size':<12} {'Throughput':<16} {'P50 TTFT':<12} {'P95 TTFT':<12} {'Wall Time':<12}"
        )
        print(f"{'':-<12} {'':-<16} {'':-<12} {'':-<12} {'':-<12}")

        instance = cls()
        for n_str, r in instance.run_benchmark.remote_gen():
            print(
                f"{n_str:<12} "
                f"{r['aggregate_throughput']:<16} "
                f"{r['p50_ttft_ms']:<12} "
                f"{r['p95_ttft_ms']:<12} "
                f"{r['wall_time_s']:<12}"
            )
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(
                    {
                        "batch_size": n_str,
                        "aggregate_throughput": r["aggregate_throughput"],
                        "p50_ttft_ms": r["p50_ttft_ms"],
                        "p95_ttft_ms": r["p95_ttft_ms"],
                        "wall_time_s": r["wall_time_s"],
                    }
                )
        print()

        # Stop the server process so GPU memory is freed.
        # Container goes idle and Modal scales it down via scaledown_window.
        if hasattr(instance, "cleanup"):
            try:
                instance.cleanup.remote()
            except Exception:
                pass
