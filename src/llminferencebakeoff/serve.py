"""Combined LLM serving with 3-way comparison: Transformers vs SGLang vs vLLM.

Single deployment command serves all three backends and comparison UI.

Usage:
    uv run modal serve src/llminferencebakeoff/serve.py
"""

import asyncio
import json
import subprocess
import time

import modal

from llminferencebakeoff.config import (
    DISABLE_PREFIX_CACHING,
    ENABLE_MTP,
    GPU_MEMORY_UTILIZATION,
    MAX_CONTEXT_LENGTH,
)
from llminferencebakeoff.utils import (
    DEFAULT_MAX_TOKENS,
    GPU_CONFIG,
    MAX_TOKENS_LIMIT,
    MODAL_TIMEOUT,
    MODEL_NAME,
    SCALEDOWN_WINDOW,
    SGLANG_PORT,
    VLLM_PORT,
    _stop_server,
    _stream_openai,
    _wait_for_server,
    hf_cache_vol,
    sglang_cache_vol,
    sglang_image,
    transformers_image,
    vllm_cache_vol,
    vllm_image,
)

app = modal.App("LlmInferenceBakeOff")
health_dict = modal.Dict.from_name("llminferencebakeoff-health", create_if_missing=True)

web_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi==0.115.0", "pydantic==2.10.0")
    .add_local_python_source("llminferencebakeoff")
)


# --- Inference backends ---


@app.cls(
    image=transformers_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    min_containers=1,
)
class TransformersInference:
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
        health_dict["transformers"] = "running"

    @modal.method()
    def generate_stream(
        self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, use_prefix_caching: bool = False
    ):
        # Uses TextIteratorStreamer so generate() runs in a thread while tokens
        # stream to the UI. Per-token timestamps are not used for final metrics.
        # the TextIteratorStreamer queue introduces timing noise that inflates
        # throughput; instead, total_time / token_count is reported.
        from threading import Thread

        from transformers import TextIteratorStreamer

        start_time = time.time()
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        thread = Thread(
            target=self.model.generate,
            kwargs={
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
                "streamer": streamer,
                "max_new_tokens": max_tokens,
                "do_sample": False,
            },
        )
        thread.start()

        first_token_time = None
        total_chars = 0
        full_output = ""

        for chunk in streamer:
            current_time = time.time()
            if first_token_time is None:
                first_token_time = current_time

            chunk_chars = len(chunk)
            total_chars += chunk_chars
            full_output += chunk

            yield {
                "backend": "transformers",
                "token": chunk,
                "token_count": total_chars,
                "elapsed_ms": int((current_time - start_time) * 1000),
                "is_char_count": True,  # Indicate this is character count, not token count
            }

        thread.join()
        end_time = time.time()

        total_time_s = end_time - start_time
        ttft_s = first_token_time - start_time if first_token_time else total_time_s
        total_tokens = len(self.tokenizer.encode(full_output, add_special_tokens=False))

        yield {
            "backend": "transformers",
            "final": True,
            "token_count": total_tokens,
            "time_to_first_token_ms": int(ttft_s * 1000),
            "decode_throughput": round(total_tokens / total_time_s, 2) if total_time_s > 0 else 0,
            "avg_inter_token_latency_ms": int(total_time_s * 1000 / total_tokens)
            if total_tokens > 1
            else 0,
            "total_time_ms": int(total_time_s * 1000),
        }


@app.cls(
    image=sglang_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/modal_cache/hf": hf_cache_vol,
        "/modal_cache/sglang": sglang_cache_vol,
    },
    min_containers=1,
)
class SGLangInference:
    @modal.enter()
    def start_server(self):
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
        else:
            cmd += ["--mamba-scheduler-strategy", "extra_buffer"]
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
        health_dict["sglang"] = "running"

    @modal.exit()
    def stop_server(self):
        health_dict["sglang"] = "down"
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def generate_stream(
        self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, use_prefix_caching: bool = False
    ):
        if self.process.poll() is not None:
            raise RuntimeError(f"SGLang process exited (code {self.process.returncode})")
        cache_key = "prefix_experiment" if use_prefix_caching else None
        yield from _stream_openai(
            self.port, "sglang", [{"role": "user", "content": prompt}], max_tokens, cache_key
        )


@app.cls(
    image=vllm_image,
    gpu=GPU_CONFIG,
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    min_containers=1,
)
class VLLMInference:
    @modal.enter()
    def start_server(self):
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
        else:
            cmd += ["--enable-prefix-caching", "--mamba-cache-mode", "align"]
        if ENABLE_MTP:
            cmd += ["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 1}']

        self.process = subprocess.Popen(cmd)
        _wait_for_server(self.process, self.port, "vLLM")
        health_dict["vllm"] = "running"

    @modal.exit()
    def stop_server(self):
        health_dict["vllm"] = "down"
        _stop_server(getattr(self, "process", None))

    @modal.method()
    def generate_stream(
        self, prompt: str, max_tokens: int = DEFAULT_MAX_TOKENS, use_prefix_caching: bool = False
    ):
        if self.process.poll() is not None:
            raise RuntimeError(f"vLLM process exited (code {self.process.returncode})")
        cache_key = "prefix_experiment" if use_prefix_caching else None
        yield from _stream_openai(
            self.port, "vllm", [{"role": "user", "content": prompt}], max_tokens, cache_key
        )


@app.function(image=web_image, timeout=MODAL_TIMEOUT)
@modal.asgi_app()
def web():
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, StreamingResponse
    from pydantic import BaseModel, Field

    from llminferencebakeoff.ui import page_html

    web_app = FastAPI(title="LLM Inference Bake-Off: HuggingFace Transformers vs SGLang vs vLLM")

    class GenerateRequest(BaseModel):
        prompt: str = Field(..., description="Text prompt for generation")
        max_tokens: int = Field(default=DEFAULT_MAX_TOKENS, ge=1, le=MAX_TOKENS_LIMIT)
        use_prefix_caching: bool = Field(default=False, description="Enable prefix caching")

    @web_app.get("/v1/health")
    async def health_endpoint():
        return {
            backend: await health_dict.get.aio(backend, "starting")
            for backend in ("transformers", "sglang", "vllm")
        }

    @web_app.post("/v1/compare")
    async def compare_endpoint(request: GenerateRequest):
        async def triple_stream():
            queue: asyncio.Queue = asyncio.Queue()

            async def consume_backend(gen, backend_name):
                try:
                    async for chunk in gen:
                        await queue.put(chunk)
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    await queue.put({"error": backend_name, "message": str(e)})
                finally:
                    queue.put_nowait({"done": backend_name})

            kwargs = dict(
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                use_prefix_caching=request.use_prefix_caching,
            )
            tasks = [
                asyncio.create_task(
                    consume_backend(
                        TransformersInference().generate_stream.remote_gen.aio(**kwargs),
                        "transformers",
                    )
                ),
                asyncio.create_task(
                    consume_backend(
                        SGLangInference().generate_stream.remote_gen.aio(**kwargs), "sglang"
                    )
                ),
                asyncio.create_task(
                    consume_backend(
                        VLLMInference().generate_stream.remote_gen.aio(**kwargs), "vllm"
                    )
                ),
            ]

            completed = set()
            try:
                while len(completed) < 3:
                    chunk = await queue.get()
                    if "done" in chunk:
                        completed.add(chunk["done"])
                    else:
                        yield f"data: {json.dumps(chunk)}\n\n"
            except Exception:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

            await asyncio.gather(*tasks, return_exceptions=True)
            yield "data: [DONE]\n\n"

        return StreamingResponse(
            triple_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @web_app.get("/", response_class=HTMLResponse)
    async def viewer():
        return page_html()

    return web_app
