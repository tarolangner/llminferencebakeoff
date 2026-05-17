"""UI assets for the web comparison interface."""

from llminferencebakeoff.utils import DEFAULT_MAX_TOKENS, MAX_TOKENS_LIMIT

CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    max-width: 1400px; margin: 0 auto; padding: 16px 20px;
    background: #2C3531; min-height: 100vh; color: #D1E8E2;
}
h1 {
    color: #D9B08C; margin-bottom: 4px; font-size: 22px; text-align: center;
    font-weight: 300; letter-spacing: 2px;
}
.subtitle {
    text-align: center; color: #D1E8E2; margin-bottom: 12px;
    font-size: 11px; opacity: 0.8;
}
.input-section {
    background: #116466; padding: 12px 16px; border-radius: 6px;
    margin-bottom: 12px; border: 1px solid rgba(209, 232, 226, 0.1);
}
label {
    display: block; margin-bottom: 4px; color: #FFCB9A;
    font-weight: 500; font-size: 11px; text-transform: uppercase;
    letter-spacing: 1px;
}
textarea {
    width: 100%; padding: 8px 10px; border: 1px solid rgba(209, 232, 226, 0.2);
    border-radius: 4px; font-family: monospace; font-size: 12px;
    resize: vertical; min-height: 52px; background: rgba(44, 53, 49, 0.5);
    color: #D1E8E2; transition: border-color 0.2s;
}
textarea:focus { outline: none; border-color: #D9B08C; }
.params { margin: 8px 0; }
button {
    background: #FFCB9A; color: #2C3531; border: none;
    padding: 8px 28px; border-radius: 4px; cursor: pointer;
    font-size: 12px; font-weight: 600; width: 100%;
    text-transform: uppercase; letter-spacing: 1px;
    transition: all 0.2s;
}
button:hover { background: #D9B08C; }
button:disabled {
    background: rgba(209, 232, 226, 0.2); color: rgba(209, 232, 226, 0.5);
    cursor: not-allowed;
}
.comparison-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
.backend-panel {
    background: #116466; padding: 12px; border-radius: 6px;
    border: 1px solid rgba(209, 232, 226, 0.1);
}
.backend-header {
    font-size: 11px; font-weight: 600; margin-bottom: 8px;
    padding-bottom: 6px; border-bottom: 1px solid rgba(209, 232, 226, 0.2);
    text-transform: uppercase; letter-spacing: 1px; color: #FFCB9A;
}
.output {
    min-height: 140px; padding: 10px; background: rgba(44, 53, 49, 0.5);
    border-radius: 4px; white-space: pre-wrap; line-height: 1.5;
    font-size: 12px; margin-bottom: 8px; font-family: monospace;
    color: #D1E8E2; border: 1px solid rgba(209, 232, 226, 0.1);
}
.metrics {
    padding: 8px; background: rgba(44, 53, 49, 0.3); border-radius: 4px;
    font-family: monospace; font-size: 10px;
    border: 1px solid rgba(209, 232, 226, 0.1); color: #D1E8E2;
    white-space: pre-wrap;
}
.metric { display: block; margin: 2px 0; }
.winner-banner {
    margin-top: 12px; padding: 12px; background: #116466;
    border: 2px solid #D9B08C; border-radius: 6px;
    text-align: center; font-size: 13px; font-weight: 500;
    display: none; color: #FFCB9A; letter-spacing: 1px;
}
"""


def page_html() -> str:
    """Return the complete HTML page for the comparison UI."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <title>LLM Inference Bake-Off</title>
    <style>{CSS}</style>
</head>
<body>
    <h1>LLM Inference Bake-Off</h1>
    <div class="subtitle">Race HuggingFace Transformers against vLLM and SGLang</div>

    <div class="input-section">
        <label for="prompt">Prompt</label>
        <textarea id="prompt">Provide a brief overview of LLM inference optimization. Make no mistakes.</textarea>

        <div class="params">
            <div style="display: flex; align-items: center; gap: 20px;">
                <div style="flex: 1;">
                    <label>Max Tokens: <span id="max_tokens_val">{DEFAULT_MAX_TOKENS}</span></label>
                    <input type="range" id="max_tokens" min="1" max="{MAX_TOKENS_LIMIT}" value="{DEFAULT_MAX_TOKENS}"
                           oninput="document.getElementById('max_tokens_val').textContent=this.value">
                </div>
                <label style="display: flex; align-items: center; cursor: pointer; white-space: nowrap;">
                    <input type="checkbox" id="prepend_prefix" style="margin-right: 8px; width: 16px; height: 16px;">
                    Prepend prefix
                </label>
            </div>
        </div>

        <button onclick="runComparison()" id="generateBtn">Generate Comparison</button>
    </div>

    <div class="comparison-grid">
        <div class="backend-panel">
            <div class="backend-header transformers"><span id="status_transformers">🟡</span> HF Transformers (Baseline)</div>
            <div id="output_transformers" class="output"></div>
            <div id="metrics_transformers" class="metrics">Waiting to initialize...</div>
        </div>

        <div class="backend-panel">
            <div class="backend-header vllm"><span id="status_vllm">🟡</span> vLLM</div>
            <div id="output_vllm" class="output"></div>
            <div id="metrics_vllm" class="metrics">Waiting to initialize...</div>
        </div>

        <div class="backend-panel">
            <div class="backend-header sglang"><span id="status_sglang">🟡</span> SGLang</div>
            <div id="output_sglang" class="output"></div>
            <div id="metrics_sglang" class="metrics">Waiting to initialize...</div>
        </div>
    </div>

    <div id="winner" class="winner-banner"></div>

    <script>
        const results = {{ transformers: {{}}, sglang: {{}}, vllm: {{}} }};

        async function runComparison() {{
            const btn = document.getElementById('generateBtn');
            btn.disabled = true;
            document.getElementById('winner').style.display = 'none';

            ['transformers', 'sglang', 'vllm'].forEach(backend => {{
                document.getElementById(`output_${{backend}}`).textContent = '';
                document.getElementById(`metrics_${{backend}}`).textContent = 'Initializing...';
                results[backend] = {{}};
            }});

            const prompt = document.getElementById('prompt').value;
            const max_tokens = parseInt(document.getElementById('max_tokens').value);
            const prependPrefix = document.getElementById('prepend_prefix').checked;

            let finalPrompt = prompt;
            if (prependPrefix) {{
                const prefix = 'poem '.repeat(2000);
                finalPrompt = prefix + prompt;
            }}

            try {{
                const response = await fetch('/v1/compare', {{
                    method: 'POST',
                    headers: {{ 'Content-Type': 'application/json' }},
                    body: JSON.stringify({{ prompt: finalPrompt, max_tokens, use_prefix_caching: prependPrefix }})
                }});

                const reader = response.body.getReader();
                const decoder = new TextDecoder();

                while (true) {{
                    const {{ done, value }} = await reader.read();
                    if (done) break;

                    const chunk = decoder.decode(value);
                    const lines = chunk.split('\\n');

                    for (const line of lines) {{
                        if (line.startsWith('data: ')) {{
                            const data = line.slice(6);
                            if (data === '[DONE]') continue;

                            try {{
                                const json = JSON.parse(data);
                                const backend = json.backend;

                                if (json.final) {{
                                    const metrics = `
Tokens: ${{json.token_count}}
TTFT: ${{json.time_to_first_token_ms}}ms
Decode Speed: ${{json.decode_throughput}} tok/s
Avg Inter-token: ${{json.avg_inter_token_latency_ms}}ms
Total Time: ${{(json.total_time_ms / 1000).toFixed(2)}}s`;

                                    document.getElementById(`metrics_${{backend}}`).textContent = metrics;
                                    results[backend].finalMetrics = json;
                                }} else if (json.token) {{
                                    document.getElementById(`output_${{backend}}`).textContent += json.token;

                                    const countLabel = json.is_char_count ? 'Characters' : 'Tokens';
                                    const metrics = `
${{countLabel}}: ${{json.token_count}}
Time: ${{(json.elapsed_ms / 1000).toFixed(1)}}s
(computing final metrics...)`;

                                    document.getElementById(`metrics_${{backend}}`).textContent = metrics;
                                }}
                            }} catch (e) {{}}
                        }}
                    }}
                }}

                btn.disabled = false;
                showWinner();

            }} catch (error) {{
                alert('Error: ' + error.message);
                btn.disabled = false;
            }}
        }}

        const STATUS_EMOJI = {{ running: '🟢', starting: '🟡', down: '🔴' }};

        async function pollHealth() {{
            try {{
                const res = await fetch('/v1/health');
                const data = await res.json();
                for (const [backend, status] of Object.entries(data)) {{
                    const el = document.getElementById(`status_${{backend}}`);
                    if (el) el.textContent = STATUS_EMOJI[status] || '🟡';
                }}
            }} catch (e) {{}}
        }}

        pollHealth();
        setInterval(pollHealth, 10000);

        function showWinner() {{
            const t = results.transformers.finalMetrics;
            const s = results.sglang.finalMetrics;
            const v = results.vllm.finalMetrics;

            if (!t || !s || !v) return;

            const backends = [
                ['HF Transformers', t],
                ['SGLang', s],
                ['vLLM', v],
            ];

            const byTTFT = [...backends].sort((a, b) => a[1].time_to_first_token_ms - b[1].time_to_first_token_ms);
            const byThroughput = [...backends].sort((a, b) => b[1].decode_throughput - a[1].decode_throughput);

            const fmtTTFT = ([n, m]) => `${{n}}: ${{m.time_to_first_token_ms}}ms`;
            const fmtThroughput = ([n, m]) => `${{n}}: ${{m.decode_throughput}} tok/s`;

            const banner = document.getElementById('winner');
            banner.innerHTML = `
                <div><strong>TTFT</strong> : ${{byTTFT.map(fmtTTFT).join(' &lt; ')}}</div>
                <div style="margin-top:8px;"><strong>Throughput</strong>: ${{byThroughput.map(fmtThroughput).join(' &gt; ')}}</div>
            `;
            banner.style.display = 'block';
        }}
    </script>
</body>
</html>"""
