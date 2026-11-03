Demo page: prompt box, checkpoint slider, riddle button, benchmark table.

Static shell in this folder (index.html, app.js, style.css) reads `manifest.json`. Training exports append checkpoints `{step, tokens_seen, url}` and benchmark rows; `tests/test_demo_manifest.py` checks the schema. Inference (ONNX/WebGPU or other free hosting) hooks into app.js once checkpoints exist. Gemini may take over styling/benchmarks if he picks the demo lane.
