# Usage Guide

This document explains how to run the Flask inference server that wraps the
`black-forest-labs/FLUX.1-dev` Diffusers pipeline and how to exercise the HTTP
API for text-to-image generation with optional ControlNet conditioning. For a
deeper dive into the architecture and operational tips, see `docs/architecture.md`
and `docs/operations.md`.

## Prerequisites
- Python 3.10 or newer
- Access to the FLUX.1-dev and optional ControlNet checkpoints on Hugging Face.
  Export `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_TOKEN` with a
  user token that can download the models.
- A GPU with bfloat16 support is strongly recommended. CPU or Apple `mps`
  execution is possible but slow.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The application automatically redirects Hugging Face and Torch caches to
`./download/` so the repository stays tidy. The folder structure is created on
demand the first time you import `src.inference`.

## Running the Server
```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

The server exposes:
- `GET /health` – basic readiness probe that returns `{"status":"ok"}`.
- `POST /api/generate` – generates an image from a text prompt.

## Request Payload

| Field | Type | Required | Description |
| --- | --- | --- | --- |
| `prompt` | string | ✅ | Positive text prompt. |
| `negative_prompt` | string | ❌ | Negative guidance prompt. |
| `width` | int | ❌ | Output width (default `512`). |
| `height` | int | ❌ | Output height (default `512`). |
| `num_inference_steps` | int | ❌ | Sampling steps (default `28`). |
| `guidance_scale` | float | ❌ | CFG scale (default `3.5`). |
| `seed` | int | ❌ | RNG seed for reproducible results. |
| `controlnet` | object | ❌ | ControlNet configuration (see below). |

The `controlnet` object must contain:
- `preset`: one of `canny` or `depth` (add more in `CONTROLNET_MODELS`).
- `image`: base64 string or data URL (`data:image/png;base64,...`) of the
  conditioning image. The server decodes the string and resizes it to the output
  resolution before feeding it into the pipeline.

### Example Request
```bash
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A macro photo of blue crystals on a black backdrop",
    "width": 768,
    "height": 1024,
    "num_inference_steps": 30,
    "guidance_scale": 3.0,
    "seed": 123
  }' | jq -r '.image' > output.base64
```

To include ControlNet guidance, base64-encode the conditioning image:
```bash
CONTROL_IMAGE=$(base64 -i control.png)
curl -X POST http://localhost:5000/api/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"prompt\": \"Cyberpunk skyline at night\",
    \"controlnet\": {
      \"preset\": \"canny\",
      \"image\": \"$CONTROL_IMAGE\"
    }
  }"
```

## Response Format
```json
{
  "image": "data:image/png;base64,<...>",
  "metadata": {
    "width": 512,
    "height": 512,
    "controlnet": "canny"
  }
}
```

Strip the `data:image/png;base64,` prefix and decode the remainder to obtain a
PNG file:
```bash
base64 --decode output.base64 > output.png
```

## Tips
- The server blocks while generating an image. Wrap it in a job queue if you
  need concurrent workloads.
- `FluxGenerator` automatically detects CUDA, Metal (`mps`), or CPU; override
  the device by passing `FluxGenerator(device="cuda")` if embedding elsewhere.
- Ensure ControlNet images share the same aspect ratio as the requested output
  to minimize artifacts (the server resizes but does not crop).
