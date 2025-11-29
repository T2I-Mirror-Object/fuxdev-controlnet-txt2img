# FLUX.1-dev Flask Inference Server

Small Flask service that runs [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) text-to-image inference with an optional ControlNet conditioning step. All model assets and auxiliary downloads are stored under `./download/` to keep the workspace tidy.

## Prerequisites
- Python 3.10+
- Access to the FLUX.1-dev (and ControlNet) checkpoints on Hugging Face. Export `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, or `HUGGINGFACE_TOKEN` with a valid user token so diffusers can download the weights.
- A modern GPU with bfloat16 support is recommended; CPU execution is possible but extremely slow.

## Environment Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
The application automatically sets `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `DIFFUSERS_CACHE`, and `TORCH_HOME` so every download lands below `./download/`.

## Running the server
```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```
The health endpoint is available at `GET /health`.

## Generate an image
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
The response contains a `data:image/png;base64,...` string. Strip the prefix and decode it with `base64 --decode` to obtain the PNG file.

## ControlNet guidance
Provide a preset and a base64-encoded control image (data URLs are accepted):
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
Supported presets are `canny` and `depth`. Add more by extending `CONTROLNET_MODELS` in `src/inference.py`.

## Project structure
- `app.py` – Flask entrypoint exposing `/api/generate`.
- `src/inference.py` – Lazy-loading wrapper around Diffusers pipelines with download management and helper utilities.
- `download/` – Cache directory for Hugging Face models and Torch assets.

## Notes
- Generation is blocking; consider adding a task queue for production workloads.
- If you use custom ControlNet condition images, ensure they roughly match the requested output dimensions for best results (the server will resize them to the requested resolution).
