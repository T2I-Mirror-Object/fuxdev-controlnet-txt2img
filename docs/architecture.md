# Architecture Notes

This service wraps the `black-forest-labs/FLUX.1-dev` Diffusers pipelines in a
small Flask application. The code favors lazily loading heavy pipelines and
sharing a single generator instance across requests so that most invocations are
gated by inference time rather than setup cost.

## High-Level Layout

```
Client --> Flask (app.py) --> FluxGenerator (src/inference.py) --> Diffusers
   ^             |                    |                           |
   |             |                    |                           +-- Hugging Face caches (download/)
   |             |                    +-- Optional ControlNet pipe
   |             +-- Base64 encoding +-- Torch Generator seed control
   +-- HTML Playground (templates/index.html)
```

- `app.py` wires the `/` UI, `/health`, and `/api/generate` endpoints.
- `FluxGenerator` orchestrates the text-to-image and ControlNet pipelines,
  handling cache configuration, device selection, and RNG management.
- `templates/index.html` offers a lightweight playground that submits requests
  to `/api/generate`.
- All model downloads are redirected under `download/` (see `_configure_cache_env`).

## Request Flow

1. `POST /api/generate` validates the JSON payload and parses the optional
   ControlNet object (`ControlNetConfig`).
2. `FluxGenerator.generate` prepares a Torch `Generator`, seeds it when asked,
   and builds the keyword arguments shared by both the text-to-image and
   ControlNet flows.
3. When ControlNet is present:
   - `_decode_base64_image` sanitizes the string, optionally trimming data URLs.
   - The control image is resized to the requested output resolution.
   - `_load_controlnet_pipe` retrieves or instantiates a
     `FluxControlNetPipeline`, caching the result per preset.
4. Without ControlNet, `_load_text2img_pipe` returns the shared
   `DiffusionPipeline`.
5. The selected pipeline produces an image which is encoded to base64 and
   returned to the caller together with light metadata.

## Model and Cache Management

- `_configure_cache_env` ensures the Hugging Face and Torch cache environment
  variables point to `download/hf-cache` and `download/torch-cache`.
- Pipelines are lazily constructed and stored on the `FluxGenerator` instance,
  so every subsequent request avoids model re-loading.
- Device selection uses CUDA when available, falls back to CPU, and applies an
  `mps` override on Apple hardware. If CUDA runs out of memory, the code
  automatically reverts the pipeline to CPU with sequential offload enabled.

## Extending ControlNet Support

Add new entries to `FluxGenerator.CONTROLNET_MODELS` (in `src/inference.py`) with
the preset name and the corresponding ControlNet checkpoint. The API will start
accepting the new preset immediately and the model will download on first use.
