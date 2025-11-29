# Operations & Deployment

This guide focuses on running the inference service reliably, managing the
large model assets, and customizing the runtime environment.

## Environment Setup

1. Install Python 3.10+ and create a virtual environment.
2. Install dependencies with `pip install -r requirements.txt`.
3. Export one of `HUGGINGFACEHUB_API_TOKEN`, `HF_TOKEN`, or
   `HUGGINGFACE_TOKEN` with a Hugging Face user token that can access the
   FLUX.1-dev models (and ControlNet checkpoints if used).

The first import of `src.inference` automatically configures the following
environment variables to live under `./download/`:

- `HF_HOME`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `DIFFUSERS_CACHE`
- `TORCH_HOME`

This keeps the repository tidy and ensures repeated runs reuse the same cached
weights. Mount `download/` on persistent storage if you deploy inside ephemeral
containers.

## Running the Server

```bash
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

- `flux_generator` is created at module import, so keep the worker process warm
  to avoid reloading the pipelines.
- Behind a process manager (Gunicorn, systemd, etc.), configure a single worker
  per GPU. Additional concurrency should be handled at the queueing layer
  because generation is blocking.

## Hardware Considerations

- CUDA with bfloat16 provides the best throughput. The code gracefully falls
  back to CPU or Apple `mps`, albeit with longer latency.
- If CUDA runs out of memory when loading the model, the code catches the
  `torch.cuda.OutOfMemoryError` and automatically enables sequential CPU
  offload—no manual intervention necessary.
- ControlNet doubles VRAM usage because the ControlNet weights load alongside
  the base FLUX pipeline. Monitor GPU memory if you plan to expose many
  presets.

## Customizing ControlNet Presets

1. Edit `FluxGenerator.CONTROLNET_MODELS` in `src/inference.py` and add a new
   key/value pair where the key is the preset name and the value is the
   Hugging Face repo ID (e.g., `"depth": "black-forest-labs/..."`).
2. Restart the server. The API starts accepting the new preset immediately.
3. The first request downloads and caches the new checkpoint. Subsequent runs
   reuse `download/hf-cache`.

## Troubleshooting

- **Missing prompt / invalid payload**: The Flask handler raises a `BadRequest`
  before reaching the pipeline. Validate the JSON on the client side.
- **Model download failures**: Confirm the Hugging Face token is exported and
  the machine has outbound network access the first time the models load.
- **Slow generation**: Lower `num_inference_steps`, reduce image resolution, or
  move to GPU hardware. CPU-only execution is functional but not practical for
  production traffic.
- **Corrupted cache**: Delete `download/hf-cache` and `download/torch-cache`.
  They will be recreated on the next run.
