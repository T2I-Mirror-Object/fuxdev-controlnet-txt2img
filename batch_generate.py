import os
import csv
import torch
from diffusers import FluxPipeline
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[0]
DOWNLOAD_ROOT = PROJECT_ROOT / "download"
HF_CACHE = DOWNLOAD_ROOT / "hf-cache"
TORCH_CACHE = DOWNLOAD_ROOT / "torch-cache"


def _ensure_cache_dirs() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    TORCH_CACHE.mkdir(parents=True, exist_ok=True)


def _configure_cache_env() -> None:
    _ensure_cache_dirs()
    env_map = {
        "HF_HOME": HF_CACHE,
        "HUGGINGFACE_HUB_CACHE": HF_CACHE,
        "TRANSFORMERS_CACHE": HF_CACHE,
        "DIFFUSERS_CACHE": HF_CACHE,
        "TORCH_HOME": TORCH_CACHE,
    }
    for key, value in env_map.items():
        os.environ.setdefault(key, str(value))


_configure_cache_env()


def load_prompts_from_csv(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

# # Check for CUDA
# if not torch.cuda.is_available():
#     print("Warning: CUDA is not available. Running on CPU will be extremely slow.")

# Load prompts from CSV
print("Loading prompts from: prompts.csv")
prompts = load_prompts_from_csv("prompts.csv")
print(f"Loaded {len(prompts)} prompts\n")

# Load Pipeline
print(f"Loading FLUX.1-dev... (Cache: download/)")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    torch_dtype=torch.bfloat16
)

# Enable CUDA optimizations (manages ~50GB memory requirement)
pipe.enable_model_cpu_offload()

# Generate Images
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

print(f"\n{'='*60}")
print(f"Starting generation for {len(prompts)} prompts")
print(f"{'='*60}\n")

for idx, prompt in enumerate(prompts[1:], 1):
    print(f"[{idx}/{len(prompts)}] Generating: '{prompt[:60]}{'...' if len(prompt) > 60 else ''}'")

    try:
        # Generate one image per prompt
        image = pipe(
            prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=30,
            max_sequence_length=512,
            generator=torch.Generator("cpu").manual_seed(idx)
        ).images[0]

        # Save with index-based filename
        filename = f"image_{idx:03d}.png"
        file_path = os.path.join(output_dir, filename)
        image.save(file_path)

        print(f"  ✓ Saved: {filename}\n")

    except Exception as e:
        print(f"  ✗ Error: {str(e)}\n")
        continue

print(f"{'='*60}")
print(f"✓ Complete! {len(prompts)} images saved to: {output_dir}/")
print(f"{'='*60}")