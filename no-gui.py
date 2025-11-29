import os
import datetime
import argparse
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

# 2. Parse Command Line Arguments
parser = argparse.ArgumentParser(description="Generate images with FLUX.1")
parser.add_argument(
    "--prompt", 
    type=str, 
    default="A cat holding a sign that says hello world",
    help="The text prompt to generate an image for."
)
args = parser.parse_args()

# Check for CUDA
if not torch.cuda.is_available():
    print("Warning: CUDA is not available. Running on CPU will be extremely slow.")

# 3. Load Pipeline
print(f"Loading FLUX.1-dev... (Cache: download/)")
pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", 
    torch_dtype=torch.bfloat16
)

# Enable CUDA optimizations
pipe.enable_model_cpu_offload()

# 4. Generate Image
print(f"Generating image for prompt: '{args.prompt}'")

image = pipe(
    args.prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]

# 5. Save to 'results' folder
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)  # Creates the folder if it doesn't exist

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"flux_gen_{timestamp}.png"

# Combine folder and filename safely (works on Windows and Linux)
file_path = os.path.join(output_dir, filename)

image.save(file_path)
print(f"Success! Image saved to: {file_path}")