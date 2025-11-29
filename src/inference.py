"""Utilities for running FLUX.1-dev inference with optional ControlNet."""
from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from diffusers import ControlNetModel, DiffusionPipeline, FluxControlNetPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
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


@dataclass
class ControlNetConfig:
    """Configuration provided by the API client for ControlNet guidance."""

    preset: str
    image_b64: str


class FluxGenerator:
    """Lazy loader around FLUX.1-dev pipelines."""

    MODEL_ID = "black-forest-labs/FLUX.1-dev"
    CONTROLNET_MODELS: Dict[str, str] = {
        "canny": "black-forest-labs/FLUX.1-dev-Controlnet-Canny",
        "depth": "black-forest-labs/FLUX.1-dev-Controlnet-Depth",
    }

    def __init__(self, device: Optional[str] = None) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        if self.device == "mps":  # macOS GPU fallback
            self.torch_dtype = torch.float16
        self._text2img_pipe: Optional[DiffusionPipeline] = None
        self._control_pipes: Dict[str, FluxControlNetPipeline] = {}

    def _prepare_pipeline(self, pipe: DiffusionPipeline) -> DiffusionPipeline:
        """Move a pipeline to the requested device with graceful CUDA fallback."""

        pipe.set_progress_bar_config(disable=True)
        if self.device == "cuda" and torch.cuda.is_available():
            try:
                pipe.to(self.device)
            except torch.cuda.OutOfMemoryError:
                pipe.to("cpu")
                pipe.enable_sequential_cpu_offload()
        else:
            pipe.to(self.device)
        return pipe

    def _load_text2img_pipe(self) -> DiffusionPipeline:
        if self._text2img_pipe is None:
            pipe = DiffusionPipeline.from_pretrained(
                self.MODEL_ID,
                torch_dtype=self.torch_dtype,
                cache_dir=str(HF_CACHE),
            )
            self._text2img_pipe = self._prepare_pipeline(pipe)
        return self._text2img_pipe

    def _load_controlnet_pipe(self, preset: str) -> FluxControlNetPipeline:
        if preset not in self.CONTROLNET_MODELS:
            raise ValueError(
                f"Unsupported ControlNet preset '{preset}'. Choose from {sorted(self.CONTROLNET_MODELS)}."
            )
        if preset not in self._control_pipes:
            controlnet = ControlNetModel.from_pretrained(
                self.CONTROLNET_MODELS[preset],
                torch_dtype=self.torch_dtype,
                cache_dir=str(HF_CACHE),
            )
            pipe = FluxControlNetPipeline.from_pretrained(
                self.MODEL_ID,
                controlnet=controlnet,
                torch_dtype=self.torch_dtype,
                cache_dir=str(HF_CACHE),
            )
            self._control_pipes[preset] = self._prepare_pipeline(pipe)
        return self._control_pipes[preset]

    def generate(
        self,
        *,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        controlnet: Optional[ControlNetConfig] = None,
        seed: Optional[int] = None,
    ) -> Image.Image:
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)

        common_kwargs = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": width,
            "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "generator": generator,
        }

        if controlnet:
            control_image = _decode_base64_image(controlnet.image_b64).resize((width, height))
            pipe = self._load_controlnet_pipe(controlnet.preset)
            images = pipe(control_image=control_image, **common_kwargs).images
        else:
            pipe = self._load_text2img_pipe()
            images = pipe(**common_kwargs).images
        return images[0]


def _decode_base64_image(data: str) -> Image.Image:
    if "," in data and data.strip().lower().startswith("data:"):
        data = data.split(",", maxsplit=1)[1]
    try:
        binary = base64.b64decode(data)
    except Exception as exc:  # pragma: no cover - defensive branch
        raise ValueError("Invalid base64 image supplied for ControlNet input") from exc
    image = Image.open(io.BytesIO(binary)).convert("RGB")
    return image


def pil_to_base64(image: Image.Image, format: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")
