from __future__ import annotations

import logging
from http import HTTPStatus
from typing import Any, Dict, Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import BadRequest

from src.inference import ControlNetConfig, FluxGenerator, pil_to_base64

logger = logging.getLogger(__name__)
flux_generator = FluxGenerator()


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health() -> Any:
        return {"status": "ok"}

    @app.get("/")
    def index() -> Any:
        """Serve the academic-styled playground UI."""
        presets = sorted(FluxGenerator.CONTROLNET_MODELS.keys())
        return render_template(
            "index.html",
            controlnet_presets=presets,
            flux_version=FluxGenerator.MODEL_ID,
        )

    @app.post("/api/generate")
    def generate() -> Any:
        payload = request.get_json(force=True, silent=True)
        if not payload or "prompt" not in payload:
            raise BadRequest("'prompt' is required in the request payload")

        controlnet_cfg: Optional[ControlNetConfig] = None
        controlnet_payload = payload.get("controlnet")
        if controlnet_payload:
            try:
                controlnet_cfg = ControlNetConfig(
                    preset=controlnet_payload["preset"],
                    image_b64=controlnet_payload["image"],
                )
            except KeyError as exc:
                raise BadRequest("ControlNet preset and image must be provided") from exc

        try:
            image = flux_generator.generate(
                prompt=payload["prompt"],
                negative_prompt=payload.get("negative_prompt"),
                width=int(payload.get("width", 512)),
                height=int(payload.get("height", 512)),
                num_inference_steps=int(payload.get("num_inference_steps", 28)),
                guidance_scale=float(payload.get("guidance_scale", 3.5)),
                controlnet=controlnet_cfg,
                seed=payload.get("seed"),
            )
        except ValueError as exc:
            logger.exception("Invalid request for generation")
            return jsonify({"error": str(exc)}), HTTPStatus.BAD_REQUEST
        except Exception as exc:  # pragma: no cover - best effort error handling
            logger.exception("Failed to generate image")
            return jsonify({"error": str(exc)}), HTTPStatus.INTERNAL_SERVER_ERROR

        image_b64 = pil_to_base64(image)
        return jsonify(
            {
                "image": f"data:image/png;base64,{image_b64}",
                "metadata": {
                    "width": image.width,
                    "height": image.height,
                    "controlnet": controlnet_cfg.preset if controlnet_cfg else None,
                },
            }
        )

    return app


app = create_app()
