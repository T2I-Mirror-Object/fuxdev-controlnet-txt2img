# HTTP API Reference

The Flask server exposes a single JSON API for text-to-image generation plus a
couple of utility endpoints. All responses use UTF-8 encoded JSON.

## Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Lightweight readiness probe. Returns `{"status": "ok"}`. |
| `GET` | `/` | Renders the academic-style playground UI. Useful for manual testing. |
| `POST` | `/api/generate` | Generates an image from the supplied prompt and optional ControlNet config. |

`/api/generate` accepts the following payload:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `prompt` | string | ✅ | Main positive prompt passed to the Diffusers pipeline. |
| `negative_prompt` | string | ❌ | Negative guidance prompt. |
| `width` | int | ❌ | Output width in pixels (default `512`). |
| `height` | int | ❌ | Output height in pixels (default `512`). |
| `num_inference_steps` | int | ❌ | Scheduler steps (default `28`). |
| `guidance_scale` | float | ❌ | CFG scale (default `3.5`). |
| `seed` | int | ❌ | RNG seed. Provide a number to get deterministic results. |
| `controlnet` | object | ❌ | Enables ControlNet conditioning—see below. |

The `controlnet` object must include:

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `preset` | string | ✅ | One of `canny` or `depth` by default. Extend `CONTROLNET_MODELS` to add more. |
| `image` | string | ✅ | Base64 string or data URL of the conditioning image. |

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
  }'
```

### Responses

`200 OK`
```json
{
  "image": "data:image/png;base64,<...>",
  "metadata": {
    "width": 768,
    "height": 1024,
    "controlnet": null
  }
}
```

`400 Bad Request`
- Missing prompt, malformed ControlNet payload, or unsupported preset.
- Example: `{"error": "'prompt' is required in the request payload"}`

`500 Internal Server Error`
- Any unhandled exception during generation. The JSON response contains an
  `error` field with the message captured server-side.

### Decoding the Image

The `image` field contains a data URL string. Strip the prefix and decode the
remainder to save the PNG:

```bash
jq -r '.image' response.json | sed 's/^data:image\\/png;base64,//' | base64 --decode > output.png
```
