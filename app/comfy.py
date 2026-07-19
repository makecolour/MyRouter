"""ComfyUI integration: instance resolution (DB), workflow build, generate/poll."""

import asyncio
import json
import logging
import mimetypes
import time
import uuid
from typing import Any, Iterator, List, Optional, Tuple
from urllib.parse import urlencode

import httpx
from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .models import ComfyInstance
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.comfy")

_http: Optional[httpx.AsyncClient] = None


def init_http() -> None:
    global _http
    _http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))


async def close_http() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


async def probe_instance(base_url: str) -> bool:
    """Quick reachability check for the dashboard (tunnels come and go)."""
    if _http is None:
        return False
    try:
        response = await _http.get(f"{base_url.rstrip('/')}/queue", timeout=3.0)
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def resolve_comfy_instance(model: Optional[str]) -> ComfyInstance:
    """Map the request's `model` to a comfy_instances row (default: first enabled).

    Queried per request on purpose: the table is the runtime-modifiable list
    of tunneled ComfyUI sites, so additions/removals apply immediately.
    """
    async with SessionLocal() as session:
        if model:
            instance = (
                await session.execute(
                    select(ComfyInstance).where(
                        ComfyInstance.name == model,
                        ComfyInstance.enabled.is_(True),
                    )
                )
            ).scalar_one_or_none()
            if instance is None:
                names = (
                    (
                        await session.execute(
                            select(ComfyInstance.name)
                            .where(ComfyInstance.enabled.is_(True))
                            .order_by(ComfyInstance.name)
                        )
                    )
                    .scalars()
                    .all()
                )
                raise openai_error(
                    404,
                    f"ComfyUI instance '{model}' not found or disabled. "
                    f"Available instances: {', '.join(names) or '(none)'}.",
                    code="model_not_found",
                )
            return instance

        instance = (
            await session.execute(
                select(ComfyInstance)
                .where(ComfyInstance.enabled.is_(True))
                .order_by(ComfyInstance.name)
                .limit(1)
            )
        ).scalar_one_or_none()
        if instance is None:
            raise openai_error(
                503,
                "No ComfyUI instances configured (comfy_instances table is "
                "empty or all rows are disabled).",
                "server_error",
            )
        return instance


def build_comfy_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    *,
    negative_prompt: Optional[str] = None,
    checkpoint: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    denoise: Optional[float] = None,
) -> dict:
    """Built-in text2img workflow in ComfyUI API format (SD 1.5 / SDXL).

    Every knob falls back to settings/defaults when None; valid values per
    instance are discoverable via GET /v1/comfy/info.
    """
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps if steps is not None else settings.comfy_steps,
                "cfg": cfg if cfg is not None else settings.comfy_cfg,
                "sampler_name": sampler or "euler",
                "scheduler": scheduler or "normal",
                "denoise": denoise if denoise is not None else 1.0,
                "model": ["4", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint or settings.comfy_checkpoint},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["4", 1]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": (
                    negative_prompt
                    if negative_prompt is not None
                    else settings.comfy_negative_prompt
                ),
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["4", 2]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "sidecar"},
        },
    }


async def fetch_instance_info(base_url: str) -> dict:
    """Installed checkpoints/samplers/schedulers of one ComfyUI instance."""
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    try:
        ckpt_resp = await _http.get(f"{base}/object_info/CheckpointLoaderSimple")
        ckpt_resp.raise_for_status()
        ks_resp = await _http.get(f"{base}/object_info/KSampler")
        ks_resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"ComfyUI info request failed ({base}): {exc}", "api_error"
        )
    try:
        checkpoints = ckpt_resp.json()["CheckpointLoaderSimple"]["input"]["required"][
            "ckpt_name"
        ][0]
        ksampler_inputs = ks_resp.json()["KSampler"]["input"]["required"]
        samplers = ksampler_inputs["sampler_name"][0]
        schedulers = ksampler_inputs["scheduler"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise openai_error(
            502, f"Unexpected ComfyUI object_info shape ({base}): {exc}", "api_error"
        )
    return {
        "checkpoints": checkpoints,
        "samplers": samplers,
        "schedulers": schedulers,
    }


async def fetch_image_bytes(url: str) -> bytes:
    """Download a generated image from a ComfyUI /view URL (for b64/binary)."""
    assert _http is not None, "comfy http client not initialized (lifespan)"
    try:
        response = await _http.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"Failed to fetch the generated image: {exc}", "api_error"
        )
    return response.content


# --------------------------------------------------------------------------
# Importing input files (images / masks) into ComfyUI's input dir
# --------------------------------------------------------------------------


async def upload_image(
    base_url: str,
    data: bytes,
    filename: str,
    *,
    image_type: str = "input",
    subfolder: str = "",
    overwrite: bool = True,
) -> dict:
    """POST an image to ComfyUI's /upload/image → {name, subfolder, type}.

    `image_type` "temp" lands the file in the auto-cleaned temp dir (ephemeral,
    not the permanent input gallery); a LoadImage node references it via the
    annotated ref (see `annotated_ref`).
    """
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    form = {"type": image_type, "overwrite": "true" if overwrite else "false"}
    if subfolder:
        form["subfolder"] = subfolder
    try:
        resp = await _http.post(
            f"{base}/upload/image",
            files={"image": (filename, data, mime)},
            data=form,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"ComfyUI image upload failed ({base}): {exc}", "api_error"
        )
    return resp.json()


async def upload_mask(
    base_url: str,
    data: bytes,
    filename: str,
    original_ref: dict,
    *,
    image_type: str = "input",
    subfolder: str = "",
    overwrite: bool = True,
) -> dict:
    """POST a mask to /upload/mask — composited into `original_ref`'s alpha.

    `original_ref` is a previously-uploaded image's `{filename, subfolder,
    type}` (i.e. the /upload/image result renamed name→filename).
    """
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    mime = mimetypes.guess_type(filename)[0] or "image/png"
    form = {
        "type": image_type,
        "overwrite": "true" if overwrite else "false",
        "original_ref": json.dumps(original_ref),
    }
    if subfolder:
        form["subfolder"] = subfolder
    try:
        resp = await _http.post(
            f"{base}/upload/mask",
            files={"image": (filename, data, mime)},
            data=form,
        )
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"ComfyUI mask upload failed ({base}): {exc}", "api_error"
        )
    return resp.json()


def annotated_ref(result: dict) -> str:
    """Turn an /upload/* result into a LoadImage-ready string.

    Plain input-dir files use the bare name; temp/output files carry the
    `[type]` annotation ComfyUI's `get_annotated_filepath` understands.
    """
    name = result.get("name", "")
    subfolder = result.get("subfolder") or ""
    image_type = result.get("type") or "input"
    path = f"{subfolder}/{name}" if subfolder else name
    return path if image_type == "input" else f"{path} [{image_type}]"


def _iter_api_nodes(workflow: dict) -> Iterator[Tuple[str, str, dict]]:
    """Yield (node_id, class_type, inputs) for API-format workflows."""
    if not isinstance(workflow, dict):
        return
    for node_id, node in workflow.items():
        if isinstance(node, dict) and "class_type" in node:
            yield str(node_id), node["class_type"], node.get("inputs") or {}


async def analyze_upload_slots(base_url: str, workflow: dict) -> List[dict]:
    """Find every node input that takes an uploaded file.

    Uses the instance's own metadata: an input whose options dict carries a
    truthy `*_upload` flag (e.g. `image_upload` on LoadImage) is exactly what
    ComfyUI's frontend renders an upload button for. Generic — no hardcoded
    node list. API-format only (that's what /prompt runs and what we wire).
    """
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    try:
        resp = await _http.get(f"{base}/object_info")
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"ComfyUI object_info failed ({base}): {exc}", "api_error"
        )
    object_info = resp.json()
    if not isinstance(object_info, dict):
        return []
    slots: List[dict] = []
    for node_id, class_type, inputs in _iter_api_nodes(workflow):
        info = object_info.get(class_type)
        if not isinstance(info, dict):
            continue
        spec = info.get("input") or {}
        for section in ("required", "optional"):
            for input_name, val in (spec.get(section) or {}).items():
                opts = (
                    val[1]
                    if isinstance(val, list) and len(val) > 1 and isinstance(val[1], dict)
                    else {}
                )
                flag = next(
                    (k for k, v in opts.items() if k.endswith("_upload") and v), None
                )
                if flag:
                    slots.append(
                        {
                            "node_id": node_id,
                            "class_type": class_type,
                            "input_name": input_name,
                            "upload_kind": flag[: -len("_upload")],
                            "current_value": inputs.get(input_name),
                        }
                    )
    return slots


def make_ephemeral(workflow: dict) -> dict:
    """Rewrite SaveImage nodes → PreviewImage so outputs land in the temp dir
    (auto-cleaned, not in the permanent gallery). Returns a shallow-safe copy.
    """
    if not isinstance(workflow, dict):
        return workflow
    out = {}
    for key, node in workflow.items():
        if isinstance(node, dict) and node.get("class_type") == "SaveImage":
            node = dict(node)
            node["class_type"] = "PreviewImage"
            inputs = dict(node.get("inputs", {}))
            inputs.pop("filename_prefix", None)  # PreviewImage has no such input
            node["inputs"] = inputs
        out[key] = node
    return out


async def delete_history(base_url: str, prompt_id: str) -> None:
    """Drop a prompt from ComfyUI history (best-effort; never raises)."""
    if _http is None or not prompt_id:
        return
    try:
        await _http.post(
            f"{base_url.rstrip('/')}/history", json={"delete": [prompt_id]}
        )
    except httpx.HTTPError as exc:
        logger.warning("ComfyUI history delete failed (%s): %s", prompt_id, exc)


async def fetch_queue(base_url: str) -> dict:
    """Queue depth of one ComfyUI instance."""
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    try:
        response = await _http.get(f"{base}/queue")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise openai_error(
            502, f"ComfyUI queue request failed ({base}): {exc}", "api_error"
        )
    data = response.json()
    return {
        "running": len(data.get("queue_running", [])),
        "pending": len(data.get("queue_pending", [])),
    }


def apply_workflow_placeholders(
    workflow: dict, prompt: str, width: int, height: int, seed: int
) -> dict:
    """Substitute placeholders in a caller-supplied ComfyUI workflow.

    A string value that is exactly "{seed}", "{width}" or "{height}" becomes
    the typed integer; "{prompt}" / "{negative_prompt}" are replaced textually
    wherever they occur inside string values. A workflow without placeholders
    is sent as-is (the caller has full control).
    """
    typed = {"{seed}": seed, "{width}": width, "{height}": height}
    textual = {
        "{prompt}": prompt,
        "{negative_prompt}": settings.comfy_negative_prompt,
    }

    def substitute(value: Any) -> Any:
        if isinstance(value, str):
            if value in typed:
                return typed[value]
            for token, replacement in textual.items():
                if token in value:
                    value = value.replace(token, replacement)
            return value
        if isinstance(value, dict):
            return {key: substitute(item) for key, item in value.items()}
        if isinstance(value, list):
            return [substitute(item) for item in value]
        return value

    return substitute(workflow)


async def comfy_generate(
    base_url: str, workflow: dict, ephemeral: bool = False
) -> str:
    """Queue one generation on a ComfyUI instance, poll history, return the URL.

    When `ephemeral`, the history entry is deleted once the result is
    retrieved (the workflow should already be `make_ephemeral`-rewritten so the
    output itself lands in the temp dir).
    """
    assert _http is not None, "comfy http client not initialized (lifespan)"

    payload = {"prompt": workflow, "client_id": uuid.uuid4().hex}

    try:
        response = await _http.post(f"{base_url}/prompt", json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("ComfyUI /prompt request failed (%s): %s", base_url, exc)
        raise openai_error(
            502, f"ComfyUI /prompt request failed ({base_url}): {exc}", "api_error"
        )

    prompt_id = response.json().get("prompt_id")
    if not prompt_id:
        raise openai_error(
            502, f"ComfyUI did not return a prompt_id: {response.text}", "api_error"
        )
    logger.info("ComfyUI job queued (%s, prompt_id=%s)", base_url, prompt_id)

    deadline = time.monotonic() + settings.comfy_timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(settings.comfy_poll_interval)
        try:
            history_response = await _http.get(f"{base_url}/history/{prompt_id}")
            history_response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("ComfyUI history poll failed, retrying: %s", exc)
            continue

        entry = history_response.json().get(prompt_id)
        if not entry:
            continue  # still queued/executing

        status = entry.get("status") or {}
        if status.get("status_str") == "error":
            logger.error("ComfyUI workflow errored: %s", status)
            raise openai_error(
                502,
                f"ComfyUI workflow failed (prompt_id={prompt_id}). Check that "
                f"the referenced checkpoints/nodes exist on that instance.",
                "api_error",
            )

        for node_output in (entry.get("outputs") or {}).values():
            for image in node_output.get("images", []):
                query = urlencode(
                    {
                        "filename": image["filename"],
                        "subfolder": image.get("subfolder", ""),
                        "type": image.get("type", "output"),
                    }
                )
                url = f"{base_url}/view?{query}"
                logger.info(
                    "ComfyUI job complete (prompt_id=%s): %s", prompt_id, url
                )
                if ephemeral:
                    await delete_history(base_url, prompt_id)
                return url

    raise openai_error(
        504,
        f"ComfyUI generation timed out after {settings.comfy_timeout:.0f}s "
        f"(prompt_id={prompt_id}).",
        "api_error",
        "timeout",
    )
