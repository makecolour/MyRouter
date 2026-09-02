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

# base_url -> (architecture dict, monotonic expiry). Probing /object_info is a
# remote round trip, but instances are tunnels that come and go and models can
# be installed at runtime, so the answer must expire — and it is dropped
# outright whenever /prompt rejects a graph (see comfy_generate).
_arch_cache: "dict[str, Tuple[dict, float]]" = {}
_ARCH_TTL = 300.0


def init_http() -> None:
    global _http
    _http = httpx.AsyncClient(timeout=httpx.Timeout(60.0))


async def close_http() -> None:
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None
    _arch_cache.clear()


def _describe_comfy_error(response: httpx.Response) -> str:
    """Flatten a ComfyUI rejection body into one diagnostic line.

    A 400 from /prompt is graph VALIDATION, and the body is the only place
    that says which node failed and what the valid values were — httpx's
    HTTPStatusError string is just the status line plus an MDN link. The
    shape is::

        {"error": {"message": ..., "details": ...},
         "node_errors": {"4": {"class_type": ..., "errors": [
             {"message": ..., "details": "ckpt_name: 'x' not in []"}]}}}

    Nothing about that is versioned, so every level falls back to the raw
    body rather than assuming.
    """
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return (response.text or "").strip()[:500] or f"HTTP {response.status_code}"
    if not isinstance(body, dict):
        return str(body)[:500]

    parts: List[str] = []
    error = body.get("error")
    if isinstance(error, dict):
        headline = error.get("message") or error.get("type")
        details = error.get("details")
        if headline:
            parts.append(f"{headline}{f' ({details})' if details else ''}")
    elif error:
        parts.append(str(error))

    node_errors = body.get("node_errors")
    if isinstance(node_errors, dict):
        for node_id, node in node_errors.items():
            if not isinstance(node, dict):
                parts.append(f"node {node_id}: {node}")
                continue
            class_type = node.get("class_type") or "?"
            for err in node.get("errors") or []:
                if isinstance(err, dict):
                    detail = err.get("details") or err.get("message") or ""
                else:
                    detail = str(err)
                parts.append(f"node {node_id} ({class_type}): {detail}")

    return "; ".join(p for p in parts if p) or (response.text or "")[:500]


def _unreachable_error(base_url: str, exc: Exception):
    """503 for 'the box/tunnel is down', as opposed to 'the graph is wrong'.

    Cloudflare answers for a dead tunnel with its own 502/503/504, which is
    indistinguishable from a ComfyUI error unless we say so explicitly.
    """
    return openai_error(
        503,
        f"ComfyUI instance is unreachable ({base_url}) — the tunnel or the "
        f"machine behind it looks down: {exc}",
        "api_error",
        "instance_unreachable",
    )


def _is_gateway_status(status: int) -> bool:
    """502/503/504 here means the tunnel answered, not ComfyUI."""
    return status in (502, 503, 504)


def _check_comfy_response(response: httpx.Response, base: str, what: str) -> None:
    """Raise a *useful* error for a non-2xx ComfyUI response.

    Replaces bare raise_for_status() at every site a caller's input can reach,
    so the failure says what ComfyUI actually complained about instead of
    "Client error '400 Bad Request'".
    """
    if response.status_code < 400:
        return
    if _is_gateway_status(response.status_code):
        raise _unreachable_error(base, f"HTTP {response.status_code} from the tunnel")
    raise openai_error(
        502,
        f"ComfyUI {what} failed ({base}): {_describe_comfy_error(response)}",
        "api_error",
    )


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


def _pick_clip(clips: List[str], want: str, fallback_index: int) -> str:
    """Choose one of the Flux text encoders by name, tolerating renames.

    A Flux install pairs a CLIP-L with a T5-XXL; the files are conventionally
    named but not guaranteed to be, so match on the distinguishing substring
    and fall back to position rather than hardcoding a filename.
    """
    for name in clips:
        if want in name.lower():
            return name
    if clips:
        return clips[min(fallback_index, len(clips) - 1)]
    return ""


def build_flux_workflow(
    prompt: str,
    width: int,
    height: int,
    seed: int,
    arch: dict,
    *,
    negative_prompt: Optional[str] = None,
    checkpoint: Optional[str] = None,
    steps: Optional[int] = None,
    cfg: Optional[float] = None,
    sampler: Optional[str] = None,
    scheduler: Optional[str] = None,
    denoise: Optional[float] = None,
    guidance: Optional[float] = None,
) -> dict:
    """Built-in text2img workflow for a Flux instance, in ComfyUI API format.

    Mirrors build_comfy_workflow's signature so the caller can swap on the
    detected architecture. Structural differences from the SD graph, all
    required rather than stylistic:

    * UNETLoader + DualCLIPLoader(type="flux") + VAELoader replace the single
      CheckpointLoaderSimple — a Flux install has no all-in-one checkpoint.
    * EmptySD3LatentImage, not EmptyLatentImage: Flux latents are 16-channel.
    * KSampler runs at CFG 1.0 and the real guidance rides on a FluxGuidance
      node; a normal CFG would burn the image.
    * The negative branch is ConditioningZeroOut — Flux dev has no true
      negative prompt, so `negative_prompt` is accepted and ignored here.
    """
    unets = arch.get("unets") or []
    clips = arch.get("clips") or []
    vaes = [v for v in (arch.get("vaes") or []) if v.endswith((".safetensors", ".pt"))]

    unet = checkpoint or (unets[0] if unets else settings.comfy_checkpoint)
    clip_l = _pick_clip(clips, "clip_l", 0)
    clip_t5 = _pick_clip(clips, "t5", 1)
    vae = vaes[0] if vaes else "ae.safetensors"

    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": steps if steps is not None else settings.comfy_flux_steps,
                # Flux dev is a guidance-distilled model: sampling CFG stays at
                # 1.0 and node "10" carries the strength instead.
                "cfg": cfg if cfg is not None else 1.0,
                "sampler_name": sampler or settings.comfy_flux_sampler,
                "scheduler": scheduler or settings.comfy_flux_scheduler,
                "denoise": denoise if denoise is not None else 1.0,
                "model": ["4", 0],
                "positive": ["10", 0],
                "negative": ["11", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "UNETLoader",
            # "default" because the shipped Flux weights are already fp8-scaled.
            "inputs": {"unet_name": unet, "weight_dtype": "default"},
        },
        "5": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": 1},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["12", 0]},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["3", 0], "vae": ["13", 0]},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"images": ["8", 0], "filename_prefix": "sidecar"},
        },
        "10": {
            "class_type": "FluxGuidance",
            "inputs": {
                "conditioning": ["6", 0],
                "guidance": (
                    guidance
                    if guidance is not None
                    else settings.comfy_flux_guidance
                ),
            },
        },
        "11": {
            # Flux dev ignores a negative prompt; zeroing the positive
            # conditioning is the sanctioned way to feed KSampler's negative.
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["6", 0]},
        },
        "12": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l,
                "clip_name2": clip_t5,
                "type": "flux",
            },
        },
        "13": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae},
        },
    }


def _node_options(payload: Any, node: str, field: str) -> List[str]:
    """The list of accepted values for one node input, or [] when absent.

    ComfyUI encodes a combo input as ``[[...choices...], {...metadata...}]``,
    so an installed-model list is always ``["input"]["required"][field][0]``.
    A node that isn't installed at all just yields [].
    """
    try:
        spec = payload[node]["input"]["required"][field]
    except (KeyError, IndexError, TypeError):
        return []
    if isinstance(spec, list) and spec and isinstance(spec[0], list):
        return [v for v in spec[0] if isinstance(v, str)]
    return []


async def _object_info(base: str, node: str) -> dict:
    """GET /object_info/<node>, tolerating a node that isn't installed.

    Fetches one node rather than the whole catalogue — /object_info in full is
    over 2 MB on a instance with custom nodes.
    """
    assert _http is not None, "comfy http client not initialized (lifespan)"
    try:
        response = await _http.get(f"{base}/object_info/{node}")
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    if _is_gateway_status(response.status_code):
        raise _unreachable_error(base, f"HTTP {response.status_code} from the tunnel")
    if response.status_code == 404:
        return {}  # node not installed on this instance
    if response.status_code >= 400:
        raise openai_error(
            502,
            f"ComfyUI object_info failed ({base}): {_describe_comfy_error(response)}",
            "api_error",
        )
    try:
        body = response.json()
    except (ValueError, json.JSONDecodeError):
        return {}
    return body if isinstance(body, dict) else {}


async def detect_architecture(base_url: str) -> dict:
    """Which graph family this instance can actually run.

    Returns ``{"architecture": "sd"|"flux", ...discovered model names}``.

    The built-in text2img graph starts with CheckpointLoaderSimple, which only
    works on instances holding all-in-one checkpoints. A Flux box has none of
    those — it has a UNET plus separate CLIP and VAE files — so sending the SD
    graph there is rejected at validation with an empty-list `value_not_in_list`
    and no choice of `checkpoint` can ever succeed. Detect it instead of
    guessing, so an instance works without per-instance configuration.
    """
    base = base_url.rstrip("/")
    cached = _arch_cache.get(base)
    if cached and cached[1] > time.monotonic():
        return cached[0]

    ckpt_info = await _object_info(base, "CheckpointLoaderSimple")
    checkpoints = _node_options(ckpt_info, "CheckpointLoaderSimple", "ckpt_name")

    unet_info = await _object_info(base, "UNETLoader")
    unets = _node_options(unet_info, "UNETLoader", "unet_name")

    if checkpoints:
        result = {"architecture": "sd", "checkpoints": checkpoints, "unets": unets}
    elif unets:
        clip_info = await _object_info(base, "DualCLIPLoader")
        vae_info = await _object_info(base, "VAELoader")
        result = {
            "architecture": "flux",
            "checkpoints": [],
            "unets": unets,
            "clips": _node_options(clip_info, "DualCLIPLoader", "clip_name1"),
            "vaes": _node_options(vae_info, "VAELoader", "vae_name"),
        }
    else:
        # Neither loader has a file: the instance is running but has no models.
        # Say so here rather than letting the graph fail with a cryptic 400.
        raise openai_error(
            502,
            f"ComfyUI instance ({base}) has no models installed — neither "
            f"checkpoints (CheckpointLoaderSimple) nor UNETs (UNETLoader) "
            f"offer any files, so no workflow can run.",
            "api_error",
            "no_models_installed",
        )

    _arch_cache[base] = (result, time.monotonic() + _ARCH_TTL)
    logger.info(
        "ComfyUI instance %s detected as '%s'", base, result["architecture"]
    )
    return result


async def fetch_instance_info(base_url: str) -> dict:
    """Installed checkpoints/samplers/schedulers of one ComfyUI instance."""
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    try:
        ckpt_resp = await _http.get(f"{base}/object_info/CheckpointLoaderSimple")
        ks_resp = await _http.get(f"{base}/object_info/KSampler")
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    _check_comfy_response(ckpt_resp, base, "info request")
    _check_comfy_response(ks_resp, base, "info request")
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
    info = {
        "checkpoints": checkpoints,
        "samplers": samplers,
        "schedulers": schedulers,
    }
    # An empty `checkpoints` list is not "no models" — it is the signature of a
    # Flux box, and saying which family this instance runs (plus the files it
    # does have) is what makes that legible instead of looking like a fault.
    try:
        arch = await detect_architecture(base)
    except Exception:  # info is a diagnostic endpoint: never fail on extras
        logger.warning("Architecture detection failed for %s", base, exc_info=True)
        return info
    info["architecture"] = arch["architecture"]
    info["unet_models"] = arch.get("unets", [])
    info["clip_models"] = arch.get("clips", [])
    info["vae_models"] = arch.get("vaes", [])
    return info


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
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    _check_comfy_response(resp, base, "image upload")
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
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    _check_comfy_response(resp, base, "mask upload")
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


API_FORMAT_HINT = (
    "This looks like a ComfyUI UI export (it has 'nodes'/'links'). The API "
    'needs the API-format graph ({"<id>": {"class_type", "inputs"}}). In '
    "ComfyUI: open Settings, enable 'Dev mode', then use 'Save (API Format)' "
    "and paste that JSON."
)


def is_ui_workflow(workflow: Any) -> bool:
    """True for a ComfyUI UI/full-graph export (top-level 'nodes' list)."""
    return isinstance(workflow, dict) and isinstance(workflow.get("nodes"), list)


def is_api_workflow(workflow: Any) -> bool:
    """True when at least one value is an API-format node ({class_type, …})."""
    return isinstance(workflow, dict) and any(
        isinstance(v, dict) and "class_type" in v for v in workflow.values()
    )


def require_api_workflow(workflow: Any) -> None:
    """Raise a clear 400 when a UI export is supplied where the API graph is
    required (analyze / generate). Provisioning accepts UI format separately.
    """
    if not is_api_workflow(workflow) and is_ui_workflow(workflow):
        raise openai_error(400, API_FORMAT_HINT)


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
    require_api_workflow(workflow)
    assert _http is not None, "comfy http client not initialized (lifespan)"
    base = base_url.rstrip("/")
    try:
        resp = await _http.get(f"{base}/object_info")
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    _check_comfy_response(resp, base, "object_info")
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
    except httpx.HTTPError as exc:
        raise _unreachable_error(base, exc)
    _check_comfy_response(response, base, "queue request")
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
    except httpx.HTTPError as exc:
        # Never reached ComfyUI at all: connect refused, DNS, timeout.
        logger.error("ComfyUI /prompt unreachable (%s): %s", base_url, exc)
        raise _unreachable_error(base_url, exc)

    if response.status_code >= 400:
        if _is_gateway_status(response.status_code):
            # Cloudflare (or another proxy) answered — the box is down.
            logger.error(
                "ComfyUI /prompt got a gateway error (%s): HTTP %s",
                base_url,
                response.status_code,
            )
            raise _unreachable_error(
                base_url, f"HTTP {response.status_code} from the tunnel"
            )
        # ComfyUI itself rejected the graph. Its body names the failing node,
        # the input, and the values it would have accepted — surface all of it,
        # because the bare status line says nothing actionable.
        detail = _describe_comfy_error(response)
        # A rejection can mean our cached idea of this instance's architecture
        # is stale (models added/removed since the probe), so re-probe next time
        # rather than letting one bad detection wedge the instance permanently.
        _arch_cache.pop(base_url, None)
        logger.error(
            "ComfyUI rejected the workflow (%s): HTTP %s: %s",
            base_url,
            response.status_code,
            detail,
        )
        raise openai_error(
            502,
            f"ComfyUI rejected the workflow ({base_url}): {detail}",
            "api_error",
            "workflow_rejected",
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
