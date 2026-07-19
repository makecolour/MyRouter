"""ComfyUI endpoints — authenticated with per-instance ComfyUI keys.

The instance is bound to the API key; the OpenAI `model` field is optional
and only validated against the key's instance.
"""

import base64
import logging
import random
import time
import uuid

import httpx
from fastapi import APIRouter, Depends
from fastapi.responses import Response

from ..comfy import (
    analyze_upload_slots,
    annotated_ref,
    apply_workflow_placeholders,
    build_comfy_workflow,
    comfy_generate,
    fetch_image_bytes,
    fetch_instance_info,
    fetch_queue,
    make_ephemeral,
    resolve_comfy_instance,
    upload_image,
    upload_mask,
)
from .. import comfy_provision
from ..config import settings
from ..schemas import (
    ComfyAnalyzeRequest,
    ComfyProvisionRequest,
    ComfyUploadRequest,
    ImageGenerationRequest,
    decode_image_source,
    openai_error,
)
from ..security import AuthContext, describe_error, log_request, require_comfy_auth

logger = logging.getLogger("ai-sidecar.images")
router = APIRouter()


async def _key_instance(ctx: AuthContext):
    """The comfy_instances row bound to this API key (must be enabled)."""
    return await resolve_comfy_instance(ctx.comfy_instance)


# 9Router "Output Format" aliases (accepted in response_format/output_format).
_DELIVERY_ALIASES = {
    "url": "url",
    "b64_json": "b64",
    "base64": "b64",
    "json": "b64",
    "binary": "binary",
    "binary_file": "binary",
    "file": "binary",
    "raw": "binary",
}


def _delivery_mode(request: ImageGenerationRequest) -> str:
    value = (request.response_format or request.output_format or "url").strip().lower()
    mode = _DELIVERY_ALIASES.get(value)
    if mode is None:
        raise openai_error(
            400,
            f"Invalid response_format '{value}'. Valid values: url, "
            f"b64_json (aliases: base64, json), binary (aliases: file, "
            f"binary_file, raw).",
        )
    return mode


def _parse_size(size: str) -> tuple:
    normalized = (size or "").strip().lower()
    if normalized in ("", "auto"):
        normalized = settings.comfy_default_size.lower()
    width_str, _, height_str = normalized.partition("x")
    try:
        return int(width_str), int(height_str)
    except ValueError:
        raise openai_error(
            400, f"Invalid size '{size}'; expected '<width>x<height>' or 'auto'."
        )


@router.post("/v1/images/generations")
async def images_generations(
    request: ImageGenerationRequest,
    ctx: AuthContext = Depends(require_comfy_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await _handle(request, ctx)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx,
            "/v1/images/generations",
            request.model or (ctx.comfy_instance if ctx else None),
            status,
            started,
            error,
        )


async def _handle(request: ImageGenerationRequest, ctx: AuthContext):
    if request.model and request.model.strip() != ctx.comfy_instance:
        raise openai_error(
            400,
            f"This API key is bound to ComfyUI instance "
            f"'{ctx.comfy_instance}'; it cannot target '{request.model}'.",
        )

    mode = _delivery_mode(request)
    width, height = _parse_size(request.size)

    instance = await _key_instance(ctx)
    base_url = instance.base_url.rstrip("/")

    ephemeral = (
        request.ephemeral if request.ephemeral is not None else settings.comfy_ephemeral
    )
    auto_provision = (
        request.auto_provision
        if request.auto_provision is not None
        else settings.comfy_auto_provision
    )

    n = max(1, min(request.n, 4))
    if mode == "binary":
        n = 1  # a raw file body can only carry one image
    logger.info(
        "images -> ComfyUI '%s' (%s) (n=%d, size=%dx%d, mode=%s, ephemeral=%s)",
        instance.name,
        base_url,
        n,
        width,
        height,
        mode,
        ephemeral,
    )

    # Self-heal: install missing nodes / download missing models the supplied
    # workflow needs before we try to render with it. Only meaningful with a
    # caller-supplied workflow (the built-in one uses stock nodes).
    if auto_provision and request.workflow:
        report = await comfy_provision.provision(
            base_url, request.workflow, wait=True
        )
        logger.info("auto-provision report: %s", report)
    urls = []
    for index in range(n):
        if request.seed is not None:
            seed = request.seed + index  # reproducible, distinct per image
        else:
            seed = random.randint(0, 2**32 - 1)
        if request.workflow:
            workflow = apply_workflow_placeholders(
                request.workflow, request.prompt, width, height, seed
            )
        else:
            workflow = build_comfy_workflow(
                request.prompt,
                width,
                height,
                seed,
                negative_prompt=request.negative_prompt,
                checkpoint=request.checkpoint,
                steps=request.steps,
                cfg=request.cfg,
                sampler=request.sampler,
                scheduler=request.scheduler,
                denoise=request.denoise,
            )
        if ephemeral:
            workflow = make_ephemeral(workflow)
        logger.info("ComfyUI generation %d/%d (seed=%d)", index + 1, n, seed)
        urls.append(await comfy_generate(base_url, workflow, ephemeral=ephemeral))

    if mode == "url":
        return {"created": int(time.time()), "data": [{"url": url} for url in urls]}

    images = [await fetch_image_bytes(url) for url in urls]
    if mode == "binary":
        return Response(content=images[0], media_type="image/png")
    return {
        "created": int(time.time()),
        "data": [
            {"b64_json": base64.b64encode(image).decode("ascii")} for image in images
        ],
    }


@router.get("/v1/comfy/info")
async def comfy_info(ctx: AuthContext = Depends(require_comfy_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        info = await fetch_instance_info(instance.base_url)
        status = 200
        return {
            "instance": instance.name,
            "base_url": instance.base_url,
            **info,
        }
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/comfy/info", ctx.comfy_instance, status, started, error)


@router.get("/v1/comfy/{instance_name}/info")
async def comfy_info_named(
    instance_name: str, ctx: AuthContext = Depends(require_comfy_auth)
):
    if instance_name != ctx.comfy_instance:
        raise openai_error(
            400,
            f"This API key is bound to ComfyUI instance "
            f"'{ctx.comfy_instance}'; it cannot query '{instance_name}'.",
        )
    return await comfy_info(ctx)


@router.get("/v1/comfy/queue")
async def comfy_queue(ctx: AuthContext = Depends(require_comfy_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        queue = await fetch_queue(instance.base_url)
        status = 200
        return {"instance": instance.name, **queue}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/comfy/queue", ctx.comfy_instance, status, started, error)


@router.post("/v1/comfy/provision")
async def comfy_provision_endpoint(
    request: ComfyProvisionRequest,
    ctx: AuthContext = Depends(require_comfy_auth),
):
    """Push a workflow → ComfyUI-Manager installs missing nodes/models.

    `wait=false` returns immediately once the install queue is started; poll
    GET /v1/comfy/provision/status for progress.
    """
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        report = await comfy_provision.provision(
            instance.base_url,
            request.workflow,
            wait=request.wait,
            timeout=request.timeout,
        )
        status = 200
        return {"instance": instance.name, **report}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/comfy/provision", ctx.comfy_instance, status, started, error
        )


@router.get("/v1/comfy/provision/status")
async def comfy_provision_status(ctx: AuthContext = Depends(require_comfy_auth)):
    """Manager install-queue status (done_count/total_count/is_processing)."""
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        try:
            queue = await comfy_provision.queue_status(instance.base_url.rstrip("/"))
        except httpx.HTTPError as exc:
            raise openai_error(
                502, f"ComfyUI Manager status failed: {exc}", "api_error"
            )
        status = 200
        return {"instance": instance.name, **queue}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx,
            "/v1/comfy/provision/status",
            ctx.comfy_instance,
            status,
            started,
            error,
        )


@router.post("/v1/comfy/analyze")
async def comfy_analyze(
    request: ComfyAnalyzeRequest,
    ctx: AuthContext = Depends(require_comfy_auth),
):
    """Discover which workflow node inputs take an uploaded file.

    Returns one slot per file input (LoadImage etc.): its node_id + input_name,
    so a caller/UI knows exactly where each imported file goes.
    """
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        slots = await analyze_upload_slots(instance.base_url, request.workflow)
        status = 200
        return {"instance": instance.name, "slots": slots}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/comfy/analyze", ctx.comfy_instance, status, started, error
        )


@router.post("/v1/comfy/upload")
async def comfy_upload(
    request: ComfyUploadRequest,
    ctx: AuthContext = Depends(require_comfy_auth),
):
    """Import an image (or inpaint mask) into ComfyUI's input/temp dir.

    Returns the stored `{name, subfolder, type}` plus a LoadImage-ready `ref`
    (temp uploads carry the `[temp]` annotation). Ephemeral uploads land in the
    auto-cleaned temp dir instead of the permanent input gallery.
    """
    started = time.perf_counter()
    status = 500
    error = None
    try:
        instance = await _key_instance(ctx)
        data, ext = decode_image_source(request.image)
        if data is None:  # http(s) URL — fetch where the shared client lives
            data = await fetch_image_bytes(request.image)
        limit = int(settings.vision_max_image_mb * 1024 * 1024)
        if len(data) > limit:
            raise openai_error(
                400,
                f"Image exceeds the {settings.vision_max_image_mb:.0f} MB limit.",
            )
        filename = request.filename or f"myrouter_{uuid.uuid4().hex}{ext}"
        ephemeral = (
            request.ephemeral
            if request.ephemeral is not None
            else settings.comfy_ephemeral
        )
        image_type = "temp" if ephemeral else "input"
        if request.mask:
            if not request.original_ref:
                raise openai_error(
                    400,
                    "A mask upload requires original_ref (a prior upload's "
                    "{filename, subfolder, type}).",
                )
            result = await upload_mask(
                instance.base_url,
                data,
                filename,
                request.original_ref,
                image_type=image_type,
            )
        else:
            result = await upload_image(
                instance.base_url, data, filename, image_type=image_type
            )
        status = 200
        return {"instance": instance.name, "ref": annotated_ref(result), **result}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/comfy/upload", ctx.comfy_instance, status, started, error
        )
