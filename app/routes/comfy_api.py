"""/comfyui/v1 — ComfyUI provider surface for 9Router (header comfy key)."""

import logging
import time

from fastapi import APIRouter, Depends

from ..schemas import ImageGenerationRequest
from ..security import AuthContext, describe_error, log_request, require_comfy_auth
from .images import _handle as images_handle
from .images import (
    comfy_analyze,
    comfy_info,
    comfy_provision_endpoint,
    comfy_provision_status,
    comfy_queue,
    comfy_upload,
)
from .models_list import comfy_model_entries

logger = logging.getLogger("ai-sidecar.comfy-api")
router = APIRouter(prefix="/comfyui/v1")


@router.get("/models")
async def comfyui_models(ctx: AuthContext = Depends(require_comfy_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        data = await comfy_model_entries(ctx)
        status = 200
        return {"object": "list", "data": data}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/comfyui/v1/models", None, status, started, error)


@router.post("/images/generations")
async def comfyui_generations(
    request: ImageGenerationRequest,
    ctx: AuthContext = Depends(require_comfy_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await images_handle(request, ctx)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx,
            "/comfyui/v1/images/generations",
            request.model or ctx.comfy_instance,
            status,
            started,
            error,
        )


# Discovery + provisioning — same handlers as the legacy surface.
router.add_api_route("/comfy/info", comfy_info, methods=["GET"])
router.add_api_route("/comfy/queue", comfy_queue, methods=["GET"])
router.add_api_route("/comfy/provision", comfy_provision_endpoint, methods=["POST"])
router.add_api_route(
    "/comfy/provision/status", comfy_provision_status, methods=["GET"]
)
router.add_api_route("/comfy/analyze", comfy_analyze, methods=["POST"])
router.add_api_route("/comfy/upload", comfy_upload, methods=["POST"])
