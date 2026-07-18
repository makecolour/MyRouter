"""/v1/models — discovery for 9Router: Gemini models + notebooks + comfy instances.

Each backend is listed independently (own try/except), so one failing
backend degrades to a partial list instead of an error.
"""

import inspect
import logging
import time
from typing import List

from fastapi import APIRouter, Depends
from gemini_webapi.constants import Model as GeminiModel
from sqlalchemy import select

from ..db import SessionLocal
from ..models import ComfyInstance
from ..pool import get_gemini_client, get_notebook_client
from ..security import AuthContext, describe_error, get_auth, log_request

logger = logging.getLogger("ai-sidecar.models")
router = APIRouter()


@router.get("/v1/models")
async def list_models(ctx: AuthContext = Depends(get_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await _handle(ctx)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/models", None, status, started, error)


def _static_gemini_models() -> List[dict]:
    names = []
    for member in GeminiModel:
        name = getattr(member, "model_name", None)
        if name and name != "unspecified":
            names.append(name)
    return names


async def _handle(ctx: AuthContext) -> dict:
    created = int(time.time())

    # Comfy keys see exactly their one bound instance.
    if ctx.kind == "comfy":
        async with SessionLocal() as session:
            instance = (
                await session.execute(
                    select(ComfyInstance).where(
                        ComfyInstance.name == ctx.comfy_instance,
                        ComfyInstance.enabled.is_(True),
                    )
                )
            ).scalar_one_or_none()
        data = (
            [
                {
                    "id": instance.name,
                    "object": "model",
                    "created": created,
                    "owned_by": "comfyui",
                }
            ]
            if instance
            else []
        )
        return {"object": "list", "data": data}

    data: List[dict] = [
        {"id": "gemini", "object": "model", "created": created, "owned_by": "google"}
    ]

    # --- Gemini models (per-account: Pro tiers show up when available) ---
    try:
        gemini_client = await get_gemini_client(ctx.profile_name)
        models = gemini_client.list_models()
        if inspect.isawaitable(models):
            models = await models
        if models:
            for m in models:
                name = getattr(m, "model_name", None)
                if not name or name == "unspecified":
                    continue
                # NOTE: is_available is NOT checked — with NotebookLM-exported
                # cookies the registry marks Pro/Thinking unavailable even
                # though generate_content works with them (verified live).
                data.append(
                    {
                        "id": name,
                        "object": "model",
                        "created": created,
                        "owned_by": "google",
                        "display_name": getattr(m, "display_name", None),
                    }
                )
        else:
            for name in _static_gemini_models():
                data.append(
                    {
                        "id": name,
                        "object": "model",
                        "created": created,
                        "owned_by": "google",
                    }
                )
    except Exception as exc:
        logger.warning(
            "Could not list Gemini models for '%s' (falling back to the "
            "static list): %s",
            ctx.profile_name,
            exc,
        )
        for name in _static_gemini_models():
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": created,
                    "owned_by": "google",
                }
            )

    # --- NotebookLM notebooks ---
    try:
        nb_client = await get_notebook_client(ctx.profile_name)
        notebooks = await nb_client.notebooks.list()
    except Exception as exc:
        logger.warning("Could not list notebooks for '%s': %s", ctx.profile_name, exc)
        notebooks = []
    for notebook in notebooks:
        notebook_id = getattr(notebook, "id", None) or getattr(
            notebook, "notebook_id", None
        )
        if not notebook_id:
            continue
        data.append(
            {
                "id": notebook_id,
                "object": "model",
                "created": created,
                "owned_by": "notebooklm",
                "display_name": getattr(notebook, "title", None),
            }
        )

    # Google keys don't list ComfyUI entries — image access needs a
    # per-instance comfy key (v3.5 key-kind split).
    return {"object": "list", "data": data}
