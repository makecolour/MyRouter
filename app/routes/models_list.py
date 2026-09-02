"""/v1/models — combined discovery surface (legacy) + shared entry builders.

The per-backend builders are reused by the split provider surfaces
(/gemini/v1, /notebooklm/{api_key}/v1, /comfyui/v1).
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


def _static_gemini_models() -> List[str]:
    """Hardcoded fallback for when the account registry comes back empty.

    NOTE: gemini_webapi marks `constants.Model` "deprecated, pending removal" —
    model identity is discovered per account at init() now, so these members go
    stale whenever Google renames or retiers a model. Merely iterating the enum
    is warning-free (the lib only warns when a member is passed to
    generate_content), but this fallback should go away once the live registry
    in gemini_model_entries() is trusted to always populate.
    """
    names = []
    for member in GeminiModel:
        name = getattr(member, "model_name", None)
        if name and name != "unspecified":
            names.append(name)
    return names


async def gemini_model_entries(ctx: AuthContext) -> List[dict]:
    """gemini alias + the account's registry models (static fallback)."""
    created = int(time.time())
    data: List[dict] = [
        {"id": "gemini", "object": "model", "created": created, "owned_by": "google"}
    ]
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
    return data


async def notebook_model_entries(ctx: AuthContext) -> List[dict]:
    """Every notebook of the key's profile as a model entry (id + title)."""
    created = int(time.time())
    nb_client = await get_notebook_client(ctx.profile_name)
    notebooks = await nb_client.notebooks.list()
    data: List[dict] = []
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
    return data


async def comfy_model_entries(ctx: AuthContext) -> List[dict]:
    """The single instance bound to a comfy key."""
    created = int(time.time())
    async with SessionLocal() as session:
        instance = (
            await session.execute(
                select(ComfyInstance).where(
                    ComfyInstance.name == ctx.comfy_instance,
                    ComfyInstance.enabled.is_(True),
                )
            )
        ).scalar_one_or_none()
    if instance is None:
        return []
    return [
        {
            "id": instance.name,
            "object": "model",
            "created": created,
            "owned_by": "comfyui",
        }
    ]


@router.get("/v1/models")
async def list_models(ctx: AuthContext = Depends(get_auth)):
    """Legacy combined surface: everything the key kind can reach."""
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


async def _handle(ctx: AuthContext) -> dict:
    if ctx.kind == "comfy":
        return {"object": "list", "data": await comfy_model_entries(ctx)}
    if ctx.kind == "copilot":
        from .copilot_api import copilot_model_entries

        return {"object": "list", "data": copilot_model_entries()}

    data = await gemini_model_entries(ctx)
    try:
        data.extend(await notebook_model_entries(ctx))
    except Exception as exc:
        logger.warning("Could not list notebooks for '%s': %s", ctx.profile_name, exc)
    # Google keys don't list ComfyUI entries — image access needs a
    # per-instance comfy key.
    return {"object": "list", "data": data}
