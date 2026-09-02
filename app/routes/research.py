"""NotebookLM deep research (google keys) — start, poll, import, cancel.

Research is long-running (deep mode routinely runs for many minutes), so this
follows the same async shape as artifact generation: start returns a task_id
immediately and the caller polls, rather than holding a request open for the
library's 1800s default wait.

  POST   /v1/notebooklm/{nb}/research               start a research task
  GET    /v1/notebooklm/{nb}/research/{task_id}     poll it
  POST   /v1/notebooklm/{nb}/research/{task_id}/import   import found sources
  DELETE /v1/notebooklm/{nb}/research/{task_id}     cancel it

Note there is no `research.discover()` in notebooklm-py 0.8.1 despite what the
0.8.0 docs describe — discovery is expressed as `mode` on start(), and the
resolved DiscoveryMode is echoed back on the polled task.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from notebooklm import (
    AmbiguousResearchTaskError,
    NotebookNotFoundError,
    RPCError,
    ValidationError,
)
from pydantic import BaseModel, ConfigDict

from ..pool import get_notebook_client
from ..schemas import openai_error
from ..security import AuthContext, describe_error, log_request, require_google_auth
from .notebooklm import _to_dict

logger = logging.getLogger("ai-sidecar.notebooklm.research")
router = APIRouter()


class ResearchStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str
    # "web" or "drive"; "fast" or "deep" (deep is web-only — the library
    # raises ValidationError on an invalid pair, which maps to 400 below).
    source: str = "web"
    mode: str = "fast"


class ResearchImportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Each entry carries at least 'url' and 'title'; deep-research results
    # from poll() may also carry 'report_markdown' + 'research_task_id'.
    sources: List[Dict[str, Any]]
    # Poll-and-retry until each import is confirmed, instead of firing once.
    verify: bool = False


@asynccontextmanager
async def _tracked(ctx: AuthContext, endpoint: str, target: str):
    """Shared timing/status/error logging for this module's handlers."""
    started = time.perf_counter()
    state = {"status": 500, "error": None}
    try:
        yield state
        state["status"] = 200
    except Exception as exc:
        state["status"] = getattr(exc, "status_code", 500)
        state["error"] = describe_error(exc)
        raise
    finally:
        log_request(ctx, endpoint, target, state["status"], started, state["error"])


def _research_error(exc: Exception, notebook_id: str):
    """Map the research-specific failures onto OpenAI-shaped errors."""
    if isinstance(exc, NotebookNotFoundError):
        return openai_error(
            404, f"Notebook '{notebook_id}' was not found.", code="not_found"
        )
    if isinstance(exc, AmbiguousResearchTaskError):
        # 0.8.0 stopped guessing between concurrent tasks. Our poll endpoint
        # always passes a task_id, so this means the id did not match one.
        return openai_error(
            409,
            f"Several research tasks are in flight for '{notebook_id}' and the "
            f"given task_id did not select one: {exc}",
            "invalid_request_error",
            "ambiguous_task",
        )
    if isinstance(exc, ValidationError):
        # e.g. mode="deep" with source="drive" — deep is web-only.
        return openai_error(400, f"Invalid research request: {exc}")
    return None


@router.post("/v1/notebooklm/{notebook_id}/research")
async def research_start(
    notebook_id: str,
    request: ResearchStartRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/research/start", notebook_id):
        client = await get_notebook_client(ctx.profile_name)
        try:
            result = await client.research.start(
                notebook_id, request.query, source=request.source, mode=request.mode
            )
        except Exception as exc:
            mapped = _research_error(exc, notebook_id)
            if mapped:
                raise mapped
            if isinstance(exc, RPCError):
                # 0.8.0: a "couldn't-start" payload raises rather than
                # returning None, so this is a real refusal, not an empty run.
                raise openai_error(
                    502, f"NotebookLM could not start research: {exc}", "api_error"
                )
            raise
        return {"notebook_id": notebook_id, **_to_dict(result)}


@router.get("/v1/notebooklm/{notebook_id}/research/{task_id}")
async def research_poll(
    notebook_id: str,
    task_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/research/poll", notebook_id):
        client = await get_notebook_client(ctx.profile_name)
        try:
            # ALWAYS pass task_id: with 2+ tasks in flight and task_id=None,
            # 0.8.0 raises AmbiguousResearchTaskError instead of guessing.
            task = await client.research.poll(notebook_id, task_id)
        except Exception as exc:
            mapped = _research_error(exc, notebook_id)
            if mapped:
                raise mapped
            raise
        # ResearchTask dropped dict-subscript access in 0.8.0 — _to_dict reads
        # attributes, which is the supported path.
        return {"notebook_id": notebook_id, "task_id": task_id, **_to_dict(task)}


@router.post("/v1/notebooklm/{notebook_id}/research/{task_id}/import")
async def research_import(
    notebook_id: str,
    task_id: str,
    request: ResearchImportRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/research/import", notebook_id):
        client = await get_notebook_client(ctx.profile_name)
        importer = (
            client.research.import_sources_with_verification
            if request.verify
            else client.research.import_sources
        )
        try:
            imported = await importer(notebook_id, task_id, request.sources)
        except Exception as exc:
            mapped = _research_error(exc, notebook_id)
            if mapped:
                raise mapped
            raise
        return {
            "notebook_id": notebook_id,
            "task_id": task_id,
            "imported": imported,
        }


@router.delete("/v1/notebooklm/{notebook_id}/research/{task_id}")
async def research_cancel(
    notebook_id: str,
    task_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    async with _tracked(ctx, "/v1/notebooklm/research/cancel", notebook_id):
        client = await get_notebook_client(ctx.profile_name)
        try:
            # cancel() takes the RUN id, which start() returns as report_id
            # on the ResearchStart; callers pass whichever id they hold.
            await client.research.cancel(notebook_id, task_id)
        except Exception as exc:
            mapped = _research_error(exc, notebook_id)
            if mapped:
                raise mapped
            raise
        return {"notebook_id": notebook_id, "task_id": task_id, "cancelled": True}
