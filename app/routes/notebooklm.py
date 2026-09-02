"""NotebookLM command API (google keys): generate, list, status, download.

Chat stays on /v1/chat/completions (notebook id as model). These endpoints
expose the rest of notebooklm-py's artifacts surface:

  POST /v1/notebooklm/generate                     kick off artifact generation
  GET  /v1/notebooklm/{nb}/artifacts[?type=]       list artifacts
  GET  /v1/notebooklm/{nb}/status/{task_id}        poll generation status
  GET  /v1/notebooklm/{nb}/download/{type}         download artifact as a file
  GET  /v1/notebooklm/{nb}/artifacts/{id}/prompt   read the generation prompt
  POST /v1/notebooklm/{nb}/artifacts/{id}/rename   rename an artifact
  DELETE /v1/notebooklm/{nb}/artifacts/{id}        delete an artifact
"""

import dataclasses
import inspect
import logging
import os
import tempfile
import time
from typing import Any, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import FileResponse
from notebooklm import (
    ArtifactFeatureUnavailableError,
    ArtifactNotFoundError,
    NotebookNotFoundError,
    RateLimitError,
    RPCError,
)
from pydantic import BaseModel, ConfigDict
from starlette.background import BackgroundTask

from ..pool import get_notebook_client
from ..schemas import openai_error
from ..security import AuthContext, describe_error, log_request, require_google_auth

logger = logging.getLogger("ai-sidecar.notebooklm")
router = APIRouter()

# artifact type -> client.artifacts method name
_GENERATORS = {
    "audio": "generate_audio",
    "video": "generate_video",
    "report": "generate_report",
    "study_guide": "generate_study_guide",
    "quiz": "generate_quiz",
    "flashcards": "generate_flashcards",
    "slide_deck": "generate_slide_deck",
    "infographic": "generate_infographic",
    "data_table": "generate_data_table",
    "mind_map": "generate_mind_map",
    # 0.8.x: a separate cinematic renderer, distinct from generate_video.
    # It takes `instructions`, so the free-text sniff already handles it, and
    # its output downloads through the same video path.
    "cinematic_video": "generate_cinematic_video",
}

# artifact type -> (method, default ext, media type, per-format overrides)
_OFFICE_PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
_TEXT_FORMATS = {
    "json": (".json", "application/json"),
    "markdown": (".md", "text/markdown"),
    "html": (".html", "text/html"),
}
_DOWNLOADERS = {
    "audio": ("download_audio", ".mp4", "audio/mp4", None),
    "video": ("download_video", ".mp4", "video/mp4", None),
    "cinematic_video": ("download_video", ".mp4", "video/mp4", None),
    "infographic": ("download_infographic", ".png", "image/png", None),
    "slide_deck": (
        "download_slide_deck",
        ".pdf",
        "application/pdf",
        {"pdf": (".pdf", "application/pdf"), "pptx": (".pptx", _OFFICE_PPTX)},
    ),
    "report": ("download_report", ".md", "text/markdown", None),
    "mind_map": ("download_mind_map", ".json", "application/json", None),
    "data_table": ("download_data_table", ".csv", "text/csv", None),
    "quiz": ("download_quiz", ".json", "application/json", _TEXT_FORMATS),
    "flashcards": ("download_flashcards", ".json", "application/json", _TEXT_FORMATS),
}

_DICT_FIELDS = (
    "task_id",
    "artifact_id",
    "id",
    "status",
    "type",
    "artifact_type",
    "title",
    "name",
    "created_at",
    "updated_at",
    "error",
    "note_id",
)


def _to_dict(obj: Any) -> dict:
    """Serialize lib result objects (dataclasses or plain) to JSON-safe dicts."""
    if obj is None:
        return {}
    data: dict
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        data = dataclasses.asdict(obj)
    else:
        data = {
            field: getattr(obj, field)
            for field in _DICT_FIELDS
            if getattr(obj, field, None) is not None
        }
    return {
        key: value
        if isinstance(value, (str, int, float, bool, dict, list, type(None)))
        else str(value)
        for key, value in data.items()
    }


class NotebookLMGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    notebook_id: str
    type: str
    # Free-text guidance; mapped to the lib's `instructions` or
    # `custom_prompt` parameter depending on the artifact type.
    instructions: Optional[str] = None
    # Advanced passthrough kwargs for the underlying generate_* method
    # (e.g. {"language": "vi"}); validated by notebooklm-py itself.
    options: Optional[dict] = None


@router.post("/v1/notebooklm/generate")
async def notebooklm_generate(
    request: NotebookLMGenerateRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await _generate(request, ctx)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/generate", request.notebook_id, status, started, error
        )


async def _generate(request: NotebookLMGenerateRequest, ctx: AuthContext) -> dict:
    artifact_type = request.type.strip().lower()
    method_name = _GENERATORS.get(artifact_type)
    if method_name is None:
        raise openai_error(
            400,
            f"Unknown artifact type '{request.type}'. "
            f"Valid types: {', '.join(sorted(_GENERATORS))}.",
        )

    client = await get_notebook_client(ctx.profile_name)
    method = getattr(client.artifacts, method_name)

    kwargs = dict(request.options or {})
    if request.instructions:
        # Different generators name their free-text parameter differently, and
        # notebooklm-py renames them between releases (0.8.x moved
        # generate_study_guide from `instructions` to `extra_instructions`).
        # Sniff the installed signature rather than hardcoding a per-type map.
        params = inspect.signature(method).parameters
        for candidate in ("instructions", "custom_prompt", "extra_instructions"):
            if candidate in params:
                kwargs.setdefault(candidate, request.instructions)
                break
        else:
            logger.warning(
                "Artifact type '%s' accepts no instructions; ignoring", artifact_type
            )

    logger.info(
        "notebooklm generate %s (profile=%s, notebook=%s)",
        artifact_type,
        ctx.profile_name,
        request.notebook_id,
    )
    try:
        result = await method(request.notebook_id, **kwargs)
    except NotebookNotFoundError:
        raise openai_error(
            404, f"Notebook '{request.notebook_id}' was not found.", code="not_found"
        )
    except RateLimitError as exc:
        raise openai_error(
            429, f"NotebookLM rate limit: {exc}", "rate_limit_error"
        )
    except TypeError as exc:
        raise openai_error(400, f"Invalid options for '{artifact_type}': {exc}")
    except ArtifactFeatureUnavailableError as exc:
        raise openai_error(
            502,
            f"NotebookLM does not offer '{artifact_type}' on this account: {exc}",
            "api_error",
            "feature_unavailable",
        )
    except RPCError as exc:
        # Since 0.8.0 a synchronous refusal RAISES here; 0.7.x returned a
        # GenerationStatus(status="failed") that fell through as a success.
        raise openai_error(
            502,
            f"NotebookLM refused to generate '{artifact_type}': {exc}",
            "api_error",
        )
    except Exception as exc:
        logger.exception("Generation kickoff failed (%s)", artifact_type)
        raise openai_error(502, f"NotebookLM generation failed: {exc}", "api_error")

    return {
        "notebook_id": request.notebook_id,
        "type": artifact_type,
        **_to_dict(result),
    }


@router.get("/v1/notebooklm/{notebook_id}/artifacts")
async def notebooklm_artifacts(
    notebook_id: str,
    type: Optional[str] = None,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        client = await get_notebook_client(ctx.profile_name)
        try:
            artifacts = await client.artifacts.list(notebook_id)
        except NotebookNotFoundError:
            raise openai_error(
                404, f"Notebook '{notebook_id}' was not found.", code="not_found"
            )
        items = [_to_dict(a) for a in artifacts]
        if type:
            token = type.strip().lower().replace("-", "_")
            items = [
                item
                for item in items
                if token
                in str(item.get("type") or item.get("artifact_type") or "")
                .lower()
                .replace("-", "_")
            ]
        status = 200
        return {"notebook_id": notebook_id, "artifacts": items}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/artifacts", notebook_id, status, started, error
        )


class NotebookLMRenameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str


@router.get("/v1/notebooklm/{notebook_id}/artifacts/{artifact_id}/prompt")
async def notebooklm_artifact_prompt(
    notebook_id: str,
    artifact_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    """Read back the prompt an artifact was generated from (0.8.x)."""
    started = time.perf_counter()
    status = 500
    error = None
    try:
        client = await get_notebook_client(ctx.profile_name)
        try:
            prompt = await client.artifacts.get_prompt(notebook_id, artifact_id)
        except ArtifactNotFoundError:
            raise openai_error(
                404, f"Artifact '{artifact_id}' was not found.", code="not_found"
            )
        except NotebookNotFoundError:
            raise openai_error(
                404, f"Notebook '{notebook_id}' was not found.", code="not_found"
            )
        status = 200
        # get_prompt legitimately returns None for artifacts generated without
        # one — that is an empty answer, not a missing artifact.
        return {
            "notebook_id": notebook_id,
            "artifact_id": artifact_id,
            "prompt": prompt,
        }
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/artifact-prompt", notebook_id, status, started, error
        )


@router.post("/v1/notebooklm/{notebook_id}/artifacts/{artifact_id}/rename")
async def notebooklm_artifact_rename(
    notebook_id: str,
    artifact_id: str,
    request: NotebookLMRenameRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        client = await get_notebook_client(ctx.profile_name)
        try:
            # 0.8.x preflights the target and raises on a miss rather than
            # silently no-op'ing, so a bad id is a real 404 here.
            result = await client.artifacts.rename(
                notebook_id, artifact_id, request.title
            )
        except ArtifactNotFoundError:
            raise openai_error(
                404, f"Artifact '{artifact_id}' was not found.", code="not_found"
            )
        except NotebookNotFoundError:
            raise openai_error(
                404, f"Notebook '{notebook_id}' was not found.", code="not_found"
            )
        status = 200
        return {"notebook_id": notebook_id, "renamed": True, **_to_dict(result)}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/artifact-rename", notebook_id, status, started, error
        )


@router.delete("/v1/notebooklm/{notebook_id}/artifacts/{artifact_id}")
async def notebooklm_artifact_delete(
    notebook_id: str,
    artifact_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        client = await get_notebook_client(ctx.profile_name)
        try:
            # Returns None on success — success is the ABSENCE of an
            # exception (0.8.0 contract); never test it for truthiness.
            await client.artifacts.delete(notebook_id, artifact_id)
        except ArtifactNotFoundError:
            raise openai_error(
                404, f"Artifact '{artifact_id}' was not found.", code="not_found"
            )
        except NotebookNotFoundError:
            raise openai_error(
                404, f"Notebook '{notebook_id}' was not found.", code="not_found"
            )
        status = 200
        return {
            "notebook_id": notebook_id,
            "artifact_id": artifact_id,
            "deleted": True,
        }
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/artifact-delete", notebook_id, status, started, error
        )


@router.get("/v1/notebooklm/{notebook_id}/status/{task_id}")
async def notebooklm_status(
    notebook_id: str,
    task_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        client = await get_notebook_client(ctx.profile_name)
        try:
            result = await client.artifacts.poll_status(notebook_id, task_id)
        except ArtifactNotFoundError:
            raise openai_error(
                404, f"Task/artifact '{task_id}' was not found.", code="not_found"
            )
        status = 200
        return {"notebook_id": notebook_id, "task_id": task_id, **_to_dict(result)}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/notebooklm/status", notebook_id, status, started, error)


@router.get("/v1/notebooklm/{notebook_id}/download/{artifact_type}")
async def notebooklm_download(
    notebook_id: str,
    artifact_type: str,
    artifact_id: Optional[str] = None,
    format: Optional[str] = None,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await _download(
            notebook_id, artifact_type, artifact_id, format, ctx
        )
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/v1/notebooklm/download", notebook_id, status, started, error
        )


async def _download(
    notebook_id: str,
    artifact_type: str,
    artifact_id: Optional[str],
    output_format: Optional[str],
    ctx: AuthContext,
) -> FileResponse:
    key = artifact_type.strip().lower().replace("-", "_")
    entry = _DOWNLOADERS.get(key)
    if entry is None:
        raise openai_error(
            400,
            f"Unknown download type '{artifact_type}'. "
            f"Valid types: {', '.join(sorted(_DOWNLOADERS))}.",
        )
    method_name, ext, media_type, format_map = entry

    kwargs: dict = {}
    if artifact_id:
        kwargs["artifact_id"] = artifact_id
    if output_format:
        if not format_map or output_format not in format_map:
            valid = ", ".join(sorted(format_map)) if format_map else "(none)"
            raise openai_error(
                400,
                f"Format '{output_format}' is not valid for '{key}'. "
                f"Valid formats: {valid}.",
            )
        ext, media_type = format_map[output_format]
        kwargs["output_format"] = output_format

    client = await get_notebook_client(ctx.profile_name)
    method = getattr(client.artifacts, method_name)

    fd, tmp_path = tempfile.mkstemp(prefix="nblm_", suffix=ext)
    os.close(fd)
    logger.info(
        "notebooklm download %s (profile=%s, notebook=%s, artifact=%s)",
        key,
        ctx.profile_name,
        notebook_id,
        artifact_id or "(latest)",
    )
    try:
        await method(notebook_id, tmp_path, **kwargs)
    except ArtifactNotFoundError:
        os.unlink(tmp_path)
        raise openai_error(
            404, f"Artifact '{artifact_id}' was not found.", code="not_found"
        )
    except NotebookNotFoundError:
        os.unlink(tmp_path)
        raise openai_error(
            404, f"Notebook '{notebook_id}' was not found.", code="not_found"
        )
    except ValueError as exc:
        os.unlink(tmp_path)
        raise openai_error(
            404, f"No completed '{key}' artifact to download: {exc}", code="not_found"
        )
    except Exception as exc:
        os.unlink(tmp_path)
        logger.exception("Download failed (%s)", key)
        raise openai_error(502, f"NotebookLM download failed: {exc}", "api_error")

    filename = f"{key}_{notebook_id[:8]}{ext}"
    return FileResponse(
        tmp_path,
        media_type=media_type,
        filename=filename,
        background=BackgroundTask(os.unlink, tmp_path),
    )
