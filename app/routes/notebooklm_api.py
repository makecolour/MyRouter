"""/notebooklm/{api_key}/v1 — NotebookLM provider surface for 9Router.

The API key travels in the URL path (9Router embeds it in the Base URL), so
no Authorization header is needed. GET /models lists ONLY the notebooks of
the key's Google profile; chat uses a notebook id as the model.
"""

import logging
import time

from fastapi import APIRouter, Depends

from ..schemas import ChatCompletionRequest, flatten_messages, openai_error
from ..security import AuthContext, describe_error, log_request, path_google_auth
from .chat import _chat_response, _notebook_chat
from .models_list import notebook_model_entries

logger = logging.getLogger("ai-sidecar.notebooklm-api")
router = APIRouter(prefix="/notebooklm/{api_key}/v1")


@router.get("/models")
async def notebooklm_models(ctx: AuthContext = Depends(path_google_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        data = await notebook_model_entries(ctx)
        status = 200
        return {"object": "list", "data": data}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/notebooklm/v1/models", None, status, started, error)


@router.post("/chat/completions")
async def notebooklm_chat_completions(
    request: ChatCompletionRequest,
    ctx: AuthContext = Depends(path_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        prompt = flatten_messages(request.messages)
        if not prompt:
            raise openai_error(400, "The messages array resolved to an empty prompt.")
        model = request.model.strip()
        if not model or model == "gemini" or model.startswith("gemini-"):
            raise openai_error(
                404,
                f"Model '{request.model}' is not a notebook id. This surface "
                f"serves NotebookLM only — use /gemini/v1 for Gemini models.",
                code="model_not_found",
            )
        answer = await _notebook_chat(ctx, model, prompt)
        response = _chat_response(request, model, prompt, answer)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx,
            "/notebooklm/v1/chat/completions",
            request.model,
            status,
            started,
            error,
        )
