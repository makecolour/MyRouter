"""/gemini/v1 — Gemini-only provider surface for 9Router (header google key)."""

import logging
import time

from fastapi import APIRouter, Depends

from ..schemas import ChatCompletionRequest, flatten_messages, openai_error
from ..security import AuthContext, describe_error, log_request, require_google_auth
from .chat import (
    _chat_response,
    _gemini_chat,
    _gemini_stream_response,
    delete_conversation,
    list_conversations,
)
from .models_list import gemini_model_entries

logger = logging.getLogger("ai-sidecar.gemini-api")
router = APIRouter(prefix="/gemini/v1")


@router.get("/models")
async def gemini_models(ctx: AuthContext = Depends(require_google_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        data = await gemini_model_entries(ctx)
        status = 200
        return {"object": "list", "data": data}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/gemini/v1/models", None, status, started, error)


@router.post("/chat/completions")
async def gemini_chat_completions(
    request: ChatCompletionRequest,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        prompt = flatten_messages(request.messages)
        if not prompt:
            raise openai_error(400, "The messages array resolved to an empty prompt.")
        model = request.model.strip()
        if not (model == "gemini" or model.startswith("gemini-")):
            raise openai_error(
                404,
                f"Model '{request.model}' is not a Gemini model. This surface "
                f"serves Gemini only — use /notebooklm/<api-key>/v1 for "
                f"notebooks.",
                code="model_not_found",
            )
        if request.stream:
            response = await _gemini_stream_response(request, ctx, model, prompt)
        else:
            answer, conversation_id = await _gemini_chat(request, ctx, model, prompt)
            response = _chat_response(request, model, prompt, answer, conversation_id)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/gemini/v1/chat/completions", request.model, status, started, error
        )


# Conversations belong to the Gemini surface — same handlers as legacy.
router.add_api_route("/conversations", list_conversations, methods=["GET"])
router.add_api_route(
    "/conversations/{conversation_id}", delete_conversation, methods=["DELETE"]
)
