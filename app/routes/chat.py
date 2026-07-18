"""/v1/chat/completions — Gemini (by model, optional conversations) and NotebookLM.

Routing:
  * "gemini"            -> Gemini with the account's default model
  * "gemini-*"          -> Gemini with that model (e.g. gemini-3-pro)
  * anything else       -> NotebookLM notebook id

Gemini conversations (optional): request field `conversation_id` — "new"
starts a server-side thread, an existing id continues it (only the last user
message is sent); the response carries `conversation_id` back.
"""

import copy
import logging
import time
import uuid
from typing import Optional, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gemini_webapi import (
    APIError,
    GeminiError,
    ModelInvalid,
    TemporarilyBlocked,
    UsageLimitExceeded,
)
from gemini_webapi import TimeoutError as GeminiTimeoutError
from gemini_webapi.constants import DEFAULT_METADATA as GEMINI_DEFAULT_METADATA
from notebooklm import NotebookNotFoundError
from sqlalchemy import select

from ..db import SessionLocal
from ..models import GeminiConversation, utcnow
from ..pool import get_gemini_client, get_notebook_client, notebook_locks
from ..schemas import (
    ChatCompletionRequest,
    build_chat_response,
    extract_text,
    flatten_messages,
    last_user_message,
    openai_error,
    sse_chunks,
)
from ..security import AuthContext, describe_error, log_request, require_google_auth

logger = logging.getLogger("ai-sidecar.chat")
router = APIRouter()


@router.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    ctx: AuthContext = Depends(require_google_auth),
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
        log_request(ctx, "/v1/chat/completions", request.model, status, started, error)


async def _handle(request: ChatCompletionRequest, ctx: AuthContext):
    prompt = flatten_messages(request.messages)
    if not prompt:
        raise openai_error(400, "The messages array resolved to an empty prompt.")

    model = request.model.strip()
    if not model:
        raise openai_error(
            400, "Model must be 'gemini', 'gemini-*' or a NotebookLM notebook id."
        )

    conversation_id: Optional[str] = None
    if model == "gemini" or model.startswith("gemini-"):
        answer, conversation_id = await _gemini_chat(request, ctx, model, prompt)
    else:
        answer = await _notebook_chat(ctx, model, prompt)

    if request.stream:
        return StreamingResponse(
            sse_chunks(model, answer, conversation_id),
            media_type="text/event-stream",
        )
    payload = build_chat_response(model, prompt, answer)
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return payload


# Pristine snapshot of the lib's default metadata, taken at import time —
# before any chat could mutate the shared module-level list (see
# _fresh_session below). No hardcoded shape: it tracks the installed lib.
_FRESH_METADATA = copy.deepcopy(GEMINI_DEFAULT_METADATA)


def _fresh_session(client, model_kwargs: dict, metadata=None):
    """start_chat() with a PRIVATE metadata list.

    gemini_webapi 2.0 bug: every ChatSession is initialized with the shared
    module-level DEFAULT_METADATA list and the cid/rid/rcid setters mutate it
    in place — so the first conversation's cid leaks into every later "new"
    session (and even stateless calls), silently threading everything into
    ONE Gemini conversation. Replacing the slot with a private copy isolates
    each session properly.
    """
    session = client.start_chat(**model_kwargs)
    session._ChatSession__metadata = list(_FRESH_METADATA)
    if metadata:
        session.metadata = metadata
    return session


async def _gemini_chat(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
) -> Tuple[str, Optional[str]]:
    model_kwargs = {"model": model} if model.startswith("gemini-") else {}
    conv_id = (request.conversation_id or "").strip() or None

    try:
        client = await get_gemini_client(ctx.profile_name)
        if conv_id is None:
            # Stateless: the whole (flattened) history in one prompt.
            logger.info(
                "chat -> Gemini (profile=%s, model=%s)", ctx.profile_name, model
            )
            result = await client.generate_content(prompt, **model_kwargs)
            return extract_text(result), None

        message = last_user_message(request.messages) or prompt

        if conv_id == "new":
            session = _fresh_session(client, model_kwargs)
            conv_id = f"conv-{uuid.uuid4().hex}"
            logger.info(
                "chat -> Gemini new conversation %s (profile=%s, model=%s)",
                conv_id,
                ctx.profile_name,
                model,
            )
            result = await session.send_message(message)
            answer = extract_text(result)
            async with SessionLocal() as db:
                db.add(
                    GeminiConversation(
                        id=conv_id,
                        profile_name=ctx.profile_name,
                        model=model,
                        title=message[:250],
                        chat_metadata=session.metadata,
                    )
                )
                await db.commit()
            return answer, conv_id

        # Continue an existing conversation.
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(GeminiConversation).where(
                        GeminiConversation.id == conv_id,
                        GeminiConversation.profile_name == ctx.profile_name,
                    )
                )
            ).scalar_one_or_none()
        if row is None:
            raise openai_error(
                404,
                f"Conversation '{conv_id}' was not found for this API key. "
                f"Pass conversation_id='new' to start one.",
                code="conversation_not_found",
            )
        logger.info(
            "chat -> Gemini conversation %s (profile=%s, model=%s)",
            conv_id,
            ctx.profile_name,
            model,
        )
        session = _fresh_session(client, model_kwargs, metadata=row.chat_metadata)
        result = await session.send_message(message)
        answer = extract_text(result)
        async with SessionLocal() as db:
            row = await db.get(GeminiConversation, conv_id)
            if row is not None:
                row.chat_metadata = session.metadata
                row.model = model
                row.updated_at = utcnow()
                await db.commit()
        return answer, conv_id

    except ModelInvalid:
        raise openai_error(
            404,
            f"Gemini model '{model}' is invalid or unavailable for this "
            f"account. List valid models via GET /v1/models.",
            code="model_not_found",
        )
    except ValueError as exc:
        # gemini_webapi raises a plain ValueError ("Unknown model name: …")
        # for an unrecognized model string.
        if "model name" in str(exc).lower():
            raise openai_error(
                404,
                f"Gemini model '{model}' is invalid or unavailable for this "
                f"account. List valid models via GET /v1/models.",
                code="model_not_found",
            )
        raise openai_error(502, f"Gemini request failed: {exc}", "api_error")
    except UsageLimitExceeded as exc:
        raise openai_error(
            429,
            f"Gemini usage limit exceeded for model '{model}': {exc}",
            "rate_limit_error",
            "rate_limit_exceeded",
        )
    except TemporarilyBlocked as exc:
        raise openai_error(
            429,
            f"Google temporarily blocked this account's requests: {exc}",
            "rate_limit_error",
            "temporarily_blocked",
        )
    except GeminiTimeoutError as exc:
        raise openai_error(
            504,
            f"Gemini request timed out: {str(exc) or 'no response in time'}",
            "api_error",
            "timeout",
        )
    except (APIError, GeminiError) as exc:
        logger.exception(
            "Gemini API error (profile=%s, model=%s)", ctx.profile_name, model
        )
        raise openai_error(
            502,
            f"Gemini request failed: {str(exc) or type(exc).__name__}",
            "api_error",
        )
    except Exception as exc:
        if hasattr(exc, "status_code"):  # our own HTTPExceptions pass through
            raise
        logger.exception(
            "Gemini request failed (profile=%s, model=%s)", ctx.profile_name, model
        )
        raise openai_error(
            502,
            f"Gemini request failed: {str(exc) or type(exc).__name__}",
            "api_error",
        )


@router.get("/v1/conversations")
async def list_conversations(
    model: Optional[str] = None,
    ctx: AuthContext = Depends(require_google_auth),
):
    """The sidecar's own Gemini conversation history for this key's profile.

    (Google's web sidebar does not list API-created conversations — this
    endpoint is the reliable history source.)
    """
    started = time.perf_counter()
    status = 500
    error = None
    try:
        async with SessionLocal() as db:
            query = (
                select(GeminiConversation)
                .where(GeminiConversation.profile_name == ctx.profile_name)
                .order_by(GeminiConversation.updated_at.desc())
                .limit(100)
            )
            if model:
                query = query.where(GeminiConversation.model == model)
            rows = (await db.execute(query)).scalars().all()
        status = 200
        return {
            "object": "list",
            "data": [
                {
                    "id": row.id,
                    "model": row.model,
                    "title": row.title,
                    "created_at": str(row.created_at) if row.created_at else None,
                    "updated_at": str(row.updated_at) if row.updated_at else None,
                }
                for row in rows
            ],
        }
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/conversations", model, status, started, error)


@router.delete("/v1/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    ctx: AuthContext = Depends(require_google_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(GeminiConversation).where(
                        GeminiConversation.id == conversation_id,
                        GeminiConversation.profile_name == ctx.profile_name,
                    )
                )
            ).scalar_one_or_none()
            if row is None:
                raise openai_error(
                    404,
                    f"Conversation '{conversation_id}' was not found for this "
                    f"API key.",
                    code="conversation_not_found",
                )
            await db.delete(row)
            await db.commit()
        status = 200
        return {"id": conversation_id, "deleted": True}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/v1/conversations", conversation_id, status, started, error)


async def _notebook_chat(ctx: AuthContext, notebook_id: str, prompt: str) -> str:
    # NotebookLM is notebook-based, not model-based: any non-gemini model
    # string is treated directly as a notebook id.
    try:
        nb_client = await get_notebook_client(ctx.profile_name)
        logger.info(
            "chat -> NotebookLM (profile=%s, notebook=%s)",
            ctx.profile_name,
            notebook_id,
        )
        # Google drops concurrent NotebookLM requests on the same account, so
        # every request for this profile is serialized through its lock.
        async with notebook_locks[ctx.profile_name]:
            result = await nb_client.chat.ask(notebook_id, prompt)
        return extract_text(result)
    except NotebookNotFoundError:
        raise openai_error(
            404,
            f"Notebook '{notebook_id}' was not found for this account. The "
            f"model must be 'gemini', 'gemini-*' or a NotebookLM notebook id.",
            code="model_not_found",
        )
    except Exception as exc:
        if hasattr(exc, "status_code"):  # our own HTTPExceptions pass through
            raise
        logger.exception(
            "NotebookLM request failed (profile=%s, notebook=%s)",
            ctx.profile_name,
            notebook_id,
        )
        raise openai_error(
            502,
            f"NotebookLM request failed: {str(exc) or type(exc).__name__}",
            "api_error",
        )
