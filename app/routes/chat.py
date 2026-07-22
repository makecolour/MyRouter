"""/v1/chat/completions — Gemini (by model, optional conversations) and NotebookLM.

Routing:
  * "gemini"            -> Gemini with the account's default model
  * "gemini-*"          -> Gemini with that model (e.g. gemini-3-pro)
  * anything else       -> NotebookLM notebook id

Gemini conversations (optional): request field `conversation_id` — "new"
starts a server-side thread, an existing id continues it (only the last user
message is sent); the response carries `conversation_id` back.
"""

import base64
import copy
import json
import logging
import mimetypes
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from gemini_webapi import (
    ModelInvalid,
    TemporarilyBlocked,
    UsageLimitExceeded,
)
from gemini_webapi import GeneratedImage
from gemini_webapi import TimeoutError as GeminiTimeoutError
from gemini_webapi.constants import DEFAULT_METADATA as GEMINI_DEFAULT_METADATA
from notebooklm import NotebookNotFoundError
from sqlalchemy import select

from ..comfy import fetch_image_bytes
from ..config import settings
from ..db import SessionLocal
from ..models import GeminiConversation, utcnow
from ..pool import get_gemini_client, get_notebook_client, notebook_locks
from ..schemas import (
    ChatCompletionRequest,
    build_chat_response,
    collect_images,
    extract_text,
    flatten_messages,
    last_user_message,
    openai_error,
    sse_chunks,
)
from ..security import AuthContext, describe_error, log_request, require_google_auth
from ..tools import (
    build_tool_instruction,
    parse_tool_calls,
    tool_names,
    tools_requested,
)

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

    is_gemini = model == "gemini" or model.startswith("gemini-")

    if is_gemini:
        return await gemini_chat_dispatch(request, ctx, model, prompt)

    # NotebookLM: not streamable; single-answer fake SSE when stream requested.
    answer = await _notebook_chat(ctx, model, prompt)
    return _chat_response(request, model, prompt, answer)


async def gemini_chat_dispatch(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
):
    """Route a Gemini request across tools / streaming / plain chat.

    Shared by the legacy /v1 surface and /gemini/v1 so both get the same
    tool-calling + streaming behavior.
    """
    if tools_requested(request.tools, request.tool_choice) and settings.tool_emulation:
        cleaned, conv_id, tool_calls = await _gemini_chat_tools(
            request, ctx, model, prompt
        )
        if request.stream:
            return _tool_stream_response(model, prompt, cleaned, tool_calls, conv_id)
        payload = build_chat_response(model, prompt, cleaned, tool_calls=tool_calls)
        if conv_id:
            payload["conversation_id"] = conv_id
        return payload

    # Real token streaming for Gemini (low TTFT) — what OpenAI routers expect.
    if request.stream:
        return await _gemini_stream_response(request, ctx, model, prompt)

    answer, conversation_id, images = await _gemini_chat(request, ctx, model, prompt)
    return _chat_response(request, model, prompt, answer, conversation_id, images)


def _chat_response(
    request: ChatCompletionRequest,
    model: str,
    prompt: str,
    answer: str,
    conversation_id: Optional[str] = None,
    images: Optional[List[dict]] = None,
):
    """OpenAI response assembly shared by non-streaming + NotebookLM streaming."""
    if request.stream:
        return StreamingResponse(
            sse_chunks(model, prompt, answer, conversation_id),
            media_type="text/event-stream",
        )
    payload = build_chat_response(model, prompt, answer, images=images)
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


def _map_gemini_exception(exc: Exception, model: str, profile: str) -> HTTPException:
    """Translate a gemini_webapi exception into an OpenAI-shaped HTTPException."""
    if isinstance(exc, HTTPException):  # our own errors pass through
        return exc
    if isinstance(exc, ModelInvalid):
        return openai_error(
            404,
            f"Gemini model '{model}' is invalid or unavailable for this "
            f"account. List valid models via GET /v1/models.",
            code="model_not_found",
        )
    if isinstance(exc, ValueError) and "model name" in str(exc).lower():
        return openai_error(
            404,
            f"Gemini model '{model}' is invalid or unavailable for this "
            f"account. List valid models via GET /v1/models.",
            code="model_not_found",
        )
    if isinstance(exc, UsageLimitExceeded):
        return openai_error(
            429,
            f"Gemini usage limit exceeded for model '{model}': {exc}",
            "rate_limit_error",
            "rate_limit_exceeded",
        )
    if isinstance(exc, TemporarilyBlocked):
        return openai_error(
            429,
            f"Google temporarily blocked this account's requests: {exc}",
            "rate_limit_error",
            "temporarily_blocked",
        )
    if isinstance(exc, GeminiTimeoutError):
        return openai_error(
            504,
            f"Gemini request timed out: {str(exc) or 'no response in time'}",
            "api_error",
            "timeout",
        )
    logger.exception("Gemini request failed (profile=%s, model=%s)", profile, model)
    return openai_error(
        502, f"Gemini request failed: {str(exc) or type(exc).__name__}", "api_error"
    )


async def _prepare_gemini_session(
    ctx: AuthContext,
    client,
    model_kwargs: dict,
    request: ChatCompletionRequest,
    prompt: str,
    conv_id_in: Optional[str],
):
    """Resolve the chat session + what to send.

    Returns (session_or_None, send_text, conv_id_or_None, is_new, title).
    session None => stateless (send the flattened prompt). Raises a clean 404
    for an unknown conversation id.
    """
    if conv_id_in is None:
        return None, prompt, None, False, None

    message = last_user_message(request.messages) or prompt
    if conv_id_in == "new":
        session = _fresh_session(client, model_kwargs)
        return session, message, f"conv-{uuid.uuid4().hex}", True, message[:250]

    async with SessionLocal() as db:
        row = (
            await db.execute(
                select(GeminiConversation).where(
                    GeminiConversation.id == conv_id_in,
                    GeminiConversation.profile_name == ctx.profile_name,
                )
            )
        ).scalar_one_or_none()
    if row is None:
        raise openai_error(
            404,
            f"Conversation '{conv_id_in}' was not found for this API key. "
            f"Pass conversation_id='new' to start one.",
            code="conversation_not_found",
        )
    session = _fresh_session(client, model_kwargs, metadata=row.chat_metadata)
    return session, message, conv_id_in, False, None


async def _persist_conversation(
    is_new: bool,
    conv_id: str,
    ctx: AuthContext,
    model: str,
    session,
    title: Optional[str],
) -> None:
    """Save the server-side conversation metadata after a turn completes."""
    async with SessionLocal() as db:
        if is_new:
            db.add(
                GeminiConversation(
                    id=conv_id,
                    profile_name=ctx.profile_name,
                    model=model,
                    title=title,
                    chat_metadata=session.metadata,
                )
            )
        else:
            row = await db.get(GeminiConversation, conv_id)
            if row is not None:
                row.chat_metadata = session.metadata
                row.model = model
                row.updated_at = utcnow()
        await db.commit()


async def _prepare_vision_files(request: ChatCompletionRequest) -> List[Path]:
    """Materialize the request's input images to temp files with the right
    extension (gemini_webapi derives the upload MIME from the filename).

    The caller MUST unlink the returned paths after the model call.
    """
    sources = collect_images(request.messages)
    if not sources:
        return []
    limit = int(settings.vision_max_image_mb * 1024 * 1024)
    paths: List[Path] = []
    try:
        for src in sources:
            if "data" in src:
                data, ext = src["data"], src["ext"]
            else:
                data = await fetch_image_bytes(src["url"])
                mime = mimetypes.guess_type(src["url"])[0] or "image/jpeg"
                ext = mimetypes.guess_extension(mime) or ".jpg"
            if len(data) > limit:
                raise openai_error(
                    400,
                    f"Input image exceeds the "
                    f"{settings.vision_max_image_mb:.0f} MB limit.",
                )
            fd, tmp = tempfile.mkstemp(prefix="myrouter_img_", suffix=ext)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            paths.append(Path(tmp))
        return paths
    except Exception:
        _cleanup_files(paths)
        raise


def _cleanup_files(paths: List[Path]) -> None:
    for path in paths:
        try:
            path.unlink()
        except OSError:
            pass


def _minimal_image_session(gclient):
    """A curl_cffi session with ONLY the two core auth cookies.

    The pooled client injects the full ~38-cookie jar (flattened to
    .google.com) which Google's image CDN (lh3.googleusercontent.com) rejects
    with 403. A minimal PSID/PSIDTS session — like the stock GeminiClient —
    downloads generated images fine. Returns None if the cookies are missing.
    """
    from curl_cffi.requests import AsyncSession

    jar = getattr(gclient, "cookies", None)
    if jar is None:
        return None
    try:
        psid = jar.get("__Secure-1PSID")
        psidts = jar.get("__Secure-1PSIDTS")
    except Exception:
        return None
    if not psid:
        return None
    session = AsyncSession(impersonate="chrome", allow_redirects=True)
    session.cookies.set("__Secure-1PSID", psid, domain=".google.com")
    if psidts:
        session.cookies.set("__Secure-1PSIDTS", psidts, domain=".google.com")
    return session


async def _extract_gemini_images(result, gclient=None) -> List[dict]:
    """Media the Gemini reply carried: web images (public URL passthrough) and
    generated images (Google CDN URLs → downloaded and embedded as base64 so a
    browser can render them).

    Generated images are fetched with a minimal PSID/PSIDTS session (see
    `_minimal_image_session`) — the pooled client's full cookie jar 403s.
    """
    out: List[dict] = []
    dl_session = None
    try:
        for img in (getattr(result, "images", None) or [])[:8]:
            is_gen = isinstance(img, GeneratedImage)
            entry = {
                "title": getattr(img, "title", None),
                "alt": getattr(img, "alt", None),
                "kind": "generated" if is_gen else "web",
            }
            try:
                if is_gen:
                    if dl_session is None:
                        dl_session = _minimal_image_session(gclient)
                    saved = await img.save(
                        path=tempfile.gettempdir(),
                        client=dl_session,
                        full_size=False,
                    )
                    path = Path(saved)
                    data = path.read_bytes()
                    _cleanup_files([path])
                    mime = mimetypes.guess_type(saved)[0] or "image/png"
                    entry["url"] = (
                        f"data:{mime};base64," + base64.b64encode(data).decode()
                    )
                else:
                    entry["url"] = getattr(img, "url", None)
            except Exception as exc:
                logger.warning("Could not fetch a Gemini image: %s", exc)
                entry["url"] = None
                entry["error"] = f"download failed ({str(exc)[:60] or 'error'})"
            if entry.get("url") or entry.get("error"):
                out.append(entry)
        return out
    finally:
        if dl_session is not None:
            try:
                await dl_session.close()
            except Exception:
                pass


def _effective_temporary(request: ChatCompletionRequest) -> bool:
    """Whether a STATELESS chat should run as a temporary (unsaved) session.

    Only meaningful for stateless calls — a conversation (chat session) always
    persists so it can be continued, so callers apply this only when session is
    None.
    """
    return (
        request.temporary if request.temporary is not None else settings.chat_temporary
    )


async def _gemini_chat(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
) -> Tuple[str, Optional[str], List[dict]]:
    model_kwargs = {"model": model} if model.startswith("gemini-") else {}
    conv_id_in = (request.conversation_id or "").strip() or None
    try:
        client = await get_gemini_client(ctx.profile_name)
        session, send_text, conv_id, is_new, title = await _prepare_gemini_session(
            ctx, client, model_kwargs, request, prompt, conv_id_in
        )
        files = await _prepare_vision_files(request)
        logger.info(
            "chat -> Gemini (profile=%s, model=%s, conversation=%s, images=%d)",
            ctx.profile_name,
            model,
            conv_id or "stateless",
            len(files),
        )
        try:
            if session is None:
                result = await client.generate_content(
                    send_text,
                    files=files or None,
                    temporary=_effective_temporary(request),
                    **model_kwargs,
                )
            else:
                result = await session.send_message(send_text, files=files or None)
        finally:
            _cleanup_files(files)
        answer = extract_text(result)
        out_images = await _extract_gemini_images(result, client)
        if conv_id:
            await _persist_conversation(is_new, conv_id, ctx, model, session, title)
        return answer, conv_id, out_images
    except Exception as exc:
        raise _map_gemini_exception(exc, model, ctx.profile_name)


async def _gemini_chat_tools(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
) -> Tuple[str, Optional[str], Optional[List[dict]]]:
    """Prompt-emulated function calling: inject tool schemas, run non-stream
    (parsing needs the full text), parse the reply into OpenAI tool_calls.

    Returns (text, conv_id, tool_calls_or_None).
    """
    model_kwargs = {"model": model} if model.startswith("gemini-") else {}
    conv_id_in = (request.conversation_id or "").strip() or None
    instruction = build_tool_instruction(request.tools, request.tool_choice)
    try:
        client = await get_gemini_client(ctx.profile_name)
        session, send_text, conv_id, is_new, title = await _prepare_gemini_session(
            ctx, client, model_kwargs, request, prompt, conv_id_in
        )
        # Prepend the tool protocol to whatever gets sent (flattened prompt for
        # stateless, the last user message for a conversation).
        send_text = instruction + "\n\n" + send_text
        files = await _prepare_vision_files(request)
        logger.info(
            "chat -> Gemini tools (profile=%s, model=%s, tools=%d)",
            ctx.profile_name,
            model,
            len(request.tools or []),
        )
        try:
            if session is None:
                result = await client.generate_content(
                    send_text,
                    files=files or None,
                    temporary=_effective_temporary(request),
                    **model_kwargs,
                )
            else:
                result = await session.send_message(send_text, files=files or None)
        finally:
            _cleanup_files(files)
        text = extract_text(result)
        if conv_id:
            await _persist_conversation(is_new, conv_id, ctx, model, session, title)
        calls, cleaned = parse_tool_calls(text, tool_names(request.tools))
        return cleaned, conv_id, calls
    except Exception as exc:
        raise _map_gemini_exception(exc, model, ctx.profile_name)


def _sse_line(
    chunk_id: str,
    created: int,
    model: str,
    delta: dict,
    finish_reason: Optional[str] = None,
    conversation_id: Optional[str] = None,
    usage: Optional[dict] = None,
    choices: bool = True,
) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": (
            [{"index": 0, "delta": delta, "finish_reason": finish_reason}]
            if choices
            else []
        ),
    }
    if usage is not None:
        payload["usage"] = usage
    if conversation_id:
        payload["conversation_id"] = conversation_id
    return f"data: {json.dumps(payload)}\n\n"


def _delta_text(output) -> str:
    """New characters from a streamed ModelOutput (text_delta, cumulative fb)."""
    delta = getattr(output, "text_delta", None)
    if isinstance(delta, str):
        return delta
    return ""


def _estimate_usage(prompt: str, answer: str) -> dict:
    """Token estimate (~4 chars/token) — upstream exposes no real counts."""
    return {
        "prompt_tokens": max(1, len(prompt) // 4),
        "completion_tokens": max(1, len(answer) // 4),
        "total_tokens": max(2, len(prompt) // 4 + len(answer) // 4),
    }


async def _gemini_stream_response(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
) -> StreamingResponse:
    """Real token-by-token streaming for Gemini (OpenAI SSE, with usage).

    Client acquisition + session setup happen up front so auth / unknown-
    conversation errors are clean HTTP errors. The stream body then flushes
    the role chunk immediately (low TTFT) and yields content deltas as Gemini
    produces them.
    """
    model_kwargs = {"model": model} if model.startswith("gemini-") else {}
    conv_id_in = (request.conversation_id or "").strip() or None
    try:
        client = await get_gemini_client(ctx.profile_name)
        session, send_text, conv_id, is_new, title = await _prepare_gemini_session(
            ctx, client, model_kwargs, request, prompt, conv_id_in
        )
        files = await _prepare_vision_files(request)
    except Exception as exc:
        raise _map_gemini_exception(exc, model, ctx.profile_name)

    logger.info(
        "chat (stream) -> Gemini (profile=%s, model=%s, conversation=%s, images=%d)",
        ctx.profile_name,
        model,
        conv_id or "stateless",
        len(files),
    )

    async def body():
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        accumulated = ""
        errored = False
        last_output = None
        # Flush the role chunk right away so routers see the stream start
        # (low TTFT) instead of waiting for Gemini's first snapshot.
        yield _sse_line(
            chunk_id, created, model, {"role": "assistant", "content": ""},
            conversation_id=conv_id,
        )
        try:
            stream_kwargs = dict(model_kwargs)
            if session is not None:
                stream_kwargs["chat"] = session
            else:
                # Stateless -> ephemeral by default (not saved to web history).
                stream_kwargs["temporary"] = _effective_temporary(request)
            async for output in client.generate_content_stream(
                send_text, files=files or None, **stream_kwargs
            ):
                last_output = output
                delta = _delta_text(output)
                if delta:
                    accumulated += delta
                    yield _sse_line(
                        chunk_id, created, model, {"content": delta},
                        conversation_id=conv_id,
                    )
            # Emit any images the reply carried (web + generated) once the
            # text is done, as a delta with a non-standard `images` field.
            if last_output is not None:
                out_images = await _extract_gemini_images(last_output, client)
                if out_images:
                    yield _sse_line(
                        chunk_id, created, model, {"images": out_images},
                        conversation_id=conv_id,
                    )
        except Exception as exc:
            # 200 already committed — surface the error as content so the
            # caller sees why, then finish the stream.
            errored = True
            mapped = _map_gemini_exception(exc, model, ctx.profile_name)
            message = mapped.detail.get("error", {}).get("message", str(exc)) \
                if isinstance(mapped.detail, dict) else str(exc)
            logger.warning("Gemini stream error (model=%s): %s", model, message)
            if not accumulated:
                yield _sse_line(
                    chunk_id, created, model, {"content": f"[error] {message}"},
                    conversation_id=conv_id,
                )
        finally:
            _cleanup_files(files)

        if conv_id and not errored:
            try:
                await _persist_conversation(is_new, conv_id, ctx, model, session, title)
            except Exception:
                logger.warning("Failed to persist conversation %s", conv_id)

        # Usage goes ON the finish chunk (not a separate trailing chunk): many
        # routers treat finish_reason="stop" as end-of-stream and ignore any
        # later chunk, so a trailing usage chunk would be dropped → IN/OUT 0.
        usage = _estimate_usage(prompt, accumulated)
        yield _sse_line(chunk_id, created, model, {}, finish_reason="stop",
                        conversation_id=conv_id, usage=usage)
        if settings.sse_include_done:
            yield "data: [DONE]\n\n"
        logger.info(
            "stream done (model=%s): %d chars, usage=%s",
            model,
            len(accumulated),
            usage,
        )

    return StreamingResponse(body(), media_type="text/event-stream")


def _tool_stream_response(
    model: str,
    prompt: str,
    cleaned: str,
    tool_calls: Optional[List[dict]],
    conv_id: Optional[str],
) -> StreamingResponse:
    """Synthetic SSE for the stream + tools case.

    Tool-call parsing needs the full text, so the model call already ran
    non-stream; here we replay the result as a valid OpenAI stream (role →
    tool_calls OR content → finish + usage → [DONE]).
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def body():
        yield _sse_line(
            chunk_id, created, model, {"role": "assistant", "content": None},
            conversation_id=conv_id,
        )
        if tool_calls:
            delta_calls = [
                {
                    "index": i,
                    "id": tc["id"],
                    "type": "function",
                    "function": {
                        "name": tc["function"]["name"],
                        "arguments": tc["function"]["arguments"],
                    },
                }
                for i, tc in enumerate(tool_calls)
            ]
            yield _sse_line(
                chunk_id, created, model, {"tool_calls": delta_calls},
                conversation_id=conv_id,
            )
            finish = "tool_calls"
            usage = _estimate_usage(prompt, json.dumps(tool_calls))
        else:
            if cleaned:
                yield _sse_line(
                    chunk_id, created, model, {"content": cleaned},
                    conversation_id=conv_id,
                )
            finish = "stop"
            usage = _estimate_usage(prompt, cleaned)
        yield _sse_line(chunk_id, created, model, {}, finish_reason=finish,
                        conversation_id=conv_id, usage=usage)
        if settings.sse_include_done:
            yield "data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


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
