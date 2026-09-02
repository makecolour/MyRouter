"""/v1/chat/completions — Gemini (by model, optional conversations) and NotebookLM.

Routing:
  * "gemini"            -> Gemini with the account's default model
  * "gemini-*"          -> Gemini with that model (e.g. gemini-3-pro)
  * anything else       -> NotebookLM notebook id

Gemini conversations (optional): request field `conversation_id` — "new"
starts a server-side thread, an existing id continues it (only the last user
message is sent); the response carries `conversation_id` back.
"""

import asyncio
import base64
import json
import logging
import mimetypes
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from gemini_webapi import (
    ModelInvalidError,
    TemporarilyBlockedError,
    UsageLimitExceededError,
)
from gemini_webapi.exceptions import APIError as GeminiAPIError
from gemini_webapi.exceptions import GeminiError
from gemini_webapi import GeneratedImage
from gemini_webapi import TimeoutError as GeminiTimeoutError
from notebooklm import NotebookNotFoundError
from sqlalchemy import select

from ..comfy import fetch_image_bytes
from ..config import settings
from ..db import SessionLocal
from ..models import GeminiConversation, utcnow
from ..pool import (
    get_gemini_client,
    get_notebook_client,
    notebook_locks,
    record_failure,
    record_success,
)
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
    build_repair_instruction,
    build_tool_instruction,
    looks_like_a_mangled_call,
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
        if request.stream:
            # Set the turn up here (auth / bad conversation id stay clean HTTP
            # errors) but run the model INSIDE the stream body, so response
            # headers reach the caller immediately instead of after the whole
            # generation. A buffered tool turn is what routers were reading as a
            # dead socket.
            turn = await _setup_tool_turn(request, ctx, model, prompt)
            return _tool_stream_response(request, ctx, model, prompt, turn)
        turn = await _setup_tool_turn(request, ctx, model, prompt)
        cleaned, tool_calls = await _run_tool_turn(request, ctx, model, turn)
        payload = build_chat_response(model, prompt, cleaned, tool_calls=tool_calls)
        if turn.conv_id:
            payload["conversation_id"] = turn.conv_id
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


def _fresh_session(client, model_kwargs: dict, metadata=None):
    """start_chat() carrying only this conversation's metadata.

    Until gemini_webapi 2.1.0 every ChatSession shared the module-level
    DEFAULT_METADATA list and the cid/rid/rcid setters mutated it in place, so
    the first conversation's cid leaked into every later "new" session and
    threaded everything into ONE Gemini conversation. We patched the private
    slot to work around it. 2.1.0 fixed it upstream (ChatSession.__init__ now
    does ``DEFAULT_METADATA.copy()``), so plain start_chat() is safe again —
    hence the ==2.1.1 pin in requirements.txt.
    """
    session = client.start_chat(**model_kwargs)
    if metadata:
        session.metadata = metadata
    return session


def _map_gemini_exception(exc: Exception, model: str, profile: str) -> HTTPException:
    """Translate a gemini_webapi exception into an OpenAI-shaped HTTPException."""
    if isinstance(exc, HTTPException):  # our own errors pass through
        return exc
    if isinstance(exc, ModelInvalidError):
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
    if isinstance(exc, UsageLimitExceededError):
        return openai_error(
            429,
            f"Gemini usage limit exceeded for model '{model}': {exc}",
            "rate_limit_error",
            "rate_limit_exceeded",
        )
    if isinstance(exc, TemporarilyBlockedError):
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
    if _is_transient_upstream(exc):
        # 503, not 502: Google aborted this generation, the account is fine. A
        # router that reads 502 as "account dead" would otherwise blacklist a
        # healthy account and rotate to the next one for no reason.
        logger.warning(
            "Gemini aborted the generation (profile=%s, model=%s): %s",
            profile,
            model,
            exc,
        )
        return openai_error(
            503,
            f"Gemini aborted the generation upstream: "
            f"{str(exc) or type(exc).__name__}. This is a transient Google "
            f"failure — retry the request.",
            "api_error",
            "upstream_aborted",
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
                    temporary=settings.chat_temporary,
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


@dataclass
class _ToolTurn:
    """Everything a tool turn needs, resolved before the response commits."""

    client: object
    session: object
    send_text: str
    conv_id: Optional[str]
    is_new: bool
    title: Optional[str]
    files: List[Path]
    model_kwargs: dict


async def _setup_tool_turn(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, prompt: str
) -> _ToolTurn:
    """Resolve the client, session and input files for a tool turn.

    Kept separate from the model call so auth failures and unknown conversation
    ids surface as real HTTP errors — once the streaming body starts, a 200 is
    already on the wire and nothing can change that.
    """
    model_kwargs = {"model": model} if model.startswith("gemini-") else {}
    conv_id_in = (request.conversation_id or "").strip() or None
    instruction = build_tool_instruction(request.tools, request.tool_choice)
    try:
        client = await get_gemini_client(ctx.profile_name)
        session, send_text, conv_id, is_new, title = await _prepare_gemini_session(
            ctx, client, model_kwargs, request, prompt, conv_id_in
        )
        # The protocol block goes AFTER the conversation: an agentic client sends
        # tens of KB of history, and a contract stated before all of it is far
        # from the point where the model actually decides what to emit.
        send_text = send_text + "\n\n" + instruction
        files = await _prepare_vision_files(request)
    except Exception as exc:
        raise _map_gemini_exception(exc, model, ctx.profile_name)
    return _ToolTurn(
        client=client,
        session=session,
        send_text=send_text,
        conv_id=conv_id,
        is_new=is_new,
        title=title,
        files=files,
        model_kwargs=model_kwargs,
    )


def _is_transient_upstream(exc: Exception) -> bool:
    """True for Google aborting a generation mid-flight (not an account fault).

    gemini_webapi surfaces these as APIError("The original request may have been
    silently aborted"), APIError("Unknown API error code: …") and
    GeminiError("The connection to Gemini was lost …, and recovery timed out").
    All three mean "send it again", not "this account is dead".
    """
    if not isinstance(exc, (GeminiAPIError, GeminiError)):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "silently aborted",
            "unknown api error code",
            "connection to gemini was lost",
            "temporary google service issue",
        )
    )


async def _ask_gemini(turn: _ToolTurn, send_text: str):
    if turn.session is None:
        return await turn.client.generate_content(
            send_text,
            files=turn.files or None,
            temporary=settings.chat_temporary,
            **turn.model_kwargs,
        )
    return await turn.session.send_message(send_text, files=turn.files or None)


async def _run_tool_turn(
    request: ChatCompletionRequest, ctx: AuthContext, model: str, turn: _ToolTurn
) -> Tuple[str, Optional[List[dict]]]:
    """Run the model and parse the reply into OpenAI tool_calls.

    Parsing needs the whole reply before it can tell content from tool_calls, so
    this turn is genuinely non-streaming; the caller keeps the socket warm.

    Returns (text, tool_calls_or_None).
    """
    names = tool_names(request.tools)
    logger.info(
        "chat -> Gemini tools (profile=%s, model=%s, tools=%d, prompt=%d chars)",
        ctx.profile_name,
        model,
        len(request.tools or []),
        len(turn.send_text),
    )
    started = time.perf_counter()
    try:
        try:
            result = await _ask_gemini(turn, turn.send_text)
        except Exception as exc:
            if not (settings.gemini_retry_transient and _is_transient_upstream(exc)):
                raise
            logger.warning(
                "Gemini aborted the turn (model=%s, %.1fs) — retrying once: %s",
                model,
                time.perf_counter() - started,
                exc,
            )
            result = await _ask_gemini(turn, turn.send_text)

        text = extract_text(result)
        calls, cleaned = parse_tool_calls(text, names)

        # The model tried to call a tool and mangled the contract (the "OUT 8"
        # case). One terse re-ask is much cheaper than a failed agent turn.
        if calls is None and settings.tool_repair_retry and _needs_repair(
            request, text, names
        ):
            logger.info(
                "Unparseable tool call in a %d-char reply — re-asking with the "
                "contract only (model=%s)",
                len(text),
                model,
            )
            repair = await _ask_gemini(
                turn,
                build_repair_instruction(request.tools, request.tool_choice),
            )
            repair_text = extract_text(repair)
            repair_calls, repair_cleaned = parse_tool_calls(repair_text, names)
            if repair_calls:
                calls, cleaned = repair_calls, repair_cleaned
            elif not cleaned.strip():
                # No call either way. Keep the ORIGINAL unless it was empty: a
                # model just told "no prose, nothing outside the fence" with
                # nothing to call answers with a refusal, and preferring the
                # longer text (as this once did) hands the user that refusal
                # instead of the answer it already had.
                cleaned = repair_cleaned

        if turn.conv_id:
            await _persist_conversation(
                turn.is_new, turn.conv_id, ctx, model, turn.session, turn.title
            )
        logger.info(
            "tool turn done (model=%s): %.1fs, %d chars, %d tool call(s)",
            model,
            time.perf_counter() - started,
            len(cleaned or ""),
            len(calls or []),
        )
        record_success(ctx.profile_name, "gemini")
        return cleaned, calls
    except Exception as exc:
        record_failure(ctx.profile_name, "gemini", describe_error(exc))
        raise _map_gemini_exception(exc, model, ctx.profile_name)
    finally:
        _cleanup_files(turn.files)
        turn.files = []


def _needs_repair(
    request: ChatCompletionRequest, text: str, names: List[str]
) -> bool:
    """Whether a reply with no parsed call is a *failed* call worth re-asking.

    This used to be `len(text) < 200`, on the theory that a short reply was the
    stub behind the "OUT 8" case. It is not: an agentic client sends tools on
    every request, so that test fired on most normal answers — "Disk usage is
    41%.", "Done.", or a greeting — and the repair prompt ("no prose, nothing
    outside the fence") turns each of them into a refusal. Observed: "alo?" came
    back as "I cannot generate a tool-calling JSON block…" after a second 16s
    round trip.

    Brevity is not evidence. An empty reply is, and so is a tool name sitting
    next to JSON punctuation.
    """
    choice = request.tool_choice
    if isinstance(choice, str) and choice.strip().lower() == "required":
        return True
    if isinstance(choice, dict):
        return True
    if not text.strip():
        return True
    return looks_like_a_mangled_call(text, names)


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
                stream_kwargs["temporary"] = settings.chat_temporary
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


def _keepalive(chunk_id: str, created: int, model: str, conv_id: Optional[str]) -> str:
    """One heartbeat frame for a turn that has nothing to say yet.

    An SSE comment is the right tool — the spec has parsers ignore it, so it
    cannot land in content or skew the usage a router derives from the stream.
    The empty-delta form is the fallback for a router that mishandles comments.
    """
    if settings.sse_keepalive_comment:
        return ": keepalive\n\n"
    return _sse_line(chunk_id, created, model, {"content": ""}, conversation_id=conv_id)


def _tool_stream_response(
    request: ChatCompletionRequest,
    ctx: AuthContext,
    model: str,
    prompt: str,
    turn: _ToolTurn,
) -> StreamingResponse:
    """SSE for the stream + tools case, with the model call inside the body.

    Tool-call parsing needs the full reply before it can choose content vs
    tool_calls, so this cannot stream real deltas. What it can do — and what the
    old version failed to do — is commit the response headers immediately and
    keep sending bytes while the model works, so nothing between here and the
    client mistakes a slow turn for a dead connection.
    """
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    conv_id = turn.conv_id

    async def body():
        # Headers flush on the first yield — before the model is even asked.
        yield _sse_line(
            chunk_id, created, model, {"role": "assistant", "content": None},
            conversation_id=conv_id,
        )

        cleaned: str = ""
        tool_calls: Optional[List[dict]] = None
        error: Optional[str] = None

        task = asyncio.create_task(
            asyncio.wait_for(
                _run_tool_turn(request, ctx, model, turn),
                settings.gemini_turn_timeout,
            )
        )
        try:
            while True:
                done, _ = await asyncio.wait(
                    {task}, timeout=settings.sse_keepalive_interval
                )
                if done:
                    break
                yield _keepalive(chunk_id, created, model, conv_id)
            cleaned, tool_calls = await task
        except asyncio.TimeoutError:
            error = (
                f"Gemini did not finish within "
                f"{settings.gemini_turn_timeout:.0f}s."
            )
            logger.warning("Tool turn timed out (model=%s)", model)
        except Exception as exc:
            # The 200 is already committed, so an HTTP error is no longer
            # possible — report it as content and close the stream cleanly.
            mapped = _map_gemini_exception(exc, model, ctx.profile_name)
            error = (
                mapped.detail.get("error", {}).get("message", str(exc))
                if isinstance(mapped.detail, dict)
                else str(exc)
            )
            logger.warning("Tool stream error (model=%s): %s", model, error)
        finally:
            _cleanup_files(turn.files)

        if error:
            yield _sse_line(
                chunk_id, created, model, {"content": f"[error] {error}"},
                conversation_id=conv_id,
            )
            finish, usage = "stop", _estimate_usage(prompt, error)
        elif tool_calls:
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
