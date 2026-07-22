"""/copilot/v1 — Microsoft Copilot provider surface (header copilot key).

Multi-account: the key binds to a Copilot profile. Single model `copilot`;
streaming + conversations + one input image + emulated tool-calls (like Gemini).
"""

import json
import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from ..comfy import fetch_image_bytes
from ..config import settings
from ..copilot_auth import touch_used
from ..copilot_pool import (
    copilot_chat,
    copilot_stream,
    get_copilot_client,
)
from ..copilot_pool import delete_conversation as pool_delete_conversation
from ..db import SessionLocal
from ..models import CopilotConversation, utcnow
from ..schemas import (
    ChatCompletionRequest,
    build_chat_response,
    collect_images,
    flatten_messages,
    last_user_message,
    openai_error,
)
from ..security import (
    AuthContext,
    describe_error,
    log_request,
    require_copilot_auth,
)
from ..tools import build_tool_instruction, parse_tool_calls, tool_names, tools_requested

logger = logging.getLogger("ai-sidecar.copilot-api")
router = APIRouter(prefix="/copilot/v1")

_MODEL = "copilot"


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------


def copilot_model_entries() -> List[dict]:
    return [
        {
            "id": _MODEL,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "microsoft",
        }
    ]


@router.get("/models")
async def copilot_models(ctx: AuthContext = Depends(require_copilot_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        status = 200
        return {"object": "list", "data": copilot_model_entries()}
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(ctx, "/copilot/v1/models", None, status, started, error)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _images_to_dicts(images) -> List[dict]:
    out = []
    for img in images or []:
        url = getattr(img, "url", None)
        if not url:
            continue
        out.append(
            {"url": url, "title": getattr(img, "prompt", None), "kind": "generated"}
        )
    return out


async def _input_image(request: ChatCompletionRequest) -> Optional[bytes]:
    """The current turn's first input image as bytes (Copilot accepts one)."""
    sources = collect_images(request.messages)
    if not sources:
        return None
    if len(sources) > 1:
        logger.info("Copilot accepts one input image; using the first of %d", len(sources))
    src = sources[0]
    data = src["data"] if "data" in src else await fetch_image_bytes(src["url"])
    limit = int(settings.vision_max_image_mb * 1024 * 1024)
    if len(data) > limit:
        raise openai_error(
            400,
            f"Input image exceeds the {settings.vision_max_image_mb:.0f} MB limit.",
        )
    return data


async def _maybe_delete_ephemeral(
    profile: str, persist: bool, conversation_id
) -> None:
    """Stateless Copilot turns are ephemeral by default (best-effort: the upstream
    conversation is deleted after the turn — the Copilot backend has no
    temporary-send flag). A conversation_id request persists and is never
    deleted."""
    if not persist and settings.chat_temporary:
        await pool_delete_conversation(profile, conversation_id)


async def _get_conversation(conv_id: str, profile: str) -> Optional[CopilotConversation]:
    async with SessionLocal() as db:
        return (
            await db.execute(
                select(CopilotConversation).where(
                    CopilotConversation.id == conv_id,
                    CopilotConversation.profile_name == profile,
                )
            )
        ).scalar_one_or_none()


async def _persist_conversation(
    conv_id: Optional[str], profile: str, title: Optional[str]
) -> None:
    if not conv_id:
        return
    async with SessionLocal() as db:
        row = await db.get(CopilotConversation, conv_id)
        if row is None:
            db.add(
                CopilotConversation(id=conv_id, profile_name=profile, title=title)
            )
        else:
            row.updated_at = utcnow()
            if title and not row.title:
                row.title = title
        await db.commit()


async def _resolve_conversation(
    request: ChatCompletionRequest, prompt: str, profile: str
):
    """(send_text, lib_conversation_id, persist, title).

    - no id       -> stateless: send the flattened prompt, don't persist.
    - "new"       -> send the current user message, persist the new thread.
    - existing id -> verify + continue it, send the current user message.
    """
    conv_id_in = (request.conversation_id or "").strip() or None
    if conv_id_in is None:
        return prompt, None, False, None
    message = last_user_message(request.messages) or prompt
    if conv_id_in == "new":
        return message, None, True, message[:250]
    row = await _get_conversation(conv_id_in, profile)
    if row is None:
        raise openai_error(
            404,
            f"Conversation '{conv_id_in}' was not found for this API key. "
            f"Pass conversation_id='new' to start one.",
            code="conversation_not_found",
        )
    return message, conv_id_in, True, None


# --------------------------------------------------------------------------
# Chat
# --------------------------------------------------------------------------


async def copilot_chat_dispatch(request: ChatCompletionRequest, ctx: AuthContext):
    profile = ctx.copilot_profile
    prompt = flatten_messages(request.messages)
    if not prompt:
        raise openai_error(400, "The messages array resolved to an empty prompt.")

    send_text, lib_conv, persist, title = await _resolve_conversation(
        request, prompt, profile
    )
    image = await _input_image(request)

    use_tools = (
        tools_requested(request.tools, request.tool_choice) and settings.tool_emulation
    )
    if use_tools:
        return await _copilot_tools(
            request, profile, prompt, send_text, lib_conv, image, persist, title
        )

    if request.stream:
        # Surface auth/session errors cleanly BEFORE committing a 200 stream.
        await get_copilot_client(profile)
        return _copilot_stream_response(
            profile, prompt, send_text, lib_conv, image, persist, title,
            temporary=settings.chat_temporary,
        )

    reply = await copilot_chat(profile, send_text, lib_conv, image)
    conv_id = reply.conversation_id if persist else None
    await _persist_conversation(conv_id, profile, title)
    await touch_used(profile)
    await _maybe_delete_ephemeral(profile, persist, reply.conversation_id)
    images = _images_to_dicts(reply.images)
    payload = build_chat_response(_MODEL, prompt, reply.text, images=images or None)
    if conv_id:
        payload["conversation_id"] = conv_id
    return payload


async def _copilot_tools(
    request, profile, prompt, send_text, lib_conv, image, persist, title
):
    """Prompt-emulated tool calling (buffered, like the Gemini path)."""
    instruction = build_tool_instruction(request.tools, request.tool_choice)
    reply = await copilot_chat(profile, f"{instruction}\n\n{send_text}", lib_conv, image)
    conv_id = reply.conversation_id if persist else None
    await _persist_conversation(conv_id, profile, title)
    await touch_used(profile)
    await _maybe_delete_ephemeral(profile, persist, reply.conversation_id)
    tool_calls, cleaned = parse_tool_calls(reply.text, tool_names(request.tools))
    answer = cleaned if tool_calls else reply.text
    if request.stream:
        return _tool_stream_response(prompt, answer, tool_calls, conv_id)
    payload = build_chat_response(_MODEL, prompt, answer, tool_calls=tool_calls)
    if conv_id:
        payload["conversation_id"] = conv_id
    return payload


def _sse(chunk_id, created, delta, finish=None, conv=None, usage=None) -> str:
    payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": _MODEL,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    if usage is not None:
        payload["usage"] = usage
    if conv:
        payload["conversation_id"] = conv
    return f"data: {json.dumps(payload)}\n\n"


def _usage(prompt: str, answer: str) -> dict:
    return {
        "prompt_tokens": max(1, len(prompt) // 4),
        "completion_tokens": max(1, len(answer) // 4),
        "total_tokens": max(2, len(prompt) // 4 + len(answer) // 4),
    }


def _copilot_stream_response(
    profile, prompt, send_text, lib_conv, image, persist, title, temporary=False
) -> StreamingResponse:
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    async def body():
        accumulated = ""
        images = []
        conv_id = None
        yield _sse(chunk_id, created, {"role": "assistant", "content": ""})
        try:
            async for kind, val in copilot_stream(profile, send_text, lib_conv, image):
                if kind == "text":
                    accumulated += val
                    yield _sse(chunk_id, created, {"content": val})
                elif kind == "image":
                    images.append(val)
                elif kind == "conversation_id":
                    conv_id = val
        except Exception as exc:
            message = describe_error(exc)
            logger.warning("Copilot stream error (%s): %s", profile, message)
            if not accumulated:
                yield _sse(chunk_id, created, {"content": f"[error] {message}"})
        image_dicts = _images_to_dicts(images)
        if image_dicts:
            yield _sse(chunk_id, created, {"images": image_dicts})
        final_conv = conv_id if persist else None
        if final_conv:
            try:
                await _persist_conversation(final_conv, profile, title)
                await touch_used(profile)
            except Exception:
                logger.warning("Failed to persist Copilot conversation %s", final_conv)
        elif temporary and conv_id:
            # Stateless + ephemeral: drop the upstream conversation we created.
            await pool_delete_conversation(profile, conv_id)
        yield _sse(
            chunk_id, created, {}, finish="stop",
            conv=final_conv, usage=_usage(prompt, accumulated),
        )
        if settings.sse_include_done:
            yield "data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


def _tool_stream_response(prompt, answer, tool_calls, conv_id) -> StreamingResponse:
    """Synthetic SSE replay for stream + tools (mirror the Gemini path)."""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def body():
        yield _sse(chunk_id, created, {"role": "assistant", "content": None}, conv=conv_id)
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
            yield _sse(chunk_id, created, {"tool_calls": delta_calls}, conv=conv_id)
            finish, usage = "tool_calls", _usage(prompt, json.dumps(tool_calls))
        else:
            if answer:
                yield _sse(chunk_id, created, {"content": answer}, conv=conv_id)
            finish, usage = "stop", _usage(prompt, answer)
        yield _sse(chunk_id, created, {}, finish=finish, conv=conv_id, usage=usage)
        if settings.sse_include_done:
            yield "data: [DONE]\n\n"

    return StreamingResponse(body(), media_type="text/event-stream")


@router.post("/chat/completions")
async def copilot_chat_completions(
    request: ChatCompletionRequest,
    ctx: AuthContext = Depends(require_copilot_auth),
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        response = await copilot_chat_dispatch(request, ctx)
        status = 200
        return response
    except Exception as exc:
        status = getattr(exc, "status_code", 500)
        error = describe_error(exc)
        raise
    finally:
        log_request(
            ctx, "/copilot/v1/chat/completions", request.model, status, started, error
        )


# --------------------------------------------------------------------------
# Conversations (server-side history)
# --------------------------------------------------------------------------


@router.get("/conversations")
async def list_conversations(ctx: AuthContext = Depends(require_copilot_auth)):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        async with SessionLocal() as db:
            rows = (
                await db.execute(
                    select(CopilotConversation)
                    .where(CopilotConversation.profile_name == ctx.copilot_profile)
                    .order_by(CopilotConversation.updated_at.desc())
                    .limit(100)
                )
            ).scalars().all()
        status = 200
        return {
            "object": "list",
            "data": [
                {
                    "id": row.id,
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
        log_request(ctx, "/copilot/v1/conversations", None, status, started, error)


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str, ctx: AuthContext = Depends(require_copilot_auth)
):
    started = time.perf_counter()
    status = 500
    error = None
    try:
        async with SessionLocal() as db:
            row = (
                await db.execute(
                    select(CopilotConversation).where(
                        CopilotConversation.id == conversation_id,
                        CopilotConversation.profile_name == ctx.copilot_profile,
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
        log_request(
            ctx, "/copilot/v1/conversations", conversation_id, status, started, error
        )
