"""Per-account Microsoft Copilot client pool.

One `SessionCopilotClient` per profile, lazily created and pooled. The upstream
library is **synchronous** and its chat socket does not tolerate concurrent
conversations from one account, so every call runs in a worker thread
(`asyncio.to_thread`) under a **per-account lock** (serialized). Clearance and
auth failures are mapped to clean OpenAI-shaped HTTP errors.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional, Tuple

from fastapi import HTTPException

from .config import settings
from .copilot_auth import run_copilot_login, session_dir, session_exists
from .copilot_lib import (
    ChatReply,
    ClearanceRequired,
    ImageResponse,
    SessionCopilotClient,
)
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.copilot-pool")

copilot_clients: Dict[str, SessionCopilotClient] = {}
copilot_locks: Dict[str, asyncio.Lock] = {}
_init_lock = asyncio.Lock()

_SENTINEL = object()

_CLEARANCE_HELP = (
    "Copilot's Cloudflare clearance has expired and could not be refreshed "
    "headlessly. Re-run the Copilot login from the /admin dashboard (Status "
    "page) on a machine with a display, then retry."
)


def pooled_profiles() -> List[str]:
    return sorted(copilot_clients)


def _account_lock(name: str) -> asyncio.Lock:
    return copilot_locks.setdefault(name, asyncio.Lock())


def _map_copilot_exc(exc: Exception, name: str) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, ClearanceRequired):
        return openai_error(503, _CLEARANCE_HELP, "server_error", "clearance_required")
    if isinstance(exc, RuntimeError) and "not signed in" in str(exc).lower():
        return openai_error(
            502,
            f"Copilot profile '{name}' is not signed in or its session expired. "
            f"Log in from the /admin dashboard (Status page).",
            "api_error",
            "profile_auth_expired",
        )
    logger.exception("Copilot request failed for '%s'", name)
    return openai_error(
        502, f"Copilot request failed for '{name}': {exc}", "api_error"
    )


async def get_copilot_client(name: str) -> SessionCopilotClient:
    if name in copilot_clients:
        return copilot_clients[name]
    async with _init_lock:
        if name in copilot_clients:
            return copilot_clients[name]
        if not session_exists(name):
            raise openai_error(
                503,
                f"No Copilot session for profile '{name}'. Log in from the "
                f"/admin dashboard (Status page) first.",
                "server_error",
                "profile_not_authenticated",
            )
        logger.info("Creating Copilot client for '%s'…", name)
        client = SessionCopilotClient(
            str(session_dir(name)),
            # On a display host, interactive_clear lets an expired clearance be
            # refreshed via a browser popup + retry; on a headless VPS keep it
            # off so a stale clearance fails fast (503) instead of hanging.
            interactive_clear=settings.copilot_interactive_clear,
            headless_clear=settings.copilot_headless_clear,
        )
        copilot_clients[name] = client
        copilot_locks.setdefault(name, asyncio.Lock())
        return client


def _run_browser_turn(session_path: str, prompt: str, conversation_id):
    """One full browser chat turn (sync Playwright, in a worker thread)."""
    from .copilot_browser_chat import BrowserChatSession

    with BrowserChatSession(
        session_path, headless=settings.copilot_browser_headless
    ) as session:
        return session.chat(
            prompt,
            conversation_id,
            timeout=int(settings.copilot_browser_chat_timeout),
            idle_timeout=float(settings.copilot_browser_idle_timeout),
            first_frame_timeout=float(settings.copilot_browser_first_frame_timeout),
        )


async def _browser_chat(name: str, prompt: str, conversation_id, image) -> ChatReply:
    """Drive the chat through a headless browser (see copilot_browser_chat)."""
    if not session_exists(name):
        raise openai_error(
            503,
            f"No Copilot session for profile '{name}'. Log in from the "
            f"/admin dashboard (Status page) first.",
            "server_error",
            "profile_not_authenticated",
        )
    if image is not None:
        logger.warning("Copilot browser chat does not attach input images yet — ignoring.")
    # Lazy import keeps Playwright off the import path unless browser mode runs.
    from .copilot_browser_chat import SignInRequired

    session_path = str(session_dir(name))
    async with _account_lock(name):
        try:
            text, image_urls, conv = await asyncio.to_thread(
                _run_browser_turn, session_path, prompt, conversation_id
            )
        except SignInRequired as exc:
            text, image_urls, conv = await _reauth_and_retry(
                name, session_path, prompt, conversation_id, exc
            )
        except Exception as exc:
            raise _map_copilot_exc(exc, name)
    images = [ImageResponse(u, None, {}) for u in image_urls]
    return ChatReply(text, conv, images)


async def _reauth_and_retry(name, session_path, prompt, conversation_id, exc):
    """A headless turn hit the sign-in wall — re-auth interactively, then retry.

    When ``copilot_browser_interactive_login`` is on, open a VISIBLE browser (the
    same login the /admin dashboard runs) so the user signs in on the spot, then
    run the turn once more. Otherwise surface the mapped auth error unchanged.
    Held under the caller's per-account lock, so no other turn for this account
    races the login.
    """
    if not settings.copilot_browser_interactive_login:
        raise _map_copilot_exc(exc, name)
    logger.warning(
        "Copilot '%s' hit the sign-in wall — opening a browser to re-authenticate.",
        name,
    )
    try:
        ok = await run_copilot_login(name)
    except HTTPException:
        raise  # e.g. 409: another login is already running — surface as-is.
    except Exception:
        logger.exception("Copilot auto re-login for '%s' failed", name)
        raise _map_copilot_exc(exc, name)
    if not ok:
        # Login didn't complete (timeout / window closed / no token captured).
        raise _map_copilot_exc(exc, name)
    try:
        return await asyncio.to_thread(
            _run_browser_turn, session_path, prompt, conversation_id
        )
    except Exception as retry_exc:
        raise _map_copilot_exc(retry_exc, name)


async def copilot_chat(
    name: str,
    prompt: str,
    conversation_id: Optional[str] = None,
    image: Optional[bytes] = None,
) -> ChatReply:
    """Full (buffered) reply, serialized per account."""
    if settings.copilot_chat_mode == "browser":
        return await _browser_chat(name, prompt, conversation_id, image)
    client = await get_copilot_client(name)
    kwargs = {"image": image} if image is not None else {}
    async with _account_lock(name):
        try:
            return await asyncio.to_thread(
                client.chat, prompt, conversation_id, **kwargs
            )
        except Exception as exc:
            raise _map_copilot_exc(exc, name)


async def copilot_stream(
    name: str,
    prompt: str,
    conversation_id: Optional[str] = None,
    image: Optional[bytes] = None,
) -> AsyncIterator[Tuple[str, object]]:
    """Yield ('text', str) / ('image', ImageResponse) then a final
    ('conversation_id', str). The per-account lock is held for the whole stream
    (released when this generator is closed), serializing the account."""
    if settings.copilot_chat_mode == "browser":
        # Browser turns are buffered (one thread per turn); replay as one chunk.
        reply = await _browser_chat(name, prompt, conversation_id, image)
        if reply.text:
            yield ("text", reply.text)
        for img in reply.images:
            yield ("image", img)
        yield ("conversation_id", reply.conversation_id)
        return

    client = await get_copilot_client(name)
    kwargs = {"image": image} if image is not None else {}
    async with _account_lock(name):
        try:
            stream_obj = client.stream(prompt, conversation_id, **kwargs)
            gen = iter(stream_obj)
        except Exception as exc:
            raise _map_copilot_exc(exc, name)
        while True:
            try:
                item = await asyncio.to_thread(next, gen, _SENTINEL)
            except Exception as exc:
                raise _map_copilot_exc(exc, name)
            if item is _SENTINEL:
                break
            if isinstance(item, str):
                if item:
                    yield ("text", item)
            elif isinstance(item, ImageResponse):
                yield ("image", item)
        yield ("conversation_id", stream_obj.conversation_id)


async def delete_conversation(name: str, conversation_id: Optional[str]) -> None:
    """Best-effort delete of an upstream conversation (ephemeral/temporary turn).

    Never raises — an unremoved conversation is a documented limitation, not an
    error. Serialized per account like every other upstream call.
    """
    if not conversation_id:
        return
    try:
        client = await get_copilot_client(name)
        async with _account_lock(name):
            ok = await asyncio.to_thread(client.delete_conversation, conversation_id)
        logger.info(
            "Copilot ephemeral: delete conversation %s for '%s' -> %s",
            conversation_id,
            name,
            "ok" if ok else "not removed",
        )
    except Exception as exc:
        logger.warning(
            "Copilot conversation delete failed (%s, %s): %s",
            name,
            conversation_id,
            exc,
        )


def invalidate_profile(name: str) -> None:
    """Drop the pooled client (e.g. after a re-login)."""
    if copilot_clients.pop(name, None) is not None:
        logger.info("Invalidated pooled Copilot client for '%s'", name)


async def close_all() -> None:
    # CopilotClient holds no persistent socket (each turn opens/closes its own
    # curl_cffi session), so there is nothing to close — just drop references.
    copilot_clients.clear()
    copilot_locks.clear()
