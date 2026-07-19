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
from .copilot_auth import session_dir, session_exists
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
            interactive_clear=False,  # never pop a browser from a pooled request
            headless_clear=settings.copilot_headless_clear,
        )
        copilot_clients[name] = client
        copilot_locks.setdefault(name, asyncio.Lock())
        return client


async def copilot_chat(
    name: str,
    prompt: str,
    conversation_id: Optional[str] = None,
    image: Optional[bytes] = None,
) -> ChatReply:
    """Full (buffered) reply, serialized per account."""
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


def invalidate_profile(name: str) -> None:
    """Drop the pooled client (e.g. after a re-login)."""
    if copilot_clients.pop(name, None) is not None:
        logger.info("Invalidated pooled Copilot client for '%s'", name)


async def close_all() -> None:
    # CopilotClient holds no persistent socket (each turn opens/closes its own
    # curl_cffi session), so there is nothing to close — just drop references.
    copilot_clients.clear()
    copilot_locks.clear()
