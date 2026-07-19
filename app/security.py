"""Bearer-key authentication (two key kinds) and request logging.

Key kinds:
  * google — bound to a Google profile; valid for Gemini + NotebookLM.
  * comfy  — bound to one ComfyUI instance; valid for image endpoints only.
"""

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Path
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select, update

from .db import SessionLocal
from .models import ApiKey, RequestLog, utcnow
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.security")

# Keep strong references to fire-and-forget log tasks until they finish.
_log_tasks: set = set()


@dataclass
class AuthContext:
    api_key: str
    kind: str  # "google" | "comfy" | "copilot"
    profile_name: Optional[str] = None
    comfy_instance: Optional[str] = None
    copilot_profile: Optional[str] = None


# HTTPBearer (instead of a raw Header param) gives Swagger UI an Authorize
# button, so /docs can test every endpoint with a pasted API key.
_bearer_scheme = HTTPBearer(auto_error=False)


async def lookup_api_key(token: str) -> Optional[AuthContext]:
    """Resolve an enabled key string to an AuthContext (None if unknown)."""
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(ApiKey).where(
                    ApiKey.key_string == token, ApiKey.enabled.is_(True)
                )
            )
        ).scalar_one_or_none()
    if row is None:
        return None
    return AuthContext(
        api_key=token,
        kind=row.key_type or "google",
        profile_name=row.profile_name,
        comfy_instance=row.comfy_instance,
        copilot_profile=row.copilot_profile,
    )


async def get_auth(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> AuthContext:
    if credentials is None or not credentials.credentials.strip():
        raise openai_error(
            401,
            "Missing or malformed Authorization header. "
            "Expected 'Authorization: Bearer <api-key>'.",
            code="invalid_api_key",
        )
    token = credentials.credentials.strip()
    ctx = await lookup_api_key(token)
    if ctx is None:
        logger.warning("Rejected request with unknown/disabled API key (…%s)", token[-4:])
        raise openai_error(401, "Invalid API key.", code="invalid_api_key")
    return ctx


async def path_google_auth(
    api_key: str = Path(..., description="A google-type API key"),
) -> AuthContext:
    """Auth for the /notebooklm/{api_key}/v1 surface: the key rides the URL.

    9Router registers this provider with the key embedded in the Base URL, so
    no Authorization header is needed. Note: path keys show up in access
    logs — acceptable on this trusted deployment; the other surfaces use
    header auth.
    """
    ctx = await lookup_api_key(api_key.strip())
    if ctx is None:
        logger.warning("Rejected path API key (…%s)", api_key[-4:])
        raise openai_error(401, "Invalid API key.", code="invalid_api_key")
    if ctx.kind != "google" or not ctx.profile_name:
        raise openai_error(
            403,
            "The NotebookLM surface requires a Google-profile API key in the "
            "URL path.",
            code="wrong_key_type",
        )
    return ctx


async def require_google_auth(
    ctx: AuthContext = Depends(get_auth),
) -> AuthContext:
    """Gemini/NotebookLM endpoints need a Google-profile key."""
    if ctx.kind != "google" or not ctx.profile_name:
        bound = f"ComfyUI instance '{ctx.comfy_instance}'" if ctx.comfy_instance else "no profile"
        raise openai_error(
            403,
            f"This API key is bound to {bound}. Chat/NotebookLM endpoints "
            f"require a Google-profile API key.",
            code="wrong_key_type",
        )
    return ctx


async def require_comfy_auth(
    ctx: AuthContext = Depends(get_auth),
) -> AuthContext:
    """Image endpoints need a per-instance ComfyUI key."""
    if ctx.kind != "comfy" or not ctx.comfy_instance:
        raise openai_error(
            403,
            "Image endpoints require a per-instance ComfyUI API key "
            "(create one in the /admin dashboard with key type 'comfy').",
            code="wrong_key_type",
        )
    return ctx


async def require_copilot_auth(
    ctx: AuthContext = Depends(get_auth),
) -> AuthContext:
    """/copilot/v1 endpoints need a Copilot-profile key."""
    if ctx.kind != "copilot" or not ctx.copilot_profile:
        raise openai_error(
            403,
            "The Copilot surface requires a Copilot-profile API key "
            "(create one in the /admin dashboard with key type 'copilot').",
            code="wrong_key_type",
        )
    return ctx


def describe_error(exc: Exception) -> str:
    """Human-readable error text for logs — never empty."""
    detail = getattr(exc, "detail", None)
    text = str(detail) if detail is not None else str(exc)
    return text or type(exc).__name__


def log_request(
    ctx: Optional[AuthContext],
    endpoint: str,
    model: Optional[str],
    status: int,
    started: float,
    error: Optional[str] = None,
) -> None:
    """Fire-and-forget insert into request_logs + API-key usage counters."""
    latency_ms = int((time.perf_counter() - started) * 1000)
    if ctx is None:
        profile = None
    elif ctx.kind == "comfy":
        profile = f"comfy:{ctx.comfy_instance}"
    elif ctx.kind == "copilot":
        profile = f"copilot:{ctx.copilot_profile}"
    else:
        profile = ctx.profile_name

    async def _write() -> None:
        try:
            async with SessionLocal() as session:
                session.add(
                    RequestLog(
                        api_key=ctx.api_key if ctx else None,
                        profile=profile,
                        endpoint=endpoint,
                        model=model,
                        status=status,
                        latency_ms=latency_ms,
                        error=(error or None) and str(error)[:2000],
                    )
                )
                if ctx is not None:
                    await session.execute(
                        update(ApiKey)
                        .where(ApiKey.key_string == ctx.api_key)
                        .values(
                            request_count=ApiKey.request_count + 1,
                            last_used_at=utcnow(),
                        )
                    )
                await session.commit()
        except Exception as exc:
            logger.warning("request log write failed: %s", exc)

    task = asyncio.create_task(_write())
    _log_tasks.add(task)
    task.add_done_callback(_log_tasks.discard)
