"""Per-profile client pools (NotebookLM + Gemini), lazily initialized from the DB.

The two backends are initialized independently: a NotebookLM auth failure
never blocks Gemini and vice versa. NotebookLM needs the storage file
(materialized from the DB); Gemini reads cookies straight from the DB row.

When Google auth has expired, the init path automatically runs the login
subprocess (attempt_auto_relogin — usually completes silently via the
profile's persistent browser session) and retries once.
"""

import asyncio
import logging
from typing import Awaitable, Callable, Dict, List

from fastapi import HTTPException
from gemini_webapi import AuthError, GeminiClient
from gemini_webapi.constants import AccountStatus
from gemini_webapi.utils.rotate_1psidts import clear_cookies_cache
from notebooklm import AuthError as NotebookAuthError
from notebooklm import NotebookLMClient

from .config import settings
from .google_auth import (
    attempt_auto_relogin,
    attempt_headless_reauth,
    extract_gemini_cookies,
    get_profile_row,
    import_profile_from_file,
    materialize_profile,
    set_profile_status,
    sync_profile_to_db,
)
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.pool")

notebook_clients: Dict[str, NotebookLMClient] = {}
gemini_clients: Dict[str, GeminiClient] = {}
notebook_locks: Dict[str, asyncio.Lock] = {}

# Last outcome per profile, surfaced by /healthz. Enough to tell "the host is
# unreachable" from "the host is up and Google is refusing this account" without
# reading the server's logs.
profile_health: Dict[str, dict] = {}


def record_success(profile_name: str, backend: str) -> None:
    entry = profile_health.setdefault(profile_name, {})
    entry["last_success_at"] = _now_iso()
    entry["last_backend"] = backend
    entry["last_error"] = None


def record_failure(profile_name: str, backend: str, error: str) -> None:
    entry = profile_health.setdefault(profile_name, {})
    entry["last_error_at"] = _now_iso()
    entry["last_backend"] = backend
    entry["last_error"] = error[:300]


def _now_iso() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")

# Guards pool mutation so two concurrent first-requests for the same profile
# don't both run the (expensive) initialization.
_init_lock = asyncio.Lock()


class _AuthExpired(Exception):
    """Internal marker: Google rejected the stored cookies."""


def pooled_profiles() -> List[str]:
    return sorted(set(notebook_clients) | set(gemini_clients))


def _expired_error(profile_name: str) -> HTTPException:
    if settings.auto_relogin:
        message = (
            f"Google auth for profile '{profile_name}' has expired and auto "
            f"re-login did not complete. If a browser window opened on the "
            f"server machine, finish signing in there and retry; otherwise "
            f"re-login from the /admin dashboard (Status page)."
        )
    else:
        message = (
            f"Google auth for profile '{profile_name}' has expired. "
            f"Re-login from the /admin dashboard (Status page)."
        )
    return openai_error(502, message, "api_error", "profile_auth_expired")


async def _init_notebook(profile_name: str) -> NotebookLMClient:
    """One NotebookLM init attempt. Raises _AuthExpired on dead cookies."""
    # DB -> file: auth always comes from the database.
    path = await materialize_profile(profile_name)

    # from_storage() returns an async context manager; enter it manually
    # because the client must outlive this request (pooled, closed on
    # shutdown). keepalive keeps Google cookies rotating server-side and
    # notebooklm-py's default cookie saver writes them back to `path`.
    #
    # allow_headless arms the L4 master-token rung inside the library's own
    # refresh ladder (_auth/session.py): when master_token.json sits beside
    # the storage file, a lapsed session is re-minted with NO browser and this
    # call simply succeeds — _AuthExpired is never raised, so the visible
    # browser relogin below never runs. Profiles without a token are
    # unaffected; the rung finds nothing and the ladder continues.
    try:
        return await NotebookLMClient.from_storage(
            path=str(path),
            keepalive=settings.notebook_keepalive,
            allow_headless=settings.notebook_allow_headless,
        ).__aenter__()
    except Exception as exc:
        # notebooklm-py signals a dead session two ways: AuthError (typed,
        # 0.8.x error contract) and — once the client's own L1..L4 refresh
        # ladder in _auth/session.py is exhausted — a bare ValueError whose
        # message still reads "Authentication expired". Match both; either
        # one means the stored cookies are past saving and only a re-login
        # helps, which is what _AuthExpired triggers.
        if isinstance(exc, NotebookAuthError):
            raise _AuthExpired() from exc
        if isinstance(exc, ValueError) and "Authentication expired" in str(exc):
            raise _AuthExpired() from exc
        if isinstance(exc, HTTPException):
            raise
        logger.exception(
            "NotebookLM client init failed for profile '%s'", profile_name
        )
        raise openai_error(
            502,
            f"Failed to initialize NotebookLM client for profile "
            f"'{profile_name}': {exc}",
            "api_error",
        )


async def _init_gemini(profile_name: str) -> GeminiClient:
    """One Gemini init attempt. Raises _AuthExpired on dead cookies."""
    row = await get_profile_row(profile_name)
    if row is None or not (row.storage_state or {}).get("cookies"):
        # Login may have happened outside the app — import it now.
        if await import_profile_from_file(profile_name):
            row = await get_profile_row(profile_name)
    if row is None or not (row.storage_state or {}).get("cookies"):
        raise openai_error(
            503,
            f"No Google auth stored for profile '{profile_name}'. Log in "
            f"from the /admin dashboard (Status page) first.",
            "server_error",
            "profile_not_authenticated",
        )

    psid, psidts = extract_gemini_cookies(row.storage_state)
    gemini_client = GeminiClient(secure_1psid=psid, secure_1psidts=psidts)
    # Feed Gemini the FULL browser cookie jar, not just the two cookies its
    # constructor accepts. Google treats PSID-only sessions as partially
    # authenticated (account status UNAUTHENTICATED, conversations not linked
    # to the visible web history). Every google.com cookie from the profile's
    # storage state is forwarded dynamically — nothing hardcoded.
    gemini_client.cookies = {
        c["name"]: c["value"]
        for c in row.storage_state.get("cookies", [])
        if isinstance(c, dict)
        and (c.get("domain") or "").endswith("google.com")
        and c.get("name")
        and c.get("value") is not None
    }
    # gemini_webapi tries its own cookie CACHE before the cookies we supply
    # (get_access_token phase 1) — a stale cached PSIDTS from earlier runs
    # then silently degrades the session. Drop the cache so our fresh jar is
    # the first candidate. clear_cookies_cache() is the lib's public helper
    # (2.1.x); we used to reach into the private _get_cookies_cache_path.
    try:
        clear_cookies_cache(gemini_client.cookies)
        logger.info("Cleared stale gemini cookie cache for '%s'", profile_name)
    except Exception as exc:
        logger.warning("Could not clear gemini cookie cache: %s", exc)

    try:
        # auto_refresh=False: notebooklm-py's keepalive already rotates this
        # Google session's cookies (into the storage file -> DB). A second
        # rotator here would invalidate each other's PSIDTS in a loop.
        #
        # The timeouts are ours, not the library's defaults (450/120). The
        # watchdog only governs a genuinely idle socket — a thinking or queueing
        # model gets the full `timeout` — so a short one just means a zombie
        # stream is declared dead sooner. With the defaults, a stream Google had
        # silently abandoned burned 120s of watchdog plus a 120s recovery poll
        # before raising: four minutes of a caller staring at nothing.
        await gemini_client.init(
            auto_refresh=False,
            timeout=settings.gemini_timeout,
            watchdog_timeout=settings.gemini_watchdog_timeout,
        )
    except AuthError as exc:
        raise _AuthExpired() from exc
    except Exception as exc:
        if isinstance(exc, HTTPException):
            raise
        logger.exception("Gemini client init failed for profile '%s'", profile_name)
        raise openai_error(
            502,
            f"Failed to initialize Gemini client for profile "
            f"'{profile_name}': {exc}",
            "api_error",
        )

    # The status RPC reports UNAUTHENTICATED for NotebookLM-exported cookies
    # even when generate_content works (verified live) — warn, don't fail.
    # Truly dead cookies surface as AuthError at init or request time.
    if gemini_client.account_status == AccountStatus.UNAUTHENTICATED:
        logger.warning(
            "Gemini reports account status UNAUTHENTICATED for '%s' — often a "
            "false positive with NotebookLM-exported cookies; continuing",
            profile_name,
        )
    return gemini_client


async def _init_with_relogin(
    profile_name: str, initializer: Callable[[str], Awaitable]
):
    """Run one init attempt; on expired auth, re-auth and retry once.

    Two recovery rungs, cheapest first:
      1. headless re-mint from the profile's master token — no browser, no
         display, no human. Silent and fast when a token is stored.
      2. the interactive browser login, which is what a profile with no
         master token has always used (and still needs a display).
    """
    try:
        return await initializer(profile_name)
    except _AuthExpired:
        await set_profile_status(profile_name, "expired")
        logger.warning("Google auth expired for '%s'", profile_name)
        if settings.notebook_allow_headless and await attempt_headless_reauth(
            profile_name
        ):
            await invalidate_profile(profile_name)
            return await initializer(profile_name)
        if settings.auto_relogin and await attempt_auto_relogin(profile_name):
            # Fresh cookies are in the DB — drop any stale sibling clients so
            # both backends re-init from the new auth.
            await invalidate_profile(profile_name)
            return await initializer(profile_name)
        raise


async def get_notebook_client(profile_name: str) -> NotebookLMClient:
    if profile_name in notebook_clients:
        return notebook_clients[profile_name]

    async with _init_lock:
        if profile_name in notebook_clients:
            return notebook_clients[profile_name]

        logger.info("Initializing NotebookLM client for '%s'…", profile_name)
        try:
            nb_client = await _init_with_relogin(profile_name, _init_notebook)
        except _AuthExpired:
            raise _expired_error(profile_name)

        notebook_clients[profile_name] = nb_client
        notebook_locks.setdefault(profile_name, asyncio.Lock())

        # Init may already have refreshed tokens — push them back to the DB.
        try:
            await sync_profile_to_db(profile_name)
        except Exception:
            logger.warning("Post-init cookie sync failed for '%s'", profile_name)

        logger.info("NotebookLM client ready for '%s'", profile_name)
        return nb_client


async def get_gemini_client(profile_name: str) -> GeminiClient:
    if profile_name in gemini_clients:
        return gemini_clients[profile_name]

    async with _init_lock:
        if profile_name in gemini_clients:
            return gemini_clients[profile_name]

        logger.info("Initializing Gemini client for '%s'…", profile_name)
        try:
            gemini_client = await _init_with_relogin(profile_name, _init_gemini)
        except _AuthExpired:
            raise _expired_error(profile_name)

        gemini_clients[profile_name] = gemini_client
        logger.info("Gemini client ready for '%s'", profile_name)
        return gemini_client


async def invalidate_profile(profile_name: str) -> None:
    """Close and drop pooled clients (used after a re-login)."""
    nb_client = notebook_clients.pop(profile_name, None)
    gemini_client = gemini_clients.pop(profile_name, None)
    if gemini_client is not None:
        try:
            await gemini_client.close()
        except Exception as exc:
            logger.warning("Error closing Gemini client '%s': %s", profile_name, exc)
    if nb_client is not None:
        try:
            await nb_client.close()
        except Exception as exc:
            logger.warning(
                "Error closing NotebookLM client '%s': %s", profile_name, exc
            )
    if nb_client or gemini_client:
        logger.info("Invalidated pooled clients for '%s'", profile_name)


async def close_all() -> None:
    for profile in pooled_profiles():
        await invalidate_profile(profile)
