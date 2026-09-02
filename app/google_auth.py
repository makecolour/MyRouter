"""Google auth bridge: MySQL is the source of truth, files are working copies.

Flow:
- First login (CLI or dashboard) writes storage_state.json; it is imported
  into the `google_profiles` table automatically (startup scan or post-login).
- Every client init calls `materialize_profile` (DB -> file) so auth always
  comes from the DB; notebooklm-py then keeps that file fresh via its default
  cookie saver, and `sync_profile_to_db` (file -> DB) runs after init,
  periodically, and on shutdown.
"""

import asyncio
import hashlib
import json
import logging
import os
import shlex
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple

from notebooklm.auth import bootstrap_missing_storage_from_master_token
from notebooklm.paths import (
    get_browser_profile_dir,
    get_storage_path,
    list_profiles,
    master_token_path_for,
)
from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .models import GoogleProfile, utcnow
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.google-auth")

# One interactive login at a time (it opens a browser on this host).
_login_lock = asyncio.Lock()
_logins_running: set = set()

# Auto re-login bookkeeping: profile -> monotonic time of the last attempt,
# ANY outcome. A relogin that "succeeds" but still leaves init failing must
# not respawn browser windows on every request.
_AUTO_RELOGIN_COOLDOWN = 120.0
_last_auto_relogin_attempt: Dict[str, float] = {}
_auto_relogin_tasks: set = set()


def _digest(state: dict) -> str:
    return hashlib.sha256(
        json.dumps(state, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _storage_path(profile: str) -> Path:
    try:
        return get_storage_path(profile)
    except ValueError as exc:
        raise openai_error(
            503, f"Invalid profile name '{profile}': {exc}", "server_error"
        )


def _master_token_path(profile: str) -> Path:
    """master_token.json sibling of this profile's storage file.

    Derived by the library, never by string-joining: master_token_path_for()
    is notebooklm-py's SOLE derivation site for the sibling invariant (its
    docstring records four call sites that disagreed on canonicalization
    before it existed, silently breaking headless auth for symlinked or
    relative --storage aliases).
    """
    return master_token_path_for(_storage_path(profile))


def _write_master_token(profile: str, token: dict) -> Optional[Path]:
    """DB -> file for the master token, at mode 0600.

    The token mints Google session cookies on demand, so it is strictly more
    sensitive than the cookies themselves — it is written owner-only, matching
    how notebooklm-py's own writer stores it.
    """
    path = _master_token_path(profile)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(token), encoding="utf-8")
        try:
            os.chmod(path, 0o600)
        except OSError:
            # Windows ACLs don't map onto POSIX modes; the write still stands.
            pass
        return path
    except OSError as exc:
        logger.warning("Could not write master token for '%s': %s", profile, exc)
        return None


def _read_master_token(profile: str) -> Optional[dict]:
    """File -> DB for the master token; None when absent or unreadable."""
    path = _master_token_path(profile)
    if not path.exists():
        return None
    try:
        token = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read master token for '%s': %s", profile, exc)
        return None
    # Shape check mirrors notebooklm-py's own reader (_auth/master_token_file).
    if not isinstance(token, dict) or not token.get("master_token"):
        return None
    return token


async def get_profile_row(profile: str) -> Optional[GoogleProfile]:
    async with SessionLocal() as session:
        return (
            await session.execute(
                select(GoogleProfile).where(GoogleProfile.profile_name == profile)
            )
        ).scalar_one_or_none()


async def upsert_profile(
    profile: str,
    state: dict,
    *,
    status: str = "active",
    touch_login: bool = False,
    master_token: Optional[dict] = None,
    account_email: Optional[str] = None,
) -> None:
    now = utcnow()
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(GoogleProfile).where(GoogleProfile.profile_name == profile)
            )
        ).scalar_one_or_none()
        if row is None:
            row = GoogleProfile(profile_name=profile)
            session.add(row)
        row.storage_state = state
        row.state_sha256 = _digest(state)
        row.status = status
        row.last_synced_at = now
        # Only ever ADD a token here — passing None means "unchanged", not
        # "clear it". A cookie sync must never wipe a working master token.
        if master_token is not None:
            row.master_token = master_token
        if account_email is not None:
            row.account_email = account_email
        if touch_login:
            row.last_login_at = now
        await session.commit()


async def set_profile_status(profile: str, status: str) -> None:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(GoogleProfile).where(GoogleProfile.profile_name == profile)
            )
        ).scalar_one_or_none()
        if row is None:
            row = GoogleProfile(profile_name=profile, storage_state={})
            session.add(row)
        row.status = status
        await session.commit()


async def import_profile_from_file(profile: str, *, touch_login: bool = False) -> bool:
    """File -> DB. The 'first login auto-saves to the DB' primitive."""
    path = _storage_path(profile)
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read storage state for '%s': %s", profile, exc)
        return False
    if not state.get("cookies"):
        return False
    # A --master-token login writes the durable token as a sibling; pick it up
    # in the same pass so the DB holds everything a fresh host needs. Also
    # covers a token bootstrapped outside the app (CLI) before startup scan.
    token = _read_master_token(profile)
    await upsert_profile(
        profile,
        state,
        status="active",
        touch_login=touch_login,
        master_token=token,
        account_email=(token or {}).get("email"),
    )
    logger.info(
        "Imported Google auth for profile '%s' into DB (%d cookies%s)",
        profile,
        len(state["cookies"]),
        ", + master token" if token else "",
    )
    return True


async def startup_auto_import() -> None:
    """Import every filesystem profile that the DB does not know about yet."""
    for profile in list_profiles():
        try:
            if await get_profile_row(profile) is None:
                if await import_profile_from_file(profile):
                    logger.info(
                        "First-login auto-save: profile '%s' imported from filesystem",
                        profile,
                    )
        except Exception:
            logger.exception("Auto-import failed for profile '%s'", profile)


async def materialize_profile(profile: str) -> Path:
    """DB -> file. Every client init starts here: auth always comes from the DB."""
    row = await get_profile_row(profile)
    if row is None or not (row.storage_state or {}).get("cookies"):
        # Login may have happened outside the app after startup — import it now.
        if await import_profile_from_file(profile):
            row = await get_profile_row(profile)
    if row is None or not (row.storage_state or {}).get("cookies"):
        raise openai_error(
            503,
            f"No Google auth stored for profile '{profile}'. Log in from the "
            f"/admin dashboard (Status page) first.",
            "server_error",
            "profile_not_authenticated",
        )
    path = _storage_path(profile)
    path.parent.mkdir(parents=True, exist_ok=True)

    # The master token rides along with the cookies: headless re-auth is only
    # possible on this host if the sibling file exists, and the DB is the
    # source of truth for it exactly as it is for storage_state.
    if row.master_token:
        token_path = _master_token_path(profile)
        if not token_path.exists():
            if _write_master_token(profile, row.master_token):
                logger.info(
                    "Materialized master token for '%s' -> %s",
                    profile,
                    token_path.name,
                )

    if path.exists():
        try:
            file_state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            file_state = None
        if file_state and file_state.get("cookies"):
            if _digest(file_state) == row.state_sha256:
                return path  # already in sync — nothing to write
            file_mtime = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            ).replace(tzinfo=None)
            if row.last_synced_at is None or file_mtime > row.last_synced_at:
                # The file holds NEWER rotated cookies than the DB (e.g. the
                # server stopped before a periodic sync). Clobbering it with
                # the older DB copy would make Google reject the session and
                # needlessly trigger a re-login — push file -> DB instead.
                logger.info(
                    "Storage file for '%s' is newer than the DB copy — "
                    "syncing file -> DB instead of overwriting",
                    profile,
                )
                await sync_profile_to_db(profile)
                return path

    path.write_text(json.dumps(row.storage_state), encoding="utf-8")
    logger.info("Materialized auth for '%s' from DB -> %s", profile, path)
    return path


async def sync_profile_to_db(profile: str) -> bool:
    """File -> DB, skipped when the content hash is unchanged."""
    path = _storage_path(profile)
    if not path.exists():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Sync skipped, cannot read state for '%s': %s", profile, exc)
        return False
    if not state.get("cookies"):
        return False
    row = await get_profile_row(profile)
    # A re-mint rewrites master_token.json without touching the cookie digest,
    # so the token is compared independently of the early-out below.
    token = _read_master_token(profile)
    token_is_new = token is not None and (row is None or row.master_token != token)

    if row is not None and row.state_sha256 == _digest(state) and not token_is_new:
        return False
    await upsert_profile(
        profile,
        state,
        status="active",
        master_token=token if token_is_new else None,
        account_email=(token or {}).get("email") if token_is_new else None,
    )
    cookies_changed = row is None or row.state_sha256 != _digest(state)
    logger.info(
        "Synced %s for '%s' back to DB",
        " + ".join(
            part
            for part, changed in (
                ("rotated cookies", cookies_changed),
                ("master token", token_is_new),
            )
            if changed
        ),
        profile,
    )
    return True


def extract_gemini_cookies(state: dict) -> Tuple[str, Optional[str]]:
    """Pull __Secure-1PSID / __Secure-1PSIDTS from the stored cookie array.

    gemini_webapi 2.x takes explicit cookie values (its constructor silently
    ignores unknown kwargs and would then auto-load cookies from a local
    browser — the wrong account on a multi-tenant server).
    """
    jar = {
        c.get("name"): c.get("value")
        for c in (state or {}).get("cookies", [])
        if isinstance(c, dict)
    }
    psid = jar.get("__Secure-1PSID")
    psidts = jar.get("__Secure-1PSIDTS")
    if not psid:
        raise openai_error(
            503,
            "Stored Google cookies are missing __Secure-1PSID; re-login from "
            "the /admin dashboard.",
            "server_error",
            "profile_not_authenticated",
        )
    return psid, psidts


def login_in_progress(profile: Optional[str] = None) -> bool:
    if profile is None:
        return bool(_logins_running)
    return profile in _logins_running


def _build_login_invocation(
    profile: str,
    path: Path,
    fresh: bool = False,
    *,
    master_token_email: Optional[str] = None,
) -> Tuple[list, dict]:
    """Build the interactive-login subprocess command and environment.

    The env sets NOTEBOOKLM_PROFILE so the CLI's *active profile* is the
    target one — that is what selects the persistent browser dir
    (profiles/<profile>/browser_profile). Without it, the login reuses the
    default profile's already-logged-in browser session and silently saves
    the OLD account into the new profile. A brand-new profile gets an empty
    browser dir, so Google shows a fresh login/account chooser naturally.
    """
    custom = settings.login_command.strip()
    if custom:
        # Per-machine override from .env, e.g. "notebooklm login --browser
        # msedge". posix=False keeps Windows paths intact; surrounding
        # quotes are stripped per token.
        command = [t.strip('"') for t in shlex.split(custom, posix=False)]
    else:
        command = [sys.executable, "-m", "notebooklm", "login"]
    command += ["--storage", str(path)]
    if fresh:
        # Clears the profile's cached browser session first — used to switch
        # the Google account bound to a profile (dashboard checkbox).
        command.append("--fresh")
    if master_token_email:
        # One browser sign-in bootstraps a DURABLE token; afterwards cookies
        # are minted headlessly forever. --account is mandatory with
        # --master-token (the CLI rejects it otherwise).
        #
        # --force is deliberately NOT passed: it would overwrite a token
        # belonging to a DIFFERENT Google account, so a mistyped address
        # would silently rebind the profile. Let the CLI refuse instead.
        command += ["--master-token", "--account", master_token_email]
    env = {**os.environ, "NOTEBOOKLM_PROFILE": profile}
    return command, env


async def _status_after_failed_login(
    profile: str, prior_status: Optional[str], had_cookies: bool
) -> None:
    """A failed login attempt must not hide still-usable stored auth."""
    if had_cookies and prior_status in ("active", "expired"):
        await set_profile_status(profile, prior_status)
    else:
        await set_profile_status(profile, "error")


async def run_interactive_login(
    profile: str,
    fresh: bool = False,
    *,
    master_token_email: Optional[str] = None,
) -> bool:
    """Run the notebooklm-py interactive login targeted at this profile.

    Targeting works via NOTEBOOKLM_PROFILE env + `--storage PATH` (the CLI's
    `--profile-name` flag is only valid together with `--browser-cookies`).
    Opens a browser window on this host (the server runs on the user's own
    machine). Returns True when the login finished and was saved to the DB.

    With `master_token_email`, the same one browser sign-in ALSO bootstraps a
    durable master token, after which this profile never needs a browser
    again — see attempt_headless_reauth.
    """
    path = _storage_path(profile)
    if _login_lock.locked():
        raise openai_error(
            409,
            "Another interactive login is already running; finish it first.",
            "server_error",
        )
    async with _login_lock:
        _logins_running.add(profile)
        prior = await get_profile_row(profile)
        prior_status = prior.status if prior else None
        had_cookies = bool(prior and (prior.storage_state or {}).get("cookies"))
        await set_profile_status(profile, "pending_login")

        path.parent.mkdir(parents=True, exist_ok=True)
        command, env = _build_login_invocation(
            profile, path, fresh, master_token_email=master_token_email
        )
        logger.info(
            "Starting interactive login for profile '%s': %s (NOTEBOOKLM_PROFILE=%s)",
            profile,
            " ".join(command),
            profile,
        )
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*command, env=env)
            returncode = await asyncio.wait_for(
                proc.wait(), timeout=settings.login_timeout
            )
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
            await _status_after_failed_login(profile, prior_status, had_cookies)
            logger.error(
                "Interactive login for '%s' timed out after %.0fs",
                profile,
                settings.login_timeout,
            )
            return False
        except Exception:
            await _status_after_failed_login(profile, prior_status, had_cookies)
            logger.exception("Interactive login for '%s' failed to run", profile)
            return False
        finally:
            _logins_running.discard(profile)

        if returncode != 0:
            await _status_after_failed_login(profile, prior_status, had_cookies)
            logger.error(
                "Interactive login for '%s' exited with code %s", profile, returncode
            )
            return False

        if not await import_profile_from_file(profile, touch_login=True):
            await _status_after_failed_login(profile, prior_status, had_cookies)
            logger.error(
                "Login for '%s' finished but no usable storage state was found",
                profile,
            )
            return False
        logger.info("Login for '%s' completed and saved to DB", profile)
        return True


def _login_browser_channel() -> Optional[str]:
    """Playwright channel implied by LOGIN_COMMAND (msedge/chrome/bundled)."""
    tokens = settings.login_command.lower().split()
    for candidate in ("msedge", "chrome"):
        if candidate in tokens:
            return candidate
    return None


async def gemini_warmup(profile: str) -> bool:
    """Visit gemini.google.com in the profile's browser and re-export cookies.

    Cookies exported by the NotebookLM login flow are only product-attested
    for notebooklm.google.com — Gemini then reports the session as
    UNAUTHENTICATED and does NOT link API conversations to the account's
    visible chat history (verified live; forwarding the full jar does not
    help). One visit to gemini.google.com inside the same persistent browser
    session mints gemini-attested cookies; the storage state is re-exported
    with the superset and imported into the DB.
    """
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        logger.warning("playwright is not importable — gemini warmup skipped")
        return False

    browser_dir = get_browser_profile_dir(profile)
    if not Path(browser_dir).exists():
        logger.warning(
            "No persistent browser session for '%s' — gemini warmup skipped "
            "(run a login first)",
            profile,
        )
        return False

    storage_path = _storage_path(profile)
    channel = _login_browser_channel()
    logger.info(
        "Gemini warmup for '%s' (browser=%s)…", profile, channel or "chromium"
    )
    # Launch settings mirror notebooklm-py's own (known-working) login launch:
    # anti-automation-detection args, reuse of the initial page, and retries
    # on the transient TargetClosed error persistent msedge contexts throw.
    launch_kwargs = {
        "user_data_dir": str(browser_dir),
        "headless": False,
        "args": [
            "--disable-blink-features=AutomationControlled",
            "--password-store=basic",
        ],
        "ignore_default_args": ["--enable-automation"],
    }
    if channel:
        launch_kwargs["channel"] = channel

    try:
        async with async_playwright() as p:
            context = await p.chromium.launch_persistent_context(**launch_kwargs)
            try:
                for attempt in range(1, 4):
                    page = context.pages[0] if context.pages else await context.new_page()
                    try:
                        await page.goto(
                            "https://gemini.google.com/app", timeout=30_000
                        )
                        break
                    except Exception as exc:
                        if attempt == 3:
                            raise
                        logger.info(
                            "Warmup navigation retry %d/3 (%s)",
                            attempt,
                            str(exc)[:80],
                        )
                        await asyncio.sleep(attempt)
                await asyncio.sleep(4)  # let the app settle & cookies rotate
                await context.storage_state(path=str(storage_path))
            finally:
                await context.close()
    except Exception:
        logger.exception("Gemini warmup failed for '%s'", profile)
        return False

    ok = await import_profile_from_file(profile)
    if ok:
        logger.info("Gemini warmup for '%s' completed — cookies re-exported", profile)
    return ok


async def _profile_is_fresh(profile: str) -> bool:
    row = await get_profile_row(profile)
    return bool(
        row
        and row.status == "active"
        and (row.storage_state or {}).get("cookies")
    )


async def attempt_headless_reauth(profile: str) -> bool:
    """Mint fresh Google cookies from the stored master token — no browser.

    This is the whole point of the master-token bootstrap: on a host with no
    display (or a server nobody is sitting at), a lapsed session heals itself
    instead of blocking on a browser window that will never be answered.

    Returns True when a fresh storage file landed and was synced to the DB.
    Returns False when the profile has no token, or the re-mint failed — the
    caller then falls back to the interactive browser login.
    """
    row = await get_profile_row(profile)
    if row is None or not row.master_token:
        return False

    path = _storage_path(profile)
    # The token must be on disk for the library to find it — on a fresh host
    # the DB is the only copy, and storage_state.json may not exist at all
    # (which is precisely the case this bootstrap handles).
    if not _master_token_path(profile).exists():
        if not _write_master_token(profile, row.master_token):
            return False

    logger.info("Attempting headless re-auth for '%s' from master token", profile)
    try:
        ok = await bootstrap_missing_storage_from_master_token(path)
    except Exception as exc:
        logger.warning("Headless re-auth for '%s' failed: %s", profile, exc)
        return False
    if not ok:
        logger.info("Headless re-auth for '%s' produced no new cookies", profile)
        return False

    # Fresh cookies are on disk — the DB is the source of truth, so push them
    # back before anyone re-materializes and overwrites them with the old jar.
    await sync_profile_to_db(profile)
    await set_profile_status(profile, "active")
    logger.info("Headless re-auth succeeded for '%s' (no browser opened)", profile)
    return True


async def attempt_auto_relogin(profile: str) -> bool:
    """Refresh expired auth by running the login subprocess automatically.

    Usually completes silently in a few seconds via the profile's persistent
    browser session ("Already logged in"). Returns True when fresh auth landed
    in the DB within `auto_relogin_wait`; False when the login still needs a
    human in the browser window, failed, or is on cooldown.
    """
    deadline = time.monotonic() + settings.auto_relogin_wait

    if login_in_progress():
        # A login (dashboard or another request) is already running — wait
        # for it instead of starting a second one (cooldown doesn't apply
        # to waiting).
        while login_in_progress() and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
    else:
        last_attempt = _last_auto_relogin_attempt.get(profile)
        if (
            last_attempt is not None
            and time.monotonic() - last_attempt < _AUTO_RELOGIN_COOLDOWN
        ):
            # A relogin just ran; if auth still fails, another browser window
            # won't help — a human/dashboard action is needed.
            logger.info("Auto re-login for '%s' is on cooldown", profile)
            return False
        _last_auto_relogin_attempt[profile] = time.monotonic()
        logger.info("Auth expired for '%s' — starting auto re-login", profile)
        task = asyncio.create_task(run_interactive_login(profile))
        _auto_relogin_tasks.add(task)
        task.add_done_callback(_auto_relogin_tasks.discard)
        try:
            # shield: on timeout the login keeps running (the user may still
            # finish it in the browser); this request just stops waiting.
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=max(1.0, deadline - time.monotonic()),
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Auto re-login for '%s' is still waiting for user input "
                "in the browser window",
                profile,
            )
        except Exception:
            logger.exception("Auto re-login task for '%s' errored", profile)

    if await _profile_is_fresh(profile):
        logger.info("Auto re-login for '%s' succeeded", profile)
        return True
    return False
