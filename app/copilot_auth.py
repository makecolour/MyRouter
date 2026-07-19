"""Copilot account auth: filesystem sessions + interactive login subprocess.

Unlike Google auth (cookie jar in MySQL), a Copilot session is a Playwright
profile directory + token.json that isn't cleanly DB-serializable — so the
session bytes live on disk under `settings.copilot_session_root/<name>/` and the
`copilot_profiles` table stores only status/metadata. Login opens a visible
browser (needs a display) via `scripts/copilot_login.py`.
"""

import asyncio
import logging
import os
import re
import shlex
import sys
from pathlib import Path
from typing import Optional

from sqlalchemy import select

from .config import settings
from .db import SessionLocal
from .models import CopilotProfile, utcnow
from .schemas import openai_error

logger = logging.getLogger("ai-sidecar.copilot-auth")

# One interactive login at a time (it opens a browser on this host).
_login_lock = asyncio.Lock()
_logins_running: set = set()

_NAME_RE = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


def _validate_name(name: str) -> str:
    name = (name or "").strip()
    if not _NAME_RE.match(name):
        raise openai_error(
            400,
            f"Invalid Copilot profile name '{name}'. Use letters, digits, "
            f"'.', '-', '_' only.",
        )
    return name


def session_dir(name: str) -> Path:
    """Filesystem session dir for one account (validated, absolute)."""
    return (Path(settings.copilot_session_root) / _validate_name(name)).resolve()


def session_exists(name: str) -> bool:
    """True once a login has produced a token snapshot or a browser profile."""
    d = session_dir(name)
    return (d / "token.json").exists() or (d / "profile").exists()


async def get_profile_row(name: str) -> Optional[CopilotProfile]:
    async with SessionLocal() as session:
        return (
            await session.execute(
                select(CopilotProfile).where(CopilotProfile.name == name)
            )
        ).scalar_one_or_none()


async def upsert_profile(
    name: str, *, status: str, touch_login: bool = False
) -> None:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(CopilotProfile).where(CopilotProfile.name == name)
            )
        ).scalar_one_or_none()
        if row is None:
            row = CopilotProfile(name=name)
            session.add(row)
        row.status = status
        if touch_login:
            row.last_login_at = utcnow()
        await session.commit()


async def set_profile_status(name: str, status: str) -> None:
    await upsert_profile(name, status=status)


async def touch_used(name: str) -> None:
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(CopilotProfile).where(CopilotProfile.name == name)
            )
        ).scalar_one_or_none()
        if row is not None:
            row.last_used_at = utcnow()
            await session.commit()


def login_in_progress(name: Optional[str] = None) -> bool:
    if name is None:
        return bool(_logins_running)
    return name in _logins_running


def _build_login_invocation(name: str) -> list:
    """The subprocess command; the account's session dir is the last arg."""
    custom = settings.copilot_login_command.strip()
    if custom:
        command = [t.strip('"') for t in shlex.split(custom, posix=False)]
    else:
        script = Path(__file__).resolve().parent.parent / "scripts" / "copilot_login.py"
        command = [sys.executable, str(script)]
    command.append(str(session_dir(name)))
    return command


async def run_copilot_login(name: str) -> bool:
    """Run the interactive Copilot sign-in for one account.

    Opens a browser on this host (needs a display). Returns True when the login
    finished and captured a token. Mirrors google_auth.run_interactive_login's
    locking/timeout/status bookkeeping.
    """
    name = _validate_name(name)
    if _login_lock.locked():
        raise openai_error(
            409,
            "Another Copilot login is already running; finish it first.",
            "server_error",
        )
    async with _login_lock:
        _logins_running.add(name)
        await set_profile_status(name, "pending_login")
        session_dir(name).mkdir(parents=True, exist_ok=True)
        command = _build_login_invocation(name)
        logger.info("Starting Copilot login for '%s': %s", name, " ".join(command))
        proc = None
        try:
            proc = await asyncio.create_subprocess_exec(*command, env={**os.environ})
            returncode = await asyncio.wait_for(
                proc.wait(), timeout=settings.copilot_login_timeout
            )
        except asyncio.TimeoutError:
            if proc is not None:
                proc.kill()
            await set_profile_status(name, "error")
            logger.error(
                "Copilot login for '%s' timed out after %.0fs",
                name,
                settings.copilot_login_timeout,
            )
            return False
        except Exception:
            await set_profile_status(name, "error")
            logger.exception("Copilot login for '%s' failed to run", name)
            return False
        finally:
            _logins_running.discard(name)

        if returncode != 0 or not session_exists(name):
            await set_profile_status(name, "error")
            logger.error(
                "Copilot login for '%s' exited %s (session present: %s)",
                name,
                returncode,
                session_exists(name),
            )
            return False

        await upsert_profile(name, status="active", touch_login=True)
        logger.info("Copilot login for '%s' completed", name)
        return True
