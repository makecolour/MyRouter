"""Vendored `Windows-Copilot-API` bridge, adapted for multi-account server use.

The library (`third_party/windows_copilot_api/copilot/`, MIT, unmodified) ships a
single-account `CopilotClient` whose session lives in a fixed `session/` folder
under the CWD. We do NOT patch it — instead `SessionCopilotClient` subclasses it
and overrides the two path-touching methods so each Microsoft account gets its
own `<session_dir>/{token.json, profile/}`, and forces `auto_login=False` so a
pooled server client never pops an interactive browser mid-request.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Make the vendored `copilot` package importable (it has no packaging metadata).
_VENDOR = Path(__file__).resolve().parent.parent / "third_party" / "windows_copilot_api"
if str(_VENDOR) not in sys.path:
    sys.path.insert(0, str(_VENDOR))

from copilot import (  # noqa: E402  (import after sys.path tweak)
    ChatReply,
    ClearanceRequired,
    CopilotClient,
    load_auth,
)
from copilot.models import ImageResponse  # noqa: E402

__all__ = [
    "ChatReply",
    "ClearanceRequired",
    "ImageResponse",
    "SessionCopilotClient",
    "session_paths",
    "browser_login",
]


def session_paths(session_dir: str) -> Tuple[str, str]:
    """(<session_dir>/token.json, <session_dir>/profile) for one account."""
    d = Path(session_dir)
    return str(d / "token.json"), str(d / "profile")


class SessionCopilotClient(CopilotClient):
    """`CopilotClient` pinned to a per-account session dir.

    Overrides only the two methods that reach for the default `session/` paths
    (`_fresh_auth`, `_refresh_clearance`) and disables interactive auto-login —
    everything else (chat/stream/recovery) is inherited unchanged.
    """

    def __init__(self, session_dir: str, **kwargs):
        super().__init__(**kwargs)
        self._auth_path, self._profile_dir = session_paths(session_dir)

    def _fresh_auth(self) -> Optional[dict]:
        if self._anonymous:
            return None
        if self._auth is None or (
            time.time() - self._auth.get("saved_at", 0)
        ) >= self._max_age:
            # auto_login=False: a headless token refresh from the signed-in
            # profile is fine on a server, but never open a VISIBLE sign-in
            # browser from a pooled request — that path is the dashboard login.
            self._auth = load_auth(
                path=self._auth_path,
                profile_dir=self._profile_dir,
                max_age=self._max_age,
                proxy=self._proxy,
                auto_login=False,
            )
        return self._auth

    def _refresh_clearance(self, headless: bool) -> None:
        from copilot.browser import BrowserCopilot

        bot = BrowserCopilot(
            profile_dir=self._profile_dir, headless=headless, proxy=self._proxy
        )
        try:
            bot.auto_clear(path=self._auth_path)
        finally:
            bot.close()

    def delete_conversation(self, conversation_id: str) -> bool:
        """Best-effort delete of an upstream conversation (ephemeral turns)."""
        auth = self._fresh_auth()
        return self._driver.delete_conversation(
            conversation_id,
            cookies=auth["cookies"] if auth else None,
            proxy=self._proxy,
        )


def browser_login(session_dir: str, headless: bool = False) -> dict:
    """Interactive Microsoft/Google sign-in into a per-account session dir.

    Opens a visible browser (needs a display), warms up one turn to mint the
    chat token + earn Cloudflare clearance, and snapshots `token.json`. Called
    by the login subprocess (`scripts/copilot_login.py`); returns the auth dict.
    """
    from copilot.browser import BrowserCopilot

    auth_path, profile_dir = session_paths(session_dir)
    Path(profile_dir).mkdir(parents=True, exist_ok=True)
    return BrowserCopilot(profile_dir=profile_dir, headless=headless).login(
        path=auth_path
    )
