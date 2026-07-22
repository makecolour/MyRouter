"""Browser-driven Copilot chat.

Pure-HTTP (curl_cffi) chat is gated by Cloudflare in some environments: the JA3
of curl_cffi's newest impersonation (chrome146) can't reuse the ``cf_clearance``
a newer Chromium (149) earned, so the chat socket returns a browser-only
Turnstile challenge on every turn → 503. The **headless browser** does pass
Cloudflare (it earns clearance and completes the warm-up turn — proven live), so
we drive the actual chat there instead: navigate to Copilot, type the prompt
into the composer, and read the reply off the chat WebSocket's ``appendText``
frames.

Reuses the vendored :class:`copilot.browser.BrowserCopilot` (persistent context,
stealth launch, the ``framereceived`` WS hook, the composer selectors and the
Turnstile clicker). A whole turn runs inside ONE worker thread (sync Playwright
is thread-bound), launching the persistent profile — which holds the clearance
cookies — for that turn and closing it after.
"""

import logging
import time
from collections import deque
from typing import Iterator, List, Optional, Tuple

from . import copilot_lib  # side effect: puts the vendored `copilot` pkg on sys.path
from copilot.browser import BrowserCopilot, COPILOT_URL  # noqa: E402
from copilot.utils import drain_json  # noqa: E402

logger = logging.getLogger("ai-sidecar.copilot-browser")

_TEMP_CHAT_URL = f"{COPILOT_URL}chats/temporary"


class SignInRequired(RuntimeError):
    """The headless turn landed on Copilot's sign-in wall (session expired).

    A ``RuntimeError`` subclass whose message contains "not signed in", so the
    pool's ``_map_copilot_exc`` still maps it to a clean profile_auth_expired
    error by default — but the pool can also catch it specifically to open a
    visible browser for re-auth (copilot_browser_interactive_login)."""


class BrowserChatSession(BrowserCopilot):
    """A headless Copilot chat turn driven through a real browser.

    Use as a context manager for a single turn:

        with BrowserChatSession(session_dir) as s:
            text, image_urls, conv_id = s.chat(prompt)
    """

    def __init__(self, session_dir: str, headless: bool = True):
        _, profile_dir = copilot_lib.session_paths(session_dir)
        super().__init__(profile_dir=profile_dir, headless=headless)
        self._buffer = b""
        self._pending: deque = deque()

    # -- WS frame collection: override the warmup-only hook to KEEP frames ---
    def _on_chat_frame(self, payload) -> None:
        super()._on_chat_frame(payload)  # preserve _warmup_replied semantics
        try:
            raw = (
                payload
                if isinstance(payload, (bytes, bytearray))
                else str(payload).encode("utf-8", "ignore")
            )
        except Exception:
            return
        self._buffer += raw
        try:
            messages, self._buffer = drain_json(self._buffer)
        except Exception:
            messages, self._buffer = [], b""
        for msg in messages:
            if isinstance(msg, dict):
                self._pending.append(msg)

    # -- composer send -------------------------------------------------------
    def _send_message(self, text: str) -> bool:
        """Type the prompt into Copilot's composer and submit. Returns success."""
        for sel in ("textarea", "div[contenteditable='true']", "[role='textbox']"):
            try:
                self._page.wait_for_selector(sel, state="visible", timeout=8000)
            except Exception:
                continue
            try:
                self._page.click(sel)
                # fill() avoids a newline in the prompt triggering an early send;
                # fall back to typing if the element rejects fill().
                try:
                    self._page.fill(sel, text)
                except Exception:
                    self._page.keyboard.type(text, delay=5)
                self._page.keyboard.press("Enter")
                return True
            except Exception:
                continue
        return False

    # -- one turn ------------------------------------------------------------
    def run_turn(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        timeout: int = 120,
    ) -> Iterator[Tuple[str, object]]:
        """Yield ('text', str) / ('image', url) as the reply streams, then
        ('conversation_id', id). One turn per call."""
        self._ensure_started()
        self._install_ws_listener()
        self._buffer = b""
        self._pending.clear()
        self._warmup_replied = False

        url = f"{COPILOT_URL}chats/{conversation_id}" if conversation_id else _TEMP_CHAT_URL
        try:
            self._page.goto(url, wait_until="domcontentloaded")
        except Exception as exc:
            logger.info("Copilot chat nav retry (%s): %s", url, str(exc)[:80])
        self._page.wait_for_timeout(1500)
        self._click_turnstile(timeout_ms=1500)  # page-load gate, if any

        if not self._send_message(prompt):
            # Composer absent. Reload once — after a cold restart the SPA often
            # hasn't hydrated (or a page-load Turnstile hasn't cleared) on the
            # first paint, and the composer shows up on a second try.
            self._reload_and_settle()
            if not self._send_message(prompt):
                raise self._diagnose_missing_composer()

        deadline = time.time() + timeout
        while time.time() < deadline:
            while self._pending:
                msg = self._pending.popleft()
                event = msg.get("event")
                if event == "appendText":
                    text = msg.get("text")
                    if text:
                        yield ("text", text)
                elif event == "imageGenerated":
                    u = msg.get("url")
                    if u:
                        yield ("image", u)
                elif event == "done":
                    yield ("conversation_id", conversation_id)
                    return
                elif event == "error":
                    raise RuntimeError(
                        f"Copilot error frame: {msg.get('errorCode') or msg}"
                    )
            # Click any in-chat Turnstile that appears; pump the sync event loop
            # so framereceived handlers fire and append to _pending.
            self._click_turnstile(timeout_ms=300)
            self._page.wait_for_timeout(250)
        # Timed out — return whatever streamed.
        yield ("conversation_id", conversation_id)

    # -- composer-absent recovery / diagnosis --------------------------------
    def _reload_and_settle(self) -> None:
        """Reload the page once and re-run the page-load Turnstile gate.

        A cold restart competes for CPU, so the Copilot SPA may not have painted
        its composer within the selector timeout on first navigation; a plain
        page-load Cloudflare challenge may also still be resolving. Reloading is
        cheap and lets both settle before we conclude the composer is gone."""
        try:
            self._page.reload(wait_until="domcontentloaded")
        except Exception:
            return
        self._page.wait_for_timeout(1500)
        self._click_turnstile(timeout_ms=2000)

    def _page_text(self) -> str:
        try:
            return (self._page.evaluate(
                "() => document.body ? document.body.innerText : ''"
            ) or "").lower()
        except Exception:
            return ""

    def _diagnose_missing_composer(self) -> Exception:
        """Explain *why* the composer is absent, mapped to an actionable error.

        The pool's ``_map_copilot_exc`` turns these into clean OpenAI-shaped HTTP
        errors: a ``ClearanceRequired`` -> 503 clearance_required, and a
        RuntimeError containing "not signed in" -> 502 profile_auth_expired that
        tells the user to re-run the Copilot login from the /admin dashboard (the
        visible Playwright window). A blanket "UI changed" reached neither, so a
        signed-out session after a restart looked like a broken UI."""
        try:
            url = (self._page.url or "").lower()
        except Exception:
            url = ""
        on_login_wall = any(
            h in url
            for h in ("login.microsoftonline.com", "login.live.com", "/oauth")
        )
        try:
            signed_in = self.signed_in()
        except Exception:
            signed_in = False

        if self.region_blocked():
            return RuntimeError(
                "Copilot is not available in this session's region — route the "
                "browser through a proxy/VPN in a supported region."
            )
        if on_login_wall or not signed_in:
            # "not signed in" -> profile_auth_expired by default; the pool can
            # instead open a visible browser to re-auth (see SignInRequired).
            return SignInRequired(
                "Copilot profile is not signed in or its session expired — "
                "re-run the Copilot login from the /admin dashboard (Status page)."
            )
        if self._find_turnstile_frame() is not None or "just a moment" in self._page_text():
            return copilot_lib.ClearanceRequired(
                "Copilot is behind a Cloudflare challenge the headless browser "
                "could not pass — refresh clearance from the /admin dashboard."
            )
        return RuntimeError(
            "Copilot composer not found — the chat UI may have changed "
            f"(url={url[:120]})."
        )

    def chat(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        timeout: int = 120,
    ) -> Tuple[str, List[str], Optional[str]]:
        """Buffered turn → (text, image_urls, conversation_id)."""
        parts: List[str] = []
        images: List[str] = []
        conv = conversation_id
        for kind, val in self.run_turn(prompt, conversation_id, timeout):
            if kind == "text":
                parts.append(val)
            elif kind == "image":
                images.append(val)
            elif kind == "conversation_id":
                conv = val
        return "".join(parts), images, conv
