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
            raise RuntimeError("Copilot composer not found — the chat UI may have changed.")

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
