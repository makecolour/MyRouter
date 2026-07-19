"""Interactive Copilot sign-in for one account's session dir.

Run as a subprocess by `app/copilot_auth.run_copilot_login`:

    python scripts/copilot_login.py <session_dir>

Opens a visible browser for Microsoft/Google sign-in, warms up one turn to mint
the chat token and earn Cloudflare clearance, and writes <session_dir>/token.json
(+ the persistent browser profile under <session_dir>/profile). Standalone on
purpose — it imports only the vendored `copilot` package, not the app, so it
carries no DB/config dependency.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_VENDOR = _ROOT / "third_party" / "windows_copilot_api"
sys.path.insert(0, str(_VENDOR))


def main() -> int:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print("usage: python scripts/copilot_login.py <session_dir>", file=sys.stderr)
        return 2
    session_dir = Path(sys.argv[1]).resolve()
    token_path = session_dir / "token.json"
    profile_dir = session_dir / "profile"
    profile_dir.mkdir(parents=True, exist_ok=True)

    from copilot.browser import BrowserCopilot

    auth = BrowserCopilot(profile_dir=str(profile_dir), headless=False).login(
        path=str(token_path)
    )
    if not auth.get("access_token"):
        print("Sign-in did not capture an access token.", file=sys.stderr)
        return 1
    print(f"Copilot session saved to {session_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
