"""The single User-Agent the whole bridge presents to Cloudflare.

Cloudflare binds ``cf_clearance`` to the *exact* User-Agent string that earned
it. The bridge touches that one cookie from three places — the curl_cffi chat
driver (which *uses* it), the headless refresh, and the interactive login (which
*earn* it) — so all three must present a byte-identical UA or the clearance is
distrusted and the chat socket gates every turn behind a Cloudflare Turnstile.

Keeping the string here (imported by both :mod:`copilot.driver` and
:mod:`copilot.browser`) makes drift impossible.

MyRouter modification (self-aligning UA)
----------------------------------------
Upstream hardcodes a Windows Chrome 148 UA and asks the maintainer to bump the
version on every ``playwright install``. That drifts in TWO ways that make
Cloudflare distrust the earned clearance:

* **Version**: the constant must match the Chromium major Playwright bundles,
  else the browser's native ``Sec-CH-UA`` out-drifts the spoofed UA.
* **OS**: Playwright's ``user_agent`` override changes only the UA *string*, not
  the browser's ``Sec-CH-UA-Platform`` client hint — which stays the real OS. A
  hardcoded "Windows" UA on a Linux host is therefore internally contradictory.

So we derive the UA from the **actual host OS** and the **installed Playwright
Chromium version** at import: the spoofed UA then agrees with the browser's own
native client hints (OS + version), the login earns a trusted ``cf_clearance``,
and the driver replays a byte-identical UA. Self-heals across Playwright
upgrades — no manual bumping.

``IMPERSONATE_TARGET`` pins curl_cffi's TLS/HTTP2 fingerprint to a fixed profile;
``cf_clearance`` is bound to the UA (overridden on top), not the TLS profile
version, so it need not track Chromium exactly.

NOTE for re-vendoring: this file is one of four MyRouter-local modifications to
the vendored package (with ``browser.py`` login token-capture break and
``driver.py`` delete_conversation) — re-apply after any upstream re-sync.
"""

import json
import logging
import os
import platform

logger = logging.getLogger("ai-sidecar.copilot-ua")

# Fallback Chromium major if detection fails (upstream's pinned value).
_FALLBACK_MAJOR = 148


def _installed_chromium_major() -> int:
    """Chromium major version Playwright bundles (from its browsers.json)."""
    try:
        import playwright

        bj = os.path.join(
            os.path.dirname(playwright.__file__),
            "driver", "package", "browsers.json",
        )
        with open(bj, encoding="utf-8") as fh:
            data = json.load(fh)
        for browser in data.get("browsers", []):
            if browser.get("name") == "chromium":
                version = browser.get("browserVersion") or ""  # "149.0.7827.55"
                major = int(version.split(".")[0])
                if major > 0:
                    return major
    except Exception:
        pass
    return _FALLBACK_MAJOR


def _os_ua_platform():
    """(UA platform token, Sec-CH-UA-Platform value) for the real host OS.

    Must match the browser's native ``Sec-CH-UA-Platform`` (Playwright does not
    override it), so the spoofed UA line and the native hint agree.
    """
    system = platform.system()
    if system == "Linux":
        return "X11; Linux x86_64", "Linux"
    if system == "Darwin":
        return "Macintosh; Intel Mac OS X 10_15_7", "macOS"
    # Windows (and any unknown OS) -> the widely-common desktop presentation.
    return "Windows NT 10.0; Win64; x64", "Windows"


_MAJOR = _installed_chromium_major()
_OS_TOKEN, _PLATFORM = _os_ua_platform()

CHROME_UA = (
    f"Mozilla/5.0 ({_OS_TOKEN}) AppleWebKit/537.36 "
    f"(KHTML, like Gecko) Chrome/{_MAJOR}.0.0.0 Safari/537.36"
)

# Client hints that must accompany CHROME_UA so the platform/version a server
# reads from the hints agrees with the UA line (and with the browser's native
# hints). Used by the curl_cffi driver, which otherwise emits the impersonation
# profile's native hints.
CHROME_CLIENT_HINTS = {
    "sec-ch-ua-platform": f'"{_PLATFORM}"',
    "sec-ch-ua": (
        f'"Google Chrome";v="{_MAJOR}", "Chromium";v="{_MAJOR}", '
        f'"Not_A Brand";v="24"'
    ),
}

# Pinned curl_cffi impersonation profile (TLS/HTTP2 fingerprint). The UA itself
# is overridden on top, so the profile version need not equal _MAJOR.
IMPERSONATE_TARGET = "chrome146"

logger.info("Copilot UA resolved: %s (platform=%s)", CHROME_UA, _PLATFORM)
