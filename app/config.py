"""Application settings (env vars / .env via pydantic-settings)."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Anchor .env to the project root so it loads no matter which directory the
# server is started from (real environment variables still take precedence).
_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENV_FILE), env_file_encoding="utf-8", extra="ignore"
    )

    database_url: str = "mysql+aiomysql://root:@localhost:3306/myrouter"

    # Admin dashboard (/admin)
    admin_username: str = "admin"
    admin_password: str = "change-me"
    secret_key: str = "change-me-secret-key"

    # Built-in ComfyUI workflow knobs (instances live in the comfy_instances table)
    comfy_checkpoint: str = "epicrealism_naturalSinRC1VAE.safetensors"
    comfy_negative_prompt: str = "text, watermark, lowres, bad anatomy, blurry"
    comfy_steps: int = 25
    comfy_cfg: float = 7.0
    # Flux instances (no all-in-one checkpoints — UNET + separate CLIP/VAE) get
    # their own defaults, because the SD ones are actively wrong there: Flux dev
    # carries its guidance in a FluxGuidance node and samples at CFG 1.0, so
    # comfy_cfg=7.0 would burn every image. Detected per instance, no config.
    comfy_flux_steps: int = 20
    comfy_flux_guidance: float = 3.5
    comfy_flux_sampler: str = "euler"
    comfy_flux_scheduler: str = "simple"
    comfy_poll_interval: float = 2.0
    comfy_timeout: float = 300.0
    # Used when the OpenAI-style request says size "auto" (9Router sends this).
    comfy_default_size: str = "1024x1024"
    # Self-provisioning (ComfyUI-Manager V3.41): auto-install missing nodes /
    # download missing models before generating when a workflow is supplied.
    comfy_auto_provision: bool = False
    comfy_provision_timeout: float = 1800.0  # model downloads are slow
    # Ephemeral mode: don't persist MyRouter's work on the shared ComfyUI boxes
    # (SaveImage -> PreviewImage temp output + delete the history entry).
    comfy_ephemeral: bool = False

    # Google auth lifecycle
    profile_sync_interval: float = 600.0  # file -> DB cookie sync period (seconds)
    login_timeout: float = 600.0  # max seconds for the interactive login
    notebook_keepalive: float = 600.0  # notebooklm-py cookie-rotation keepalive
    # Headless re-auth: with a master token stored for the profile,
    # notebooklm-py can mint fresh Google cookies WITHOUT opening a browser.
    # Upstream defaults this off ("never auto-fires by default"); we default it
    # ON because a profile with no stored token is unaffected — the L4 rung
    # simply finds nothing and the ladder falls through to the browser login.
    notebook_allow_headless: bool = True
    # Base interactive-login command; empty -> "<python> -m notebooklm login".
    # Per-machine override, e.g.: LOGIN_COMMAND=notebooklm login --browser msedge
    # ("--storage <profile path>" is always appended to target the profile.)
    login_command: str = ""
    # When Google auth expires during an API call, automatically run the login
    # subprocess (usually completes silently via the profile's persistent
    # browser session) and retry the call once.
    auto_relogin: bool = True
    auto_relogin_wait: float = 60.0  # seconds an API call waits for the login

    # Emit the OpenAI "data: [DONE]" SSE terminator. Standard clients expect
    # it; set false if a downstream router (e.g. 9Router) leaks the [DONE]
    # sentinel into its non-stream aggregated JSON.
    sse_include_done: bool = True

    # Microsoft Copilot (/copilot/v1), one session dir per account under this root.
    copilot_session_root: str = "copilot_sessions"
    # Base interactive-login command; empty -> "<python> scripts/copilot_login.py".
    # The account's session dir is always appended as the final argument.
    copilot_login_command: str = ""
    copilot_login_timeout: float = 600.0
    # When a chat hits an expired Cloudflare clearance, refresh it automatically:
    # - copilot_interactive_clear: open a VISIBLE browser mid-request to re-clear
    #   and retry (the library's default recovery). Enable on a host WITH A
    #   DISPLAY — the request blocks ~30s while a browser pops up, passes the
    #   check, and retries. Leave OFF on a headless VPS (it would hang the
    #   request), where a stale clearance fails fast with 503 clearance_required.
    # - copilot_headless_clear: try a HEADLESS refresh first (silent, no popup)
    #   before the visible one. Unreliable on datacenter/VPN IPs.
    copilot_interactive_clear: bool = False
    copilot_headless_clear: bool = False
    # Chat transport. "browser" (default) drives the actual chat through a
    # headless Playwright browser — the only mode that works where the host's
    # curl_cffi TLS can't reuse the browser-earned Cloudflare clearance (the
    # pure-HTTP driver 503s every turn). "http" uses the vendored curl_cffi
    # driver (faster, but needs a curl_cffi impersonation whose JA3 Cloudflare
    # honors for this account).
    copilot_chat_mode: str = "browser"  # "browser" | "http"
    copilot_browser_headless: bool = True
    copilot_browser_chat_timeout: float = 120.0
    # Browser turns terminate on the chat socket's `done` frame; these two guards
    # keep a turn from waiting out the whole `chat_timeout` when that frame is
    # missed — the cause of multi-minute "completions" that are really just the
    # read loop stalling until the ceiling:
    #   * idle_timeout: once the reply has STARTED streaming, stop after this many
    #     seconds with no new frame. Copilot streams tokens sub-second, so a gap
    #     this long means the turn finished (or wedged) even without a `done`.
    #   * first_frame_timeout: if NO reply frame arrives within this many seconds
    #     of sending, the composer submit likely no-op'd (a large prompt can set
    #     the composer value without the SPA registering it) — re-send once if the
    #     prompt is still sitting unsent, otherwise give up instead of hanging.
    copilot_browser_idle_timeout: float = 15.0
    copilot_browser_first_frame_timeout: float = 45.0
    # When a browser chat turn lands on Copilot's sign-in wall (the profile's
    # session expired), open a VISIBLE Playwright window so the user can re-auth
    # right then, then retry the turn — instead of failing with a "log in from
    # the dashboard" 502. Enable on a host WITH A DISPLAY; leave OFF on a
    # headless VPS (launching a visible browser there errors). Reuses the same
    # interactive login the /admin dashboard runs.
    copilot_browser_interactive_login: bool = False

    # Vision: max size (MB) per input image; larger -> 400.
    vision_max_image_mb: float = 20.0
    # Function calling for Gemini is prompt-EMULATED (the web backend has no
    # native tool API). False -> ignore `tools` and answer as plain chat.
    tool_emulation: bool = True
    # A tool turn can't stream real deltas (parsing needs the whole reply before
    # we know whether it's content or tool_calls), so the connection is held open
    # while the model works. Emit a keepalive every N seconds so no proxy or
    # router reads the silence as a dead socket — this is what turned agentic
    # turns into "502 fetch connect timeout" at 9Router.
    sse_keepalive_interval: float = 10.0
    # Keepalives are SSE comment lines (": keepalive"), which the spec says
    # parsers ignore, so they can't pollute content or the derived usage. Set
    # false to send empty-delta chunks instead, for a router that mishandles them.
    sse_keepalive_comment: bool = True
    # Retry once when Google aborts a generation — but ONLY for the aborts
    # gemini_webapi does not already retry itself. See _is_transient_upstream,
    # which splits them by exception class.
    gemini_retry_transient: bool = True
    # How many times gemini_webapi's own @running ladder may re-send a failed
    # generation, passed per call as `current_retry`. Its default is 5, i.e. six
    # sends of the ENTIRE prompt with 5+10+15+20+25s of back-off between them.
    # An agentic client's prompt is >100 KB, so the default costs minutes of
    # latency and six times the account quota for an error a re-send will not
    # change.
    #
    # 0 means a single attempt, and that is deliberate: the failure we actually
    # see is a request that never produced a first byte, and a re-send pays the
    # same prefill cost with the same odds. One patient attempt (see
    # gemini_watchdog_timeout) beats two impatient ones.
    gemini_generate_retries: int = 0
    # Hard ceiling for one Gemini turn including the retry — keeps a wedged call
    # from outliving the caller's own timeout in silence.
    gemini_turn_timeout: float = 300.0
    # Passed to GeminiClient.init().
    #
    # `watchdog_timeout` is a FIRST-BYTE deadline, not just an idle-socket one.
    # The library picks `timeout if (is_thinking or is_queueing) else
    # min(timeout, watchdog_timeout)`, and both flags start False
    # (client.py:1558) — they only flip once a frame arrives. So a request still
    # in prefill has not signalled thinking, does not qualify for `timeout`, and
    # is killed at watchdog_timeout + 5.
    #
    # This was 45s on the reasoning that a short watchdog only shortens zombie
    # detection. It does not: it was cutting off live requests. Measured on the
    # deployment, every request that produced a first byte finished in under 8s
    # and every one that did not was killed at 50s — including prompts smaller
    # than ones that had just succeeded.
    #
    # 120s budget => ~125s to fail, inside gemini_turn_timeout. Keep it at least
    # 30s BELOW 9router's upstream timeout, or 9router gives up and retries while
    # this is still waiting, re-amplifying what the single attempt above removes.
    gemini_timeout: float = 240.0
    gemini_watchdog_timeout: float = 120.0
    # Ask gemini_webapi to accumulate raw response frames and dump them when a
    # stream suspends. The only way to see what an unknown error code actually
    # is — its ErrorCode enum knows 1013/1037/1050/1052/1060, and we keep getting
    # 1096 and 1155. Off by default: it logs whole raw responses.
    gemini_verbose: bool = False
    # Cap on a tool's OWN description in the emulated tool-calling prompt. The
    # nested _DESC_LIMIT never applied to it, so agentic clients were shipping
    # every word upstream — 56 KB of schemas for 27 tools. 500 keeps the opening
    # "what it does" sentence, which is what drives tool choice.
    tool_desc_limit: int = 500
    # When the model ignores the tool_calls contract and answers with a stub,
    # re-ask once with a contract-only nudge before giving up.
    tool_repair_retry: bool = True

    # Ephemeral chat by default: a stateless chat (no conversation_id) runs as a
    # TEMPORARY session that isn't saved to the provider's web history. Gemini
    # supports this natively; Copilot is best-effort. Using a conversation_id
    # forces non-temporary (a continued thread must persist).
    chat_temporary: bool = True

    log_level: str = "INFO"


settings = Settings()
