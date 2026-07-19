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
    # Try a HEADLESS Cloudflare-clearance refresh on expiry before giving up.
    # Default off: headless Turnstile solving is unreliable and can hang a
    # request; off -> stale clearance fails fast with 503 clearance_required
    # (re-clear via the dashboard/CLI login on a machine with a display).
    copilot_headless_clear: bool = False

    # Vision: max size (MB) per input image; larger -> 400.
    vision_max_image_mb: float = 20.0
    # Function calling for Gemini is prompt-EMULATED (the web backend has no
    # native tool API). False -> ignore `tools` and answer as plain chat.
    tool_emulation: bool = True

    log_level: str = "INFO"


settings = Settings()
