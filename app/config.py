"""Application settings (env vars / .env via pydantic-settings)."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
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

    log_level: str = "INFO"


settings = Settings()
