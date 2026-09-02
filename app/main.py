"""MyRouter — multi-tenant, OpenAI-compatible router for 9Router.

Bridges standard OpenAI JSON payloads to:
  * NotebookLM    via `notebooklm-py`     (model = "<notebook_id>")
  * Google Gemini via `gemini_webapi`     (model = "gemini")
  * ComfyUI       via its HTTP API        (/v1/images/generations; the model
    field names a row of the `comfy_instances` table)

Google auth lives in MySQL (`google_profiles`): the first login is imported
automatically, every client init materializes auth from the DB, and rotated
cookies are synced back. Admin dashboard at /admin (SQLAdmin).

Run with:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from sqladmin import Admin
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import comfy
from .admin.auth import AdminAuth
from .admin.views import (
    ApiKeyAdmin,
    ApiPlaygroundView,
    ComfyInstanceAdmin,
    CopilotProfileAdmin,
    GoogleProfileAdmin,
    RequestLogAdmin,
    StatusView,
)
from .config import settings
from .db import engine, ensure_schema
from .google_auth import startup_auto_import, sync_profile_to_db
from .pool import close_all, pooled_profiles
from . import copilot_pool
from .routes import (
    chat,
    collections,
    comfy_api,
    copilot_api,
    gemini_api,
    images,
    models_list,
    notebooklm,
    notebooklm_api,
    research,
)

logging.basicConfig(
    level=settings.log_level.upper(),
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("ai-sidecar")


def _log_config_source() -> None:
    """Make misconfiguration diagnosable from the first startup lines."""
    from sqlalchemy.engine.url import make_url

    from .config import _ENV_FILE

    if _ENV_FILE.exists():
        logger.info("Config: .env loaded from %s", _ENV_FILE)
    else:
        logger.warning(
            "Config: %s NOT FOUND — running on defaults/OS env vars only",
            _ENV_FILE,
        )
    try:
        dsn = make_url(settings.database_url).render_as_string(hide_password=True)
    except Exception:
        dsn = "<unparseable DATABASE_URL>"
    logger.info("Config: database = %s", dsn)


async def _periodic_cookie_sync() -> None:
    """Push rotated Google cookies (file, kept fresh by notebooklm-py) to the DB."""
    while True:
        await asyncio.sleep(settings.profile_sync_interval)
        for profile in pooled_profiles():
            try:
                await sync_profile_to_db(profile)
            except Exception:
                logger.warning("Periodic cookie sync failed for '%s'", profile)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    _log_config_source()
    await ensure_schema()
    await startup_auto_import()
    comfy.init_http()
    sync_task = asyncio.create_task(_periodic_cookie_sync())
    logger.info(
        "MyRouter started (dashboard: /admin, cookie sync every %.0fs)",
        settings.profile_sync_interval,
    )
    yield

    sync_task.cancel()
    for profile in pooled_profiles():
        try:
            await sync_profile_to_db(profile)
        except Exception:
            logger.warning("Shutdown cookie sync failed for '%s'", profile)
    await close_all()
    await copilot_pool.close_all()
    await comfy.close_http()
    await engine.dispose()
    logger.info("MyRouter shut down cleanly")


app = FastAPI(title="MyRouter", version="2.0.0", lifespan=lifespan)


@app.exception_handler(StarletteHTTPException)
async def openai_exception_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """Render every HTTP error with an OpenAI-style {"error": {...}} body."""
    detail = exc.detail
    if isinstance(detail, dict) and "error" in detail:
        return JSONResponse(status_code=exc.status_code, content=detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"message": str(detail), "type": "api_error", "code": None}},
    )


# Legacy combined surface (/v1/*) — kept for playground, SDKs, old providers.
app.include_router(chat.router)
app.include_router(images.router)
app.include_router(models_list.router)
app.include_router(notebooklm.router)
app.include_router(collections.router)
app.include_router(research.router)
# Split provider surfaces for 9Router (one Base URL per backend).
app.include_router(gemini_api.router)
app.include_router(notebooklm_api.router)
app.include_router(comfy_api.router)
app.include_router(copilot_api.router)


@app.get("/healthz")
async def healthz():
    return {"status": "ok", "active_profiles": pooled_profiles()}


admin = Admin(
    app,
    engine,
    title="MyRouter Admin",
    authentication_backend=AdminAuth(secret_key=settings.secret_key),
    templates_dir=str(Path(__file__).parent / "admin" / "templates"),
)
admin.add_view(StatusView)
admin.add_view(ApiPlaygroundView)
admin.add_view(ApiKeyAdmin)
admin.add_view(GoogleProfileAdmin)
admin.add_view(CopilotProfileAdmin)
admin.add_view(ComfyInstanceAdmin)
admin.add_view(RequestLogAdmin)
