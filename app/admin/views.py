"""SQLAdmin views: CRUD over the tables + the custom Status dashboard."""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Type

from sqladmin import BaseView, ModelView, expose
from sqlalchemy import case, func, select
from starlette.requests import Request
from starlette.responses import RedirectResponse
from wtforms import Form, SelectField

from ..comfy import probe_instance
from ..db import SessionLocal
from ..google_auth import (
    login_in_progress,
    run_interactive_login,
    sync_profile_to_db,
)
from ..models import ApiKey, ComfyInstance, GoogleProfile, RequestLog, utcnow
from ..pool import invalidate_profile, pooled_profiles

logger = logging.getLogger("ai-sidecar.admin")

# Keep strong references to background login tasks.
_bg_tasks: set = set()

# Outcome of the most recent dashboard-triggered login per profile, shown as
# a success/failure alert on the Status page for a few minutes.
last_login_results: Dict[str, Tuple[bool, datetime]] = {}


def _spawn(coro) -> None:
    task = asyncio.create_task(coro)
    _bg_tasks.add(task)
    task.add_done_callback(_bg_tasks.discard)


def _fmt_local(dt: Optional[datetime]) -> Optional[str]:
    """DB stores naive UTC; render in the server's local timezone."""
    if dt is None:
        return None
    return (
        dt.replace(tzinfo=timezone.utc)
        .astimezone()
        .strftime("%Y-%m-%d %H:%M:%S")
    )


async def _login_and_reload(profile: str, fresh: bool = False) -> None:
    try:
        ok = await run_interactive_login(profile, fresh=fresh)
        last_login_results[profile] = (ok, utcnow())
        if ok:
            await invalidate_profile(profile)
    except Exception:
        last_login_results[profile] = (False, utcnow())
        logger.exception("Background login for '%s' failed", profile)


class ApiKeyAdmin(ModelView, model=ApiKey):
    name = "API Key"
    name_plural = "API Keys"
    icon = "fa-solid fa-key"
    column_list = [
        ApiKey.key_string,
        ApiKey.label,
        ApiKey.key_type,
        ApiKey.profile_name,
        ApiKey.comfy_instance,
        ApiKey.enabled,
        ApiKey.request_count,
        ApiKey.last_used_at,
        ApiKey.created_at,
    ]
    form_columns = [
        ApiKey.key_string,
        ApiKey.label,
        ApiKey.key_type,
        ApiKey.profile_name,
        ApiKey.comfy_instance,
        ApiKey.enabled,
    ]
    # SQLAdmin strips PK columns from forms by default — this makes
    # key_string settable on create and editable on edit.
    form_include_pk = True
    column_searchable_list = [ApiKey.key_string, ApiKey.label, ApiKey.profile_name]
    column_sortable_list = [ApiKey.request_count, ApiKey.last_used_at, ApiKey.created_at]

    async def scaffold_form(self, rules: Optional[List[str]] = None) -> Type[Form]:
        """Dynamic dropdowns: key type, Google profiles (any status), comfy instances.

        scaffold_form builds a fresh form class per request, so the choices
        stay current without a restart.
        """
        form_class = await super().scaffold_form(rules)
        async with SessionLocal() as session:
            profile_names = (
                (
                    await session.execute(
                        select(GoogleProfile.profile_name).order_by(
                            GoogleProfile.profile_name
                        )
                    )
                )
                .scalars()
                .all()
            )
            instance_names = (
                (
                    await session.execute(
                        select(ComfyInstance.name).order_by(ComfyInstance.name)
                    )
                )
                .scalars()
                .all()
            )
        form_class.key_type = SelectField(
            "Key Type",
            choices=[
                ("google", "google — Gemini + NotebookLM (profile-bound)"),
                ("comfy", "comfy — one ComfyUI instance"),
            ],
        )
        form_class.profile_name = SelectField(
            "Profile Name (google keys)",
            choices=[("", "—")] + [(name, name) for name in profile_names],
        )
        form_class.comfy_instance = SelectField(
            "ComfyUI Instance (comfy keys)",
            choices=[("", "—")] + [(name, name) for name in instance_names],
        )
        return form_class

    async def on_model_change(self, data, model, is_created, request) -> None:
        """Normalize the kind-specific fields before insert/update."""
        key_type = data.get("key_type") or "google"
        if data.get("profile_name") == "":
            data["profile_name"] = None
        if data.get("comfy_instance") == "":
            data["comfy_instance"] = None
        if key_type == "google":
            data["comfy_instance"] = None
        elif key_type == "comfy":
            data["profile_name"] = None


class GoogleProfileAdmin(ModelView, model=GoogleProfile):
    name = "Google Profile"
    name_plural = "Google Profiles"
    icon = "fa-brands fa-google"
    # Cookie material stays out of every admin page on purpose.
    column_list = [
        GoogleProfile.profile_name,
        GoogleProfile.status,
        GoogleProfile.last_login_at,
        GoogleProfile.last_synced_at,
    ]
    column_details_exclude_list = [
        GoogleProfile.storage_state,
        GoogleProfile.state_sha256,
    ]
    form_columns = [GoogleProfile.status]
    can_create = False  # profiles are created by the login flow on the Status page


class ComfyInstanceAdmin(ModelView, model=ComfyInstance):
    name = "ComfyUI Instance"
    name_plural = "ComfyUI Instances"
    icon = "fa-solid fa-image"
    column_list = [ComfyInstance.name, ComfyInstance.base_url, ComfyInstance.enabled]
    form_columns = [ComfyInstance.name, ComfyInstance.base_url, ComfyInstance.enabled]


class RequestLogAdmin(ModelView, model=RequestLog):
    name = "Request Log"
    name_plural = "Request Logs"
    icon = "fa-solid fa-list"
    can_create = False
    can_edit = False
    can_delete = True
    column_list = [
        RequestLog.created_at,
        RequestLog.endpoint,
        RequestLog.model,
        RequestLog.profile,
        RequestLog.status,
        RequestLog.latency_ms,
    ]
    column_sortable_list = [RequestLog.created_at, RequestLog.latency_ms, RequestLog.status]
    column_default_sort = ("created_at", True)
    page_size = 50


class StatusView(BaseView):
    name = "Status"
    icon = "fa-solid fa-gauge-high"

    @expose("/status", methods=["GET"])
    async def status_page(self, request: Request):
        async with SessionLocal() as session:
            profiles = (
                (
                    await session.execute(
                        select(GoogleProfile).order_by(GoogleProfile.profile_name)
                    )
                )
                .scalars()
                .all()
            )
            instances = (
                (
                    await session.execute(
                        select(ComfyInstance).order_by(ComfyInstance.name)
                    )
                )
                .scalars()
                .all()
            )
            since = utcnow() - timedelta(hours=24)
            stats = (
                await session.execute(
                    select(
                        RequestLog.model,
                        func.count(RequestLog.id),
                        func.avg(RequestLog.latency_ms),
                        func.sum(case((RequestLog.status >= 400, 1), else_=0)),
                    )
                    .where(RequestLog.created_at >= since)
                    .group_by(RequestLog.model)
                    .order_by(func.count(RequestLog.id).desc())
                )
            ).all()
            recent_errors = (
                (
                    await session.execute(
                        select(RequestLog)
                        .where(
                            RequestLog.status >= 400, RequestLog.created_at >= since
                        )
                        .order_by(RequestLog.created_at.desc())
                        .limit(10)
                    )
                )
                .scalars()
                .all()
            )

        reachability = await asyncio.gather(
            *(probe_instance(i.base_url) for i in instances)
        )
        pooled = set(pooled_profiles())

        now = utcnow()
        context = {
            "profiles": [
                {
                    "name": p.profile_name,
                    "status": p.status,
                    "pooled": p.profile_name in pooled,
                    "logging_in": login_in_progress(p.profile_name),
                    "last_login_at": _fmt_local(p.last_login_at),
                    "last_synced_at": _fmt_local(p.last_synced_at),
                }
                for p in profiles
            ],
            "login_results": [
                {"profile": name, "ok": ok, "at": _fmt_local(at)}
                for name, (ok, at) in last_login_results.items()
                if now - at < timedelta(minutes=10)
            ],
            "instances": [
                {
                    "name": i.name,
                    "base_url": i.base_url,
                    "enabled": i.enabled,
                    "reachable": ok,
                }
                for i, ok in zip(instances, reachability)
            ],
            "stats": [
                {
                    "model": model or "(none)",
                    "count": count,
                    "avg_ms": int(avg_ms or 0),
                    "errors": int(errors or 0),
                }
                for model, count, avg_ms, errors in stats
            ],
            "recent_errors": [
                {
                    "at": _fmt_local(e.created_at),
                    "endpoint": e.endpoint,
                    "status": e.status,
                    "error": e.error,
                }
                for e in recent_errors
            ],
            "login_running": login_in_progress(),
        }
        return await self.templates.TemplateResponse(request, "status.html", context)

    @expose("/status/login", methods=["POST"])
    async def action_login(self, request: Request):
        form = await request.form()
        profile = str(form.get("profile", "")).strip()
        fresh = bool(form.get("fresh"))
        if profile and not login_in_progress():
            _spawn(_login_and_reload(profile, fresh=fresh))
            logger.info(
                "Dashboard triggered login for '%s' (fresh=%s)", profile, fresh
            )
        return RedirectResponse(url="/admin/status", status_code=303)

    @expose("/status/reload", methods=["POST"])
    async def action_reload(self, request: Request):
        form = await request.form()
        profile = str(form.get("profile", "")).strip()
        if profile:
            await invalidate_profile(profile)
        return RedirectResponse(url="/admin/status", status_code=303)

    @expose("/status/sync", methods=["POST"])
    async def action_sync(self, request: Request):
        for profile in pooled_profiles():
            try:
                await sync_profile_to_db(profile)
            except Exception:
                logger.exception("Manual sync failed for '%s'", profile)
        return RedirectResponse(url="/admin/status", status_code=303)


class ApiPlaygroundView(BaseView):
    """Test the OpenAI endpoints from the admin: pick an API key, models load
    live via /v1/models with that key, then send chat/image requests."""

    name = "API Playground"
    icon = "fa-solid fa-flask"

    @expose("/playground", methods=["GET"])
    async def playground(self, request: Request):
        async with SessionLocal() as session:
            keys = (
                (
                    await session.execute(
                        select(ApiKey)
                        .where(ApiKey.enabled.is_(True))
                        .order_by(ApiKey.key_string)
                    )
                )
                .scalars()
                .all()
            )
        context = {
            "keys": [
                {
                    "key": k.key_string,
                    "label": k.label or "",
                    "kind": k.key_type or "google",
                    "target": (
                        f"comfy:{k.comfy_instance}"
                        if (k.key_type or "google") == "comfy"
                        else (k.profile_name or "?")
                    ),
                }
                for k in keys
            ]
        }
        return await self.templates.TemplateResponse(
            request, "playground.html", context
        )
