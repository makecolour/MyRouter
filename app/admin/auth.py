"""Session-based auth for the /admin dashboard (credentials from env)."""

import secrets

from sqladmin.authentication import AuthenticationBackend
from starlette.requests import Request

from ..config import settings


class AdminAuth(AuthenticationBackend):
    async def login(self, request: Request) -> bool:
        form = await request.form()
        username = str(form.get("username", ""))
        password = str(form.get("password", ""))
        if secrets.compare_digest(
            username, settings.admin_username
        ) and secrets.compare_digest(password, settings.admin_password):
            request.session.update({"admin": True})
            return True
        return False

    async def logout(self, request: Request) -> bool:
        request.session.clear()
        return True

    async def authenticate(self, request: Request) -> bool:
        return bool(request.session.get("admin"))
