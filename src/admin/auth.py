from __future__ import annotations

import bcrypt
from fastapi import Request
from fastapi.responses import RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

COOKIE_NAME = "admin_session"
_SALT = "admin-session-v1"


def verify_password(plain: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), password_hash.encode())
    except ValueError:
        return False


def _serializer(secret: str) -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(secret, salt=_SALT)


def issue_session_token(username: str, *, secret: str) -> str:
    return _serializer(secret).dumps({"u": username})


def verify_session_token(token: str, *, secret: str, max_age_days: int) -> str | None:
    try:
        data = _serializer(secret).loads(token, max_age=max_age_days * 86400)
    except (BadSignature, SignatureExpired):
        return None
    return data.get("u")


class AuthMiddleware:
    """Pure-ASGI guard: redirects unauthenticated requests to /login.
    Whitelists /login, /static, /health."""

    _PUBLIC_PREFIXES = ("/login", "/static", "/health")

    def __init__(self, app, *, secret: str, max_age_days: int):
        self.app = app
        self.secret = secret
        self.max_age_days = max_age_days

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        request = Request(scope, receive=receive)
        path = scope["path"]
        if any(path == p or path.startswith(p + "/") for p in self._PUBLIC_PREFIXES):
            await self.app(scope, receive, send)
            return
        token = request.cookies.get(COOKIE_NAME)
        username = (
            verify_session_token(token, secret=self.secret, max_age_days=self.max_age_days)
            if token
            else None
        )
        if username is None:
            response = RedirectResponse("/login", status_code=303)
            await response(scope, receive, send)
            return
        scope.setdefault("state", {})["admin_user"] = username
        await self.app(scope, receive, send)
