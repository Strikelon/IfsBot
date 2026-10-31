from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from admin.auth import (
    COOKIE_NAME,
    AuthMiddleware,
    issue_session_token,
    verify_password,
)
from admin.deps import (
    STATIC_DIR,
    AdminSettings,
    get_settings,
    make_bot,
    make_session_factory,
    templates,
)


class _SecurityHeadersMiddleware:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_wrapper(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-content-type-options", b"nosniff"))
                headers.append((b"referrer-policy", b"same-origin"))
            await send(message)

        await self.app(scope, receive, send_wrapper)


def create_admin_app(*, settings: AdminSettings | None = None) -> FastAPI:
    settings = settings or get_settings()

    if settings.sentry_dsn:
        import sentry_sdk

        sentry_sdk.init(dsn=settings.sentry_dsn.get_secret_value(), traces_sample_rate=0.0)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        from admin.scheduler import shutdown_scheduler, start_scheduler  # noqa: PLC0415

        app.state.scheduler = start_scheduler(app)
        try:
            yield
        finally:
            shutdown_scheduler(app.state.scheduler)
            await app.state.bot.session.close()

    app = FastAPI(title="Dream Admin", lifespan=lifespan)

    # Eager state init so tests that do not run the lifespan (httpx ASGITransport
    # does not enter lifespan by default) still have session_factory/bot/settings.
    # The lifespan only manages the scheduler start/stop and closes the bot.
    app.state.settings = settings
    app.state.session_factory = make_session_factory(settings)
    app.state.bot = make_bot(settings)

    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/login", response_class=HTMLResponse)
    async def login_form(request: Request):
        return templates.TemplateResponse(request, "login.html", {"error": None})

    @app.post("/login")
    async def login_submit(
        request: Request,
        username: str = Form(...),
        password: str = Form(...),
    ):
        if username == settings.admin_username and verify_password(password, settings.admin_password_hash):
            token = issue_session_token(username, secret=settings.admin_session_secret)
            resp = RedirectResponse("/subscribers", status_code=303)
            resp.set_cookie(
                COOKIE_NAME, token, httponly=True, samesite="lax",
                secure=settings.cookie_secure, max_age=settings.session_ttl_days * 86400,
            )
            return resp
        return templates.TemplateResponse(
            request, "login.html", {"error": "Неверный логин или пароль"}, status_code=401
        )

    @app.post("/logout")
    async def logout():
        resp = RedirectResponse("/login", status_code=303)
        resp.delete_cookie(COOKIE_NAME)
        return resp

    from admin.routers import subscribers as subscribers_router  # noqa: PLC0415

    app.include_router(subscribers_router.router)

    # middleware order: security headers (outer) then auth guard (inner)
    app.add_middleware(AuthMiddleware, secret=settings.admin_session_secret, max_age_days=settings.session_ttl_days)
    app.add_middleware(_SecurityHeadersMiddleware)
    return app


def __getattr__(name: str):
    if name == "app":
        return create_admin_app()
    raise AttributeError(name)
