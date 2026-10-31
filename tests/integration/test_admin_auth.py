import bcrypt
import pytest
from httpx import ASGITransport, AsyncClient

from admin.deps import AdminSettings
from admin.main import create_admin_app


def _settings(test_db_url: str) -> AdminSettings:
    return AdminSettings(
        database_url=test_db_url,
        bot_token="123:abc",
        admin_username="boss",
        admin_password_hash=bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode(),
        admin_session_secret="sek",
        admin_id=1,
        cookie_secure=False,
    )


@pytest.mark.asyncio
async def test_unauthenticated_redirects_to_login(tmp_path):
    app = create_admin_app(settings=_settings(f"sqlite+aiosqlite:///{tmp_path}/a.db"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/subscribers", follow_redirects=False)
        assert r.status_code == 303
        assert r.headers["location"] == "/login"


@pytest.mark.asyncio
async def test_login_sets_cookie_and_bad_password_rejected(tmp_path):
    app = create_admin_app(settings=_settings(f"sqlite+aiosqlite:///{tmp_path}/a.db"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        bad = await c.post("/login", data={"username": "boss", "password": "nope"}, follow_redirects=False)
        assert bad.status_code == 401
        ok = await c.post("/login", data={"username": "boss", "password": "pw"}, follow_redirects=False)
        assert ok.status_code == 303
        assert "admin_session" in ok.cookies


@pytest.mark.asyncio
async def test_health_is_public(tmp_path):
    app = create_admin_app(settings=_settings(f"sqlite+aiosqlite:///{tmp_path}/a.db"))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        r = await c.get("/health")
        assert r.status_code == 200
