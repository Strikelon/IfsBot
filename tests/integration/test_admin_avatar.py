from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import bcrypt
import pytest
from httpx import ASGITransport, AsyncClient

from admin.deps import AdminSettings
from admin.main import create_admin_app
from data.database import create_engine, create_session_factory
from data.models import Base, User


def _settings(url, media_dir):
    return AdminSettings(
        database_url=url, bot_token="123:abc", admin_username="boss",
        admin_password_hash=bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode(),
        admin_session_secret="sek", admin_id=1, cookie_secure=False,
        admin_media_dir=str(media_dir),
    )


async def _seed(url):
    eng = create_engine(url)
    async with eng.begin() as c:
        await c.run_sync(Base.metadata.create_all)
    f = create_session_factory(eng)
    async with f() as s:
        s.add(User(telegram_user_id=1, display_name="Аня", is_active=True))
        await s.commit()
    await eng.dispose()


def _app(url, media_dir):
    app = create_admin_app(settings=_settings(url, media_dir))
    return app


async def _login(c):
    await c.post("/login", data={"username": "boss", "password": "pw"})


@pytest.mark.asyncio
async def test_avatar_returns_photo(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    await _seed(url)
    app = _app(url, tmp_path / "media")
    bot = AsyncMock()
    bot.get_user_profile_photos.return_value = SimpleNamespace(
        total_count=1, photos=[[SimpleNamespace(file_id="F")]]
    )
    bot.get_file.return_value = SimpleNamespace(file_path="photos/x.jpg")
    bot.download_file.return_value = BytesIO(b"\xff\xd8\xffJPEGDATA")
    app.state.bot = bot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/1/avatar")
        assert r.status_code == 200
        assert r.headers["content-type"] == "image/jpeg"
        assert r.content == b"\xff\xd8\xffJPEGDATA"


@pytest.mark.asyncio
async def test_avatar_404_when_no_photo(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    await _seed(url)
    app = _app(url, tmp_path / "media")
    bot = AsyncMock()
    bot.get_user_profile_photos.return_value = SimpleNamespace(total_count=0, photos=[])
    app.state.bot = bot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/1/avatar")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_avatar_cached_second_call_no_refetch(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/c.db"
    await _seed(url)
    app = _app(url, tmp_path / "media")
    bot = AsyncMock()
    bot.get_user_profile_photos.return_value = SimpleNamespace(
        total_count=1, photos=[[SimpleNamespace(file_id="F")]]
    )
    bot.get_file.return_value = SimpleNamespace(file_path="photos/x.jpg")
    bot.download_file.return_value = BytesIO(b"IMG")
    app.state.bot = bot
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        await c.get("/users/1/avatar")
        await c.get("/users/1/avatar")
    # second call served from disk cache -> get_user_profile_photos called once
    assert bot.get_user_profile_photos.await_count == 1
