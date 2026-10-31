from datetime import UTC, datetime

import bcrypt
import pytest
from httpx import ASGITransport, AsyncClient

from admin.deps import AdminSettings
from admin.main import create_admin_app
from data.database import create_engine, create_session_factory
from data.models import Base, User


def _settings(url):
    return AdminSettings(
        database_url=url, bot_token="123:abc", admin_username="boss",
        admin_password_hash=bcrypt.hashpw(b"pw", bcrypt.gensalt()).decode(),
        admin_session_secret="sek", admin_id=1, cookie_secure=False,
    )


async def _seed(url):
    engine = create_engine(url)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = create_session_factory(engine)
    async with factory() as s:
        s.add(User(telegram_user_id=1, display_name="Аня", username="anya", is_active=True))
        s.add(User(telegram_user_id=2, display_name="Боб", username="bob", is_active=False))
        await s.commit()
    await engine.dispose()


async def _login(c):
    await c.post("/login", data={"username": "boss", "password": "pw"})


@pytest.mark.asyncio
async def test_subscribers_list_renders(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/subscribers")
        assert r.status_code == 200
        assert "Аня" in r.text
        assert "Боб" in r.text


@pytest.mark.asyncio
async def test_subscribers_search_filters(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/subscribers?search=anya")
        assert "Аня" in r.text
        assert "Боб" not in r.text


@pytest.mark.asyncio
async def test_interview_badge_and_date_rendered_in_list(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/tier.db"
    await _seed(url)
    engine = create_engine(url)
    factory = create_session_factory(engine)
    async with factory() as s:
        s.add(
            User(
                telegram_user_id=777,
                display_name="Ева",
                username="eva",
                is_active=True,
                access_tier="free_interview",
                access_tier_until=datetime(2026, 11, 1, tzinfo=UTC),
            )
        )
        s.add(
            User(
                telegram_user_id=778,
                display_name="Пётр",
                username="petr",
                is_active=True,
                access_tier="early_tester",
            )
        )
        await s.commit()
    await engine.dispose()

    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/subscribers")
    assert r.status_code == 200
    assert "∞" in r.text
    assert "Безлимит до 01.11.2026" in r.text
    assert "bg-amber-400" in r.text
