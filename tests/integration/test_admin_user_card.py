from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock

import bcrypt
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select

from admin.deps import AdminSettings
from admin.main import create_admin_app
from data.database import create_engine, create_session_factory
from data.models import (
    Base,
    Dream,
    LlmService,
    LlmUsageEvent,
    Session,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
    UsageMetric,
    User,
)


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
        u = User(telegram_user_id=1, display_name="Аня", username="anya", is_active=True)
        s.add(u)
        await s.flush()
        d = Dream(user_id=u.id, recorded_at=datetime(2026, 6, 1, tzinfo=UTC), transcription="мне снилось море")
        s.add(d)
        await s.flush()
        sess = Session(dream_id=d.id, user_id=u.id, type=SessionType.QUICK, status=SessionStatus.COMPLETED)
        s.add(sess)
        await s.flush()
        s.add(SessionMessage(session_id=sess.id, direction=SessionMessageDirection.BOT_TO_USER,
                             content_type=SessionMessageContentType.TEXT, content="разбор"))
        s.add(LlmUsageEvent(user_id=u.id, session_id=sess.id, service=list(LlmService)[0], model="m",
                            usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("0.01")))
        await s.commit()
    await engine.dispose()
    return factory


async def _login(c):
    await c.post("/login", data={"username": "boss", "password": "pw"})


@pytest.fixture(autouse=True)
def notify_interview(monkeypatch):
    """Оповещение о выдаче интервью (2026-08-24) ходит в Telegram по-настоящему.

    Автоюз: любой POST tier=interview без подмены уходит в сеть и оставляет
    незакрытую aiohttp-сессию, которая всплывает ошибкой в СОСЕДНЕМ тесте.
    """
    mock = AsyncMock(return_value=True)
    monkeypatch.setattr("admin.routers.subscribers.notify_interview_granted", mock)
    return mock


@pytest.mark.asyncio
async def test_user_card_renders(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/a.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/1")
        assert r.status_code == 200
        assert "Аня" in r.text
        # Карточка показывает только цифры: текст обращения и ответы бота не выводятся.
        assert "мне снилось море" not in r.text
        assert "разбор" not in r.text


@pytest.mark.asyncio
async def test_user_card_404_for_unknown(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/b.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/99999")
        assert r.status_code == 404


@pytest.mark.asyncio
async def test_delete_user_preserves_billing(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/e.db"
    factory = await _seed(url)

    # Seed an analytics_events row: на проде (PG) её FK -> users.id без каскада
    # ронял удаление (mig 0013 добавлена ПОСЛЕ delete-хэндлера). Хэндлер обязан
    # чистить эту таблицу перед delete(User).
    from data.analytics_events import AnalyticsEventType
    from data.models import AnalyticsEvent
    async with factory() as s:
        s.add(
            AnalyticsEvent(
                user_id=1,
                telegram_user_id=1,
                event_type=AnalyticsEventType.ONB_DREAM_TOLD,
                occurred_at=datetime(2026, 6, 1, tzinfo=UTC),
                event_day=datetime(2026, 6, 1, tzinfo=UTC).date(),
            )
        )
        await s.commit()

    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.request("DELETE", "/api/users/1")
        assert r.status_code == 200
    # user gone, dreams gone, analytics gone, но llm_usage_events сохранён с user_id NULL
    from sqlalchemy import func, select
    async with factory() as s:
        assert (await s.execute(select(func.count()).select_from(User))).scalar_one() == 0
        assert (await s.execute(select(func.count()).select_from(Dream))).scalar_one() == 0
        assert (
            await s.execute(select(func.count()).select_from(AnalyticsEvent))
        ).scalar_one() == 0
        usage_rows = (await s.execute(select(LlmUsageEvent))).scalars().all()
        assert len(usage_rows) == 1
        assert usage_rows[0].user_id is None


@pytest.mark.asyncio
async def test_delete_user_cleans_memory_items(tmp_path):
    """user_memory_items имеет FK -> users/sessions/dreams без каскада (как
    analytics_events до фикса): без явной чистки hard-delete падал бы на PG
    IntegrityError уже на шаге delete(sessions) из-за source_session_id."""
    url = f"sqlite+aiosqlite:///{tmp_path}/mem.db"
    factory = await _seed(url)

    from sqlalchemy import func, select

    from data.models import UserMemoryItem

    async with factory() as s:
        session_id = (await s.execute(select(Session.id))).scalar_one()
        dream_id = (await s.execute(select(Dream.id))).scalar_one()
        s.add(
            UserMemoryItem(
                user_id=1,
                kind="fact",
                text="работает учителем",
                realm="real_life",
                source="qa_reply",
                source_session_id=session_id,
                dream_id=dream_id,
            )
        )
        await s.commit()

    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.request("DELETE", "/api/users/1")
        assert r.status_code == 200

    async with factory() as s:
        assert (await s.execute(select(func.count()).select_from(User))).scalar_one() == 0
        assert (await s.execute(select(func.count()).select_from(Session))).scalar_one() == 0
        assert (await s.execute(select(func.count()).select_from(Dream))).scalar_one() == 0
        assert (
            await s.execute(select(func.count()).select_from(UserMemoryItem))
        ).scalar_one() == 0


@pytest.mark.asyncio
async def test_delete_user_clears_fsm_state(tmp_path):
    """У fsm_state нет FK на users (PK — строка "bot:chat:user:thread::destiny"),
    поэтому hard-delete не падал, а тихо оставлял состояние висеть: юзер после
    удаления возвращался в свой старый шаг вместо чистого онбординга с /start.
    Чужие ключи при этом трогать нельзя."""
    url = f"sqlite+aiosqlite:///{tmp_path}/fsm.db"
    factory = await _seed(url)

    from sqlalchemy import func, select

    from data.models import FsmState

    async with factory() as s:
        s.add(FsmState(key="999:1:1:0::default", state="OnboardingStates:ASKING_DREAM"))
        s.add(FsmState(key="999:2:2:0::default", state="QuickStates:IN_QA"))
        await s.commit()

    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.request("DELETE", "/api/users/1")
        assert r.status_code == 200

    async with factory() as s:
        assert (await s.execute(select(func.count()).select_from(User))).scalar_one() == 0
        keys = (await s.execute(select(FsmState.key))).scalars().all()
        assert keys == ["999:2:2:0::default"]


@pytest.mark.asyncio
async def test_card_shows_tier_badge_and_access_line(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/tier_card.db"
    factory = await _seed(url)
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
        await s.commit()

    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/777")
    assert r.status_code == 200
    assert "∞" in r.text
    assert "Интервью, безлимит до 01.11.2026" in r.text


@pytest.mark.asyncio
async def test_set_tier_interview_sets_two_months(tmp_path, notify_interview):
    """Батч 2026-08-18: кнопка «Интервью» — безлимит на 2 месяца + дата интервью."""
    url = f"sqlite+aiosqlite:///{tmp_path}/t1.db"
    factory = await _seed(url)
    notify = notify_interview
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.post("/api/users/1/tier", json={"tier": "interview"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # Выдача статуса обязана дойти до человека: он уже видел пейвол и ушёл.
    assert body["notified"] is True
    notify.assert_awaited_once()
    assert notify.await_args.kwargs["telegram_user_id"] == 1
    assert body["tier"] == "free_interview"
    assert body["until"] is not None
    async with factory() as s:
        u = (await s.execute(select(User).where(User.telegram_user_id == 1))).scalar_one()
        assert u.access_tier == "free_interview"
        assert u.access_tier_until is not None
        assert u.interview_passed_at is not None
        # ~2 месяца, а не 3: срок в админке совпадает с обещанием в тексте пейвола.
        # SQLite отдаёт наивную дату — трактуем как UTC (как domain.access._as_utc).
        until = u.access_tier_until
        until = until if until.tzinfo else until.replace(tzinfo=UTC)
        assert (until - datetime.now(UTC)).days < 70


@pytest.mark.asyncio
async def test_regular_restores_previous_tier(tmp_path):
    """«Обычный тестер» возвращает исторический тир, а не стирает бейджик «1»."""
    url = f"sqlite+aiosqlite:///{tmp_path}/t2.db"
    factory = await _seed(url)
    async with factory() as s:
        u = (await s.execute(select(User).where(User.telegram_user_id == 1))).scalar_one()
        u.access_tier = "early_tester"
        await s.commit()
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        await c.post("/api/users/1/tier", json={"tier": "interview"})
        r = await c.post("/api/users/1/tier", json={"tier": "regular"})
    assert r.status_code == 200
    assert r.json()["until"] is None
    async with factory() as s:
        u = (await s.execute(select(User).where(User.telegram_user_id == 1))).scalar_one()
        assert u.access_tier == "early_tester"
        assert u.access_tier_until is None
        assert u.interview_passed_at is not None  # факт интервью остаётся


@pytest.mark.asyncio
async def test_set_tier_rejects_unknown_value(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/t3.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.post("/api/users/1/tier", json={"tier": "vip"})
    assert r.status_code == 400


@pytest.mark.asyncio
async def test_set_tier_404_for_missing_user(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/t4.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.post("/api/users/424242/tier", json={"tier": "interview"})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_tier_buttons_rendered(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path}/t5.db"
    await _seed(url)
    app = create_admin_app(settings=_settings(url))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://t") as c:
        await _login(c)
        r = await c.get("/users/1")
    assert "Статус доступа" in r.text
    assert "setTier('interview')" in r.text
    assert "setTier('regular')" in r.text
    assert "setTier('early_tester')" not in r.text  # третьей кнопки больше нет
