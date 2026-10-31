from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from adapters.telegram.middlewares import DailyActiveMiddleware
from data.analytics_events import AnalyticsEventType
from data.models import AnalyticsEvent, User


@pytest.mark.asyncio
async def test_daily_active_middleware_writes_one_per_day(test_session, schema) -> None:
    user = User(telegram_user_id=555, locale="ru")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    mw = DailyActiveMiddleware()
    called = {"n": 0}

    async def handler(event, data):
        called["n"] += 1
        return "ok"

    event = SimpleNamespace(from_user=SimpleNamespace(id=555))
    data = {"session": test_session}

    await mw(handler, event, data)
    await test_session.commit()
    await mw(handler, event, data)
    await test_session.commit()

    assert called["n"] == 2  # handler always runs
    count = await test_session.scalar(
        select(func.count())
        .select_from(AnalyticsEvent)
        .where(AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE)
    )
    assert count == 1


@pytest.mark.asyncio
async def test_daily_active_middleware_never_breaks_handler(test_session, schema) -> None:
    """Сбой записи аналитики не должен ронять обработку апдейта."""
    mw = DailyActiveMiddleware()

    async def handler(event, data):
        return "delivered"

    # from_user отсутствует — middleware должна тихо пропустить запись.
    event = SimpleNamespace(from_user=None)
    data = {"session": test_session}

    result = await mw(handler, event, data)
    assert result == "delivered"
