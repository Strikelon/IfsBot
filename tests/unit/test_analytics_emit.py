from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import func, select

from data.analytics_events import AnalyticsEventRepository, AnalyticsEventType
from data.models import AnalyticsEvent, User


async def _make_user(session, tg_id: int = 100) -> User:
    user = User(telegram_user_id=tg_id, locale="ru")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


@pytest.mark.asyncio
async def test_emit_funnel_once_inserts_first_time(test_session, schema) -> None:
    user = await _make_user(test_session)
    repo = AnalyticsEventRepository(test_session)

    inserted = await repo.emit_funnel_once(
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.FIRST_LAUNCH,
    )
    await test_session.commit()

    assert inserted is True
    count = await test_session.scalar(
        select(func.count()).select_from(AnalyticsEvent)
    )
    assert count == 1


@pytest.mark.asyncio
async def test_emit_funnel_once_is_idempotent(test_session, schema) -> None:
    """Повторный /start не должен плодить второй first_launch."""
    user = await _make_user(test_session)
    repo = AnalyticsEventRepository(test_session)

    first = await repo.emit_funnel_once(
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.FIRST_LAUNCH,
    )
    await test_session.commit()
    second = await repo.emit_funnel_once(
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.FIRST_LAUNCH,
    )
    await test_session.commit()

    assert first is True
    assert second is False
    count = await test_session.scalar(
        select(func.count()).select_from(AnalyticsEvent)
    )
    assert count == 1


@pytest.mark.asyncio
async def test_emit_funnel_once_records_metadata_and_day(test_session, schema) -> None:
    user = await _make_user(test_session)
    repo = AnalyticsEventRepository(test_session)
    when = datetime(2026, 6, 29, 8, 30, tzinfo=UTC)

    await repo.emit_funnel_once(
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_DREAM_TOLD,
        occurred_at=when,
        metadata={"dream_id": 5},
    )
    await test_session.commit()

    ev = await test_session.scalar(select(AnalyticsEvent))
    assert ev.event_type == "onb_dream_told"
    assert ev.event_day.isoformat() == "2026-06-29"
    assert ev.event_metadata == {"dream_id": 5}


@pytest.mark.asyncio
async def test_daily_active_one_per_day(test_session, schema) -> None:
    user = await _make_user(test_session, tg_id=200)
    repo = AnalyticsEventRepository(test_session)
    day = datetime(2026, 6, 29, 9, 0, tzinfo=UTC)

    a = await repo.emit_daily_active(
        user_id=user.id, telegram_user_id=user.telegram_user_id, occurred_at=day
    )
    await test_session.commit()
    b = await repo.emit_daily_active(
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        occurred_at=day + timedelta(hours=5),
    )
    await test_session.commit()

    assert a is True
    assert b is False
    count = await test_session.scalar(
        select(func.count()).select_from(AnalyticsEvent)
    )
    assert count == 1


@pytest.mark.asyncio
async def test_daily_active_distinct_days_both_recorded(test_session, schema) -> None:
    user = await _make_user(test_session, tg_id=201)
    repo = AnalyticsEventRepository(test_session)
    d1 = datetime(2026, 6, 29, 9, 0, tzinfo=UTC)
    d2 = datetime(2026, 6, 30, 1, 0, tzinfo=UTC)

    await repo.emit_daily_active(
        user_id=user.id, telegram_user_id=user.telegram_user_id, occurred_at=d1
    )
    await repo.emit_daily_active(
        user_id=user.id, telegram_user_id=user.telegram_user_id, occurred_at=d2
    )
    await test_session.commit()

    count = await test_session.scalar(
        select(func.count())
        .select_from(AnalyticsEvent)
        .where(AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE)
    )
    assert count == 2
