from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select

from data.analytics_events import AnalyticsEventType, backfill_daily_active
from data.models import AnalyticsEvent, Dream, User


async def _seed(session) -> User:
    user = User(telegram_user_id=700, locale="ru")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    # Два сна в разные дни → два distinct daily_active дня.
    session.add_all([
        Dream(
            user_id=user.id,
            recorded_at=datetime(2026, 6, 1, 8, tzinfo=UTC),
            created_at=datetime(2026, 6, 1, 8, tzinfo=UTC),
            transcription="test dream one",
            status="recorded",
        ),
        Dream(
            user_id=user.id,
            recorded_at=datetime(2026, 6, 3, 9, tzinfo=UTC),
            created_at=datetime(2026, 6, 3, 9, tzinfo=UTC),
            transcription="test dream two",
            status="recorded",
        ),
    ])
    await session.commit()
    return user


@pytest.mark.asyncio
async def test_backfill_creates_distinct_days(test_session, schema) -> None:
    await _seed(test_session)
    inserted = await backfill_daily_active(test_session)
    await test_session.commit()

    assert inserted == 2
    days = (await test_session.scalars(
        select(AnalyticsEvent.event_day)
        .where(AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE)
        .order_by(AnalyticsEvent.event_day)
    )).all()
    assert [d.isoformat() for d in days] == ["2026-06-01", "2026-06-03"]


@pytest.mark.asyncio
async def test_backfill_is_idempotent(test_session, schema) -> None:
    await _seed(test_session)
    first = await backfill_daily_active(test_session)
    await test_session.commit()
    second = await backfill_daily_active(test_session)
    await test_session.commit()

    assert first == 2
    assert second == 0
    count = await test_session.scalar(
        select(func.count())
        .select_from(AnalyticsEvent)
        .where(AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE)
    )
    assert count == 2
