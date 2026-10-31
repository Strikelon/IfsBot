from datetime import UTC, date, datetime

import pytest

from data.models import AnalyticsEvent, User


@pytest.mark.asyncio
async def test_analytics_event_can_be_created(test_session, schema) -> None:
    user = User(telegram_user_id=42, display_name="Катя", locale="ru")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    ev = AnalyticsEvent(
        user_id=user.id,
        telegram_user_id=42,
        event_type="first_launch",
        occurred_at=datetime(2026, 6, 29, 10, 0, tzinfo=UTC),
        event_day=date(2026, 6, 29),
        event_metadata={"dream_id": 7},
    )
    test_session.add(ev)
    await test_session.commit()
    await test_session.refresh(ev)

    assert ev.id is not None
    assert ev.event_type == "first_launch"
    assert ev.event_day == date(2026, 6, 29)
    assert ev.event_metadata == {"dream_id": 7}


@pytest.mark.asyncio
async def test_user_acquisition_source_defaults_none(test_session, schema) -> None:
    user = User(telegram_user_id=43, locale="ru")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)
    assert user.acquisition_source is None
