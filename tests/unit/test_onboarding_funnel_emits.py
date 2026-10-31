
import pytest
from sqlalchemy import select

from data.analytics_events import (
    AnalyticsEventType,
    emit_funnel_step,
)
from data.models import AnalyticsEvent, User


@pytest.mark.asyncio
async def test_emit_funnel_step_helper_writes_event(test_session, schema) -> None:
    user = User(telegram_user_id=900, locale="ru")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    await emit_funnel_step(
        test_session,
        user_id=user.id,
        telegram_user_id=900,
        event_type=AnalyticsEventType.ONB_DREAM_TOLD,
        metadata={"dream_id": 3},
    )
    await test_session.commit()

    ev = await test_session.scalar(select(AnalyticsEvent))
    assert ev.event_type == "onb_dream_told"
    assert ev.event_metadata == {"dream_id": 3}


@pytest.mark.asyncio
async def test_emit_funnel_step_swallows_errors(test_session, schema) -> None:
    """Аналитика не должна ронять онбординг даже при кривых аргументах."""
    # user_id, которого нет → FK-нарушение на flush; helper должен проглотить.
    await emit_funnel_step(
        test_session,
        user_id=999999,
        telegram_user_id=901,
        event_type=AnalyticsEventType.FIRST_LAUNCH,
    )
    # не должно бросить
