"""Append-события img_q_* и funnel-шаги onb_img_q_*."""
from unittest.mock import patch

import pytest
from sqlalchemy import select

from analytics.funnel import FUNNEL_STEPS
from data.analytics_events import (
    AnalyticsEventRepository,
    AnalyticsEventType,
    emit_append_event,
)
from data.models import AnalyticsEvent


def test_event_types_exist():
    assert AnalyticsEventType.IMG_Q_OFFERED == "img_q_offered"
    assert AnalyticsEventType.IMG_Q_CHOICE == "img_q_choice"
    assert AnalyticsEventType.IMG_Q_NO_IMAGES == "img_q_no_images"
    assert AnalyticsEventType.IMG_Q_ANSWERED == "img_q_answered"
    assert AnalyticsEventType.IMG_Q_SKIPPED == "img_q_skipped"
    assert AnalyticsEventType.IMG_Q_FLOW_DONE == "img_q_flow_done"
    assert AnalyticsEventType.ONB_IMG_Q_OFFERED == "onb_img_q_offered"
    assert AnalyticsEventType.ONB_IMG_Q_DONE == "onb_img_q_done"


def test_funnel_steps_order():
    i = FUNNEL_STEPS.index
    assert i("onb_intent_given") < i("onb_img_q_offered") < i("onb_img_q_done") \
        < i("onb_analysis_delivered")


@pytest.mark.asyncio
async def test_emit_is_append_not_idempotent(db_session, make_user):
    """emit() дважды → ДВЕ строки (в отличие от emit_funnel_once)."""
    user = await make_user()
    repo = AnalyticsEventRepository(db_session)
    for idx in (0, 1):
        await repo.emit(
            user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.IMG_Q_ANSWERED,
            metadata={"session_id": 1, "idx": idx, "total": 2, "image": "Отец", "via": "text"},
        )
    rows = (await db_session.execute(
        select(AnalyticsEvent).where(
            AnalyticsEvent.event_type == AnalyticsEventType.IMG_Q_ANSWERED
        )
    )).scalars().all()
    assert len(rows) == 2
    assert rows[0].event_metadata["idx"] == 0 and rows[1].event_metadata["idx"] == 1


@pytest.mark.asyncio
async def test_emit_append_event_swallows_failure(db_session, make_user):
    """SAVEPOINT-обёртка глотает сбой записи — транзакция хэндлера жива."""
    user = await make_user()
    with patch.object(AnalyticsEventRepository, "emit", side_effect=RuntimeError("boom")):
        await emit_append_event(  # не должно бросить
            db_session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.IMG_Q_OFFERED, metadata={"session_id": 1},
        )
    # Транзакция не потравлена: обычная запись после сбоя проходит.
    await AnalyticsEventRepository(db_session).emit(
        user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_OFFERED, metadata={"session_id": 1},
    )
    await db_session.flush()
