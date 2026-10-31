"""hard_delete_with_cascade: dream + sessions + session_messages + session_state +
dream_images удаляются; llm_usage_events.session_id обнуляется."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import (
    Dream,
    LlmService,
    LlmUsageEvent,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionState,
    SessionType,
    UsageMetric,
)
from data.models import (
    Session as SessionModel,
)
from data.repositories import (
    DreamRepository,
    SessionRepository,
    UserRepository,
)


@pytest.mark.asyncio
async def test_hard_delete_removes_dream_and_cascade(
    test_session: AsyncSession, schema,
) -> None:
    # Setup: user + dream + session + message + state + llm_usage_event
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=111, display_name="A")
    await user_repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="Test dream",
    )
    await test_session.commit()

    sess_repo = SessionRepository(test_session)
    s = await sess_repo.create(
        dream_id=dream.id,
        user_id=user.id,
        type_=SessionType.QUICK,
        is_onboarding_session=False,
    )
    await test_session.commit()

    # Add session_message
    sm = SessionMessage(
        session_id=s.id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=SessionMessageContentType.TEXT,
        content="hi",
        created_at=datetime.now(UTC),
    )
    test_session.add(sm)

    # Add session_state
    ss = SessionState(
        session_id=s.id,
        data={"key": "value"},
    )
    test_session.add(ss)

    # Add llm_usage_event linked to the session
    ue = LlmUsageEvent(
        user_id=user.id,
        session_id=s.id,
        service=LlmService.OPENROUTER,
        model="test-model",
        usage_metric=UsageMetric.INPUT_TOKENS,
        usage_amount=100,
        cost_usd=Decimal("0.001"),
    )
    test_session.add(ue)
    await test_session.commit()

    # Act
    result = await dream_repo.hard_delete_with_cascade(dream_id=dream.id, user_id=user.id)
    await test_session.commit()

    # Assert
    assert result is True

    # Dream удалён
    dreams = (
        await test_session.execute(select(Dream).where(Dream.id == dream.id))
    ).scalars().all()
    assert len(dreams) == 0

    # Session удалена
    sessions = (
        await test_session.execute(select(SessionModel).where(SessionModel.id == s.id))
    ).scalars().all()
    assert len(sessions) == 0

    # Session messages удалены
    msgs = (
        await test_session.execute(
            select(SessionMessage).where(SessionMessage.session_id == s.id)
        )
    ).scalars().all()
    assert len(msgs) == 0

    # Session state удалён
    states = (
        await test_session.execute(
            select(SessionState).where(SessionState.session_id == s.id)
        )
    ).scalars().all()
    assert len(states) == 0

    # LlmUsageEvent ОСТАЛСЯ, но session_id=NULL
    events = (
        await test_session.execute(
            select(LlmUsageEvent).where(LlmUsageEvent.user_id == user.id)
        )
    ).scalars().all()
    assert len(events) == 1
    assert events[0].session_id is None


@pytest.mark.asyncio
async def test_hard_delete_rejects_wrong_user(
    test_session: AsyncSession, schema,
) -> None:
    user_repo = UserRepository(test_session)
    user_a = await user_repo.get_or_create_by_tg_id(telegram_user_id=111, display_name="A")
    user_b = await user_repo.get_or_create_by_tg_id(telegram_user_id=222, display_name="B")
    dream_a = await DreamRepository(test_session).create(
        user_id=user_a.id,
        recorded_at=datetime.now(UTC),
        transcription="A's dream",
    )
    await test_session.commit()

    # User B пытается удалить сон User A
    result = await DreamRepository(test_session).hard_delete_with_cascade(
        dream_id=dream_a.id,
        user_id=user_b.id,
    )
    await test_session.commit()

    assert result is False

    # Сон A на месте
    dreams = (
        await test_session.execute(select(Dream).where(Dream.id == dream_a.id))
    ).scalars().all()
    assert len(dreams) == 1
