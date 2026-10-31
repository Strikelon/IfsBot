"""Разрез событий пре-вопросов по типу обращения.

Имена событий (`img_q_*`) намеренно не меняются — они append-only и на них завязан
отчёт с июля. Тип обращения приезжает в `metadata.entry_type`; события без этого
поля (вся история до фичи) трактуются как сны.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy import select

from adapters.telegram.handlers.image_questions import start_image_questions_flow
from analytics.image_questions import ImgQEvent, split_by_entry_type
from data.models import AnalyticsEvent, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION


def _ev(event_type: str, **meta) -> ImgQEvent:
    return ImgQEvent(user_id=1, event_type=event_type, metadata=meta)


def test_legacy_events_without_entry_type_count_as_dreams():
    """История прода (июль–август) поля не имеет — она вся про сны."""
    buckets = split_by_entry_type([
        _ev("img_q_offered", session_id=1),
        _ev("img_q_choice", session_id=1, choice="answer"),
    ])
    assert set(buckets) == {KIND_DREAM}
    assert len(buckets[KIND_DREAM]) == 2


def test_events_split_by_kind():
    buckets = split_by_entry_type([
        _ev("img_q_offered", session_id=1, entry_type=KIND_DREAM),
        _ev("img_q_offered", session_id=2, entry_type=KIND_SITUATION),
        _ev("img_q_answered", session_id=2, entry_type=KIND_SITUATION, idx=0, total=2),
        _ev("img_q_offered", session_id=3, entry_type=KIND_CONFLICT),
    ])
    assert sorted(buckets) == sorted([KIND_DREAM, KIND_SITUATION, KIND_CONFLICT])
    assert len(buckets[KIND_SITUATION]) == 2
    assert len(buckets[KIND_DREAM]) == 1


def test_unknown_kind_falls_back_to_dream():
    buckets = split_by_entry_type([_ev("img_q_offered", session_id=9, entry_type="garbage")])
    assert set(buckets) == {KIND_DREAM}


@pytest.mark.asyncio
async def test_offered_event_carries_entry_type(test_session):
    """Хэндлер обязан класть тип в metadata, иначе разрез в отчёте слепой."""
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=77)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC),
        transcription="ссора", kind=KIND_SITUATION,
    )
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    message = AsyncMock()
    message.answer = AsyncMock(return_value=AsyncMock(message_id=5))
    fsm = FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=77, user_id=77)
    )
    await start_image_questions_flow(
        message=message, state=fsm, session=test_session, user=user,
        session_id=sess.id, dream_id=dream.id, is_onboarding=False,
        settings=AsyncMock(), bot=None,
    )
    ev = (
        await test_session.execute(
            select(AnalyticsEvent).where(AnalyticsEvent.event_type == "img_q_offered")
        )
    ).scalar_one()
    assert ev.event_metadata["entry_type"] == KIND_SITUATION
