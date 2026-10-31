"""Онбординг per-kind: вопрос про переживания и метаданные воронки."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.onboarding import _life_context_loop_act
from adapters.telegram.states import OnboardingStates
from adapters.telegram.texts import ASK_PER_SESSION_CONTEXT, LOOP_ADD_BTN_DONE
from data.models import AnalyticsEvent
from data.repositories import UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_CONFLICT


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _message() -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_life_context_done_asks_per_kind_question(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42)
    await repo.update_profile(user_id=user.id, profile_life_context="я студентка")
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
    await fsm.update_data(**{FSM_KEY_ENTRY_KIND: KIND_CONFLICT})

    msg = _message()
    await _life_context_loop_act(
        text=LOOP_ADD_BTN_DONE, message=msg, state=fsm, session=test_session,
        client=AsyncMock(), model="openai/gpt-4o-mini", telegram_user_id=42,
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state
    assert msg.answer.await_args.args[0] == (
        ENTRY_TYPES[KIND_CONFLICT].per_session_question
    )


@pytest.mark.asyncio
async def test_life_context_done_dream_default_unchanged(test_session):
    """Без entry_kind в FSM (все прод-юзеры онбординга) — прежний текст сна."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42)
    await repo.update_profile(user_id=user.id, profile_life_context="я студентка")
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    msg = _message()
    await _life_context_loop_act(
        text=LOOP_ADD_BTN_DONE, message=msg, state=fsm, session=test_session,
        client=AsyncMock(), model="openai/gpt-4o-mini", telegram_user_id=42,
    )
    assert msg.answer.await_args.args[0] == ASK_PER_SESSION_CONTEXT


@pytest.mark.asyncio
async def test_onb_life_context_event_carries_entry_type(test_session):
    from adapters.telegram.handlers.onboarding import handle_life_context_input

    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)
    await fsm.update_data(**{FSM_KEY_ENTRY_KIND: KIND_CONFLICT})
    msg = _message()
    msg.text = "мне 30, работаю в найме"
    msg.voice = None
    await handle_life_context_input(
        msg, fsm, session=test_session, bot=AsyncMock(),
        openai_client=AsyncMock(), settings=SimpleNamespace(),
    )
    ev = (
        await test_session.execute(
            select(AnalyticsEvent).where(AnalyticsEvent.event_type == "onb_life_context")
        )
    ).scalar_one()
    assert ev.event_metadata == {"entry_type": KIND_CONFLICT}
