"""EntryStates: рассказ → loop → развилка (онбординг/повтор)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.entry_flow import (
    handle_entry_input,
    handle_entry_loop,
)
from adapters.telegram.states import EntryStates, OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import ASK_LIFE_CONTEXT_ONB, LOOP_ADD_BTN_DONE
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_SITUATION


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _message(text: str) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock(return_value=AsyncMock(message_id=7))
    return m


def test_entry_states_strings():
    assert EntryStates.ASKING_ENTRY.state == "EntryStates:ASKING_ENTRY"
    assert EntryStates.ENTRY_LOOP.state == "EntryStates:ENTRY_LOOP"


@pytest.mark.asyncio
async def test_entry_input_creates_dream_with_kind_and_goes_to_loop(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(EntryStates.ASKING_ENTRY)
    await fsm.update_data(**{FSM_KEY_ENTRY_KIND: KIND_SITUATION})

    msg = _message("поссорилась с руководителем")
    await handle_entry_input(
        msg, fsm, session=test_session, bot=AsyncMock(),
        openai_client=AsyncMock(), settings=SimpleNamespace(),
    )

    user = await UserRepository(test_session).get_by_tg_id(42)
    dreams = await DreamRepository(test_session).list_by_user(
        user_id=user.id, kind=KIND_SITUATION
    )
    assert len(dreams) == 1
    assert dreams[0].transcription == "поссорилась с руководителем"
    assert await fsm.get_state() == EntryStates.ENTRY_LOOP.state
    # loop-prompt типа «ситуация»
    assert msg.answer.await_args.args[0] == ENTRY_TYPES[KIND_SITUATION].loop_prompt


@pytest.mark.asyncio
async def test_loop_done_not_onboarded_goes_to_life_context(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(EntryStates.ENTRY_LOOP)
    await fsm.update_data(**{FSM_KEY_ENTRY_KIND: KIND_SITUATION})

    msg = _message(LOOP_ADD_BTN_DONE)  # точный текст кнопки — без LLM
    await handle_entry_loop(
        msg, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini", bot=None,
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_LIFE_CONTEXT.state
    assert msg.answer.await_args.args[0] == ASK_LIFE_CONTEXT_ONB
    # kind сохранён для даунстрима (per-session вопрос, разбор)
    assert (await fsm.get_data())[FSM_KEY_ENTRY_KIND] == KIND_SITUATION


@pytest.mark.asyncio
async def test_loop_done_onboarded_creates_session_and_asks_per_session(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42)
    await repo.mark_onboarded(user_id=user.id)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC),
        transcription="ссора", kind=KIND_SITUATION,
    )
    fsm = _fsm()
    await fsm.set_state(EntryStates.ENTRY_LOOP)
    await fsm.update_data(
        **{FSM_KEY_ENTRY_KIND: KIND_SITUATION, "onboarding_dream_id": dream.id}
    )

    msg = _message(LOOP_ADD_BTN_DONE)
    await handle_entry_loop(
        msg, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini", bot=None,
    )
    assert await fsm.get_state() == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    data = await fsm.get_data()
    sess = await SessionRepository(test_session).get_by_id(
        data["session_id"], user_id=user.id
    )
    assert sess is not None and sess.dream_id == dream.id
    assert not sess.is_onboarding_session
    assert msg.answer.await_args.args[0] == (
        ENTRY_TYPES[KIND_SITUATION].per_session_question
    )
