"""Вынос общей логики: /newdream и /new_* ведут себя ровно как до рефакторинга.

Baseline-тест перед выносом start_new_entry (батч 2026-08-18): фиксирует поведение
трёх гейтов и авто-паузы, чтобы рефакторинг ничего не сдвинул.
"""

from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.entry_flow import handle_new_conflict_command
from adapters.telegram.handlers.newdream import handle_newdream_command
from adapters.telegram.states import EntryStates, NewDreamStates, OnboardingStates
from adapters.telegram.texts import NOT_ONBOARDED_PROMPT, ONBOARDING_IN_PROGRESS_NOTICE
from data.models import SessionStatus, User
from data.repositories import SessionRepository, UserRepository
from domain.entry_types import KIND_CONFLICT


class _NoLimitSettings:
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    openrouter_context_model = "openai/gpt-4o-mini"


def _fsm(uid: int = 42) -> FSMContext:
    return FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=uid, user_id=uid))


def _message(uid: int = 42) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Катя")
    m.chat = Chat(id=uid, type="private")
    m.text = "/newdream"
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_newdream_rejects_user_without_onboarding(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    msg = _message()
    await handle_newdream_command(
        msg, _fsm(), test_session, _NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert msg.answer.await_args.args[0] == NOT_ONBOARDED_PROMPT


@pytest.mark.asyncio
async def test_early_onboarding_step_is_not_reset(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)
    msg = _message()
    await handle_newdream_command(
        msg, fsm, test_session, _NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert msg.answer.await_args.args[0] == ONBOARDING_IN_PROGRESS_NOTICE
    assert await fsm.get_state() == OnboardingStates.ASKING_LIFE_CONTEXT.state


@pytest.mark.asyncio
async def test_newdream_autopauses_active_session(test_session, make_quick_session):
    sess = await make_quick_session()
    user = await test_session.get(User, sess.user_id)
    fsm = _fsm(user.telegram_user_id)
    msg = _message(user.telegram_user_id)
    await handle_newdream_command(
        msg, fsm, test_session, _NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert await fsm.get_state() == NewDreamStates.ASKING_DREAM.state
    refreshed = await SessionRepository(test_session).get_by_id(sess.id, user_id=user.id)
    assert refreshed.status == SessionStatus.PAUSED


@pytest.mark.asyncio
async def test_new_conflict_command_enters_entry_states(test_session, make_quick_session):
    sess = await make_quick_session()
    user = await test_session.get(User, sess.user_id)
    fsm = _fsm(user.telegram_user_id)
    msg = _message(user.telegram_user_id)
    await handle_new_conflict_command(
        msg, fsm, test_session, _NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert await fsm.get_state() == EntryStates.ASKING_ENTRY.state
    assert (await fsm.get_data())[FSM_KEY_ENTRY_KIND] == KIND_CONFLICT
    refreshed = await SessionRepository(test_session).get_by_id(sess.id, user_id=user.id)
    assert refreshed.status == SessionStatus.PAUSED


@pytest.mark.asyncio
async def test_onboarding_closed_when_analysis_already_delivered(test_session, make_quick_session):
    """Разбор получен, «Завершить» не нажато: команда закрывает онбординг сама."""
    sess = await make_quick_session()
    user = await test_session.get(User, sess.user_id)
    user.onboarding_completed_at = None  # откатываем в состояние «онбординг не закрыт»
    await test_session.flush()
    fsm = _fsm(user.telegram_user_id)
    await fsm.set_state("QuickAnalysisStates:IN_QA")
    await fsm.update_data(session_id=sess.id, dream_id=sess.dream_id)
    msg = _message(user.telegram_user_id)

    await handle_newdream_command(
        msg, fsm, test_session, _NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )

    await test_session.refresh(user)
    assert user.onboarding_completed_at is not None
    assert await fsm.get_state() == NewDreamStates.ASKING_DREAM.state
