"""Команды /new_*: гейты онбординга, авто-пауза активной сессии, вход в EntryStates."""

from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.entry_flow import handle_new_situation_command
from adapters.telegram.states import EntryStates
from adapters.telegram.texts import NOT_ONBOARDED_PROMPT
from data.models import SessionStatus, User
from data.repositories import SessionRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_SITUATION


class _NoLimitSettings:
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    openrouter_context_model = "openai/gpt-4o-mini"


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _message() -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = "/new_external_situation"
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_not_onboarded_user_is_rejected(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    msg = _message()
    await handle_new_situation_command(
        msg, _fsm(), session=test_session, settings=_NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert msg.answer.await_args.args[0] == NOT_ONBOARDED_PROMPT


@pytest.mark.asyncio
async def test_onboarded_flow_and_autopause(test_session, make_quick_session):
    # make_quick_session создаёт onboarded-юзера + активную in_progress-сессию.
    sess = await make_quick_session()
    sess_repo = SessionRepository(test_session)
    active = await sess_repo.get_by_id(sess.id, user_id=sess.user_id)
    assert active.status == SessionStatus.IN_PROGRESS

    tg_user = await test_session.get(User, sess.user_id)
    msg = _message()
    msg.from_user = TgUser(id=tg_user.telegram_user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=tg_user.telegram_user_id, type="private")
    fsm = _fsm()

    await handle_new_situation_command(
        msg, fsm, session=test_session, settings=_NoLimitSettings(), bot=None,
        openrouter_classifier_client=None,
    )

    assert await fsm.get_state() == EntryStates.ASKING_ENTRY.state
    assert (await fsm.get_data())[FSM_KEY_ENTRY_KIND] == KIND_SITUATION
    # приглашение повторного обращения (без слова «первый»)
    sent_texts = [c.args[0] for c in msg.answer.await_args_list]
    assert ENTRY_TYPES[KIND_SITUATION].repeat_invite in sent_texts
    # активная сессия запаузена (DB-path, как у /newdream)
    refreshed = await sess_repo.get_by_id(sess.id, user_id=active.user_id)
    assert refreshed.status == SessionStatus.PAUSED
