"""Двойной тап «Завершить» не запускает вторую платную экстракцию памяти.

BusyGuardMiddleware фильтрует только Message (middlewares.py), inline-тап qa:finish
приходит как CallbackQuery и проходит мимо. Без своего lock'а два быстрых тапа дают
две конкурентные экстракции памяти — платный вызов LLM на каждую.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers import quick as quick_mod
from adapters.telegram.handlers.quick import handle_finish_quick_reply
from data.models import User


class _Settings:
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    openrouter_context_model = "openai/gpt-4o-mini"


def _message(uid: int) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Катя")
    m.chat = Chat(id=uid, type="private")
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_two_concurrent_finishes_pause_session_once(
    test_session, make_quick_session, monkeypatch
):
    sess = await make_quick_session()
    user = await test_session.get(User, sess.user_id)
    uid = user.telegram_user_id

    calls: list[int] = []

    async def slow_pause(**kwargs):
        calls.append(kwargs["session_id"])
        await asyncio.sleep(0.05)  # экстракция памяти — долгий платный вызов
        await quick_mod.SessionRepository(test_session).pause(session_id=kwargs["session_id"])

    monkeypatch.setattr(quick_mod, "_pause_session", slow_pause)

    fsm = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=uid, user_id=uid))
    await fsm.set_state("QuickAnalysisStates:IN_QA")
    await fsm.update_data(session_id=sess.id, dream_id=sess.dream_id)

    await asyncio.gather(
        handle_finish_quick_reply(
            _message(uid), fsm, test_session, bot=AsyncMock(), telegram_user_id=uid,
            settings=_Settings(), openrouter_classifier_client=None,
        ),
        handle_finish_quick_reply(
            _message(uid), fsm, test_session, bot=AsyncMock(), telegram_user_id=uid,
            settings=_Settings(), openrouter_classifier_client=None,
        ),
    )

    assert calls == [sess.id]  # вторая экстракция не запускалась
