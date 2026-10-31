"""Списки по типам: изоляция, пустые тексты, per-kind карточка и удаление."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.dreams import (
    handle_conflicts_list,
    handle_dreams_list,
    handle_situations_list,
)
from data.repositories import DreamRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_CONFLICT, KIND_DREAM, KIND_SITUATION


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _message() -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = "/dreams"
    m.answer = AsyncMock()
    return m


async def _seed(test_session):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    repo = DreamRepository(test_session)
    now = datetime.now(UTC)
    await repo.create(user_id=user.id, recorded_at=now, transcription="про тигра",
                      kind=KIND_DREAM)
    await repo.create(user_id=user.id, recorded_at=now, transcription="ссора с шефом",
                      kind=KIND_SITUATION)
    return user


@pytest.mark.asyncio
async def test_dreams_list_excludes_situations(test_session):
    await _seed(test_session)
    msg = _message()
    await handle_dreams_list(msg, _fsm(), session=test_session)
    all_sent = " | ".join(c.args[0] for c in msg.answer.await_args_list)
    assert "про тигра" in all_sent
    assert "ссора с шефом" not in all_sent


@pytest.mark.asyncio
async def test_situations_list_excludes_dreams(test_session):
    await _seed(test_session)
    msg = _message()
    await handle_situations_list(msg, _fsm(), session=test_session)
    all_sent = " | ".join(c.args[0] for c in msg.answer.await_args_list)
    assert "ссора с шефом" in all_sent
    assert "про тигра" not in all_sent
    assert ENTRY_TYPES[KIND_SITUATION].list_header.format(count=1) in all_sent


@pytest.mark.asyncio
async def test_empty_conflicts_list_text(test_session):
    await _seed(test_session)
    msg = _message()
    await handle_conflicts_list(msg, _fsm(), session=test_session)
    assert msg.answer.await_args.args[0] == ENTRY_TYPES[KIND_CONFLICT].list_empty
