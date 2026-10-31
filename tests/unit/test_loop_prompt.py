"""Хелперы loop-add prompt: показ inline-кнопки + её гашение."""

from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, InlineKeyboardMarkup, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._loop_prompt import (
    _FSM_KEY_LOOP_PROMPT_MSG_ID,
    extinguish_loop_button,
    send_loop_prompt,
)
from adapters.telegram.texts import LOOP_ADD_PROMPT_LIFE


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _make_message(user_id: int = 42) -> Message:
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    message.chat = Chat(id=user_id, type="private")
    message.answer = AsyncMock(return_value=AsyncMock(message_id=777))
    return message


@pytest.mark.asyncio
async def test_send_loop_prompt_shows_inline_button_and_stores_msg_id() -> None:
    state = _fsm()
    message = _make_message()

    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_LIFE)

    message.answer.assert_awaited_once()
    args, kwargs = message.answer.await_args
    assert args[0] == LOOP_ADD_PROMPT_LIFE
    assert isinstance(kwargs["reply_markup"], InlineKeyboardMarkup)
    data = await state.get_data()
    assert data[_FSM_KEY_LOOP_PROMPT_MSG_ID] == 777


@pytest.mark.asyncio
async def test_extinguish_removes_markup_and_clears_stored_id() -> None:
    state = _fsm()
    await state.update_data(**{_FSM_KEY_LOOP_PROMPT_MSG_ID: 777})
    bot = AsyncMock()

    await extinguish_loop_button(state=state, bot=bot, chat_id=42)

    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=777, reply_markup=None
    )
    data = await state.get_data()
    assert data[_FSM_KEY_LOOP_PROMPT_MSG_ID] is None


@pytest.mark.asyncio
async def test_extinguish_noop_when_no_stored_id() -> None:
    state = _fsm()
    bot = AsyncMock()

    await extinguish_loop_button(state=state, bot=bot, chat_id=42)

    bot.edit_message_reply_markup.assert_not_awaited()


@pytest.mark.asyncio
async def test_extinguish_swallows_edit_error() -> None:
    """Сообщение могло быть уже отредактировано/удалено — ошибку глотаем."""
    state = _fsm()
    await state.update_data(**{_FSM_KEY_LOOP_PROMPT_MSG_ID: 777})
    bot = AsyncMock()
    bot.edit_message_reply_markup = AsyncMock(side_effect=Exception("message not found"))

    await extinguish_loop_button(state=state, bot=bot, chat_id=42)  # не падает

    data = await state.get_data()
    assert data[_FSM_KEY_LOOP_PROMPT_MSG_ID] is None
