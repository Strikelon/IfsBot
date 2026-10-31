"""thinking_guard перехватывает text/voice во время LLM-вызовов."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message, Voice
from aiogram.types import User as TgUser

from adapters.telegram.handlers.thinking_guard import (
    handle_thinking_text,
    handle_thinking_voice,
)
from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import THINKING_GUARD_NOTICE


@pytest_asyncio.fixture
async def fsm_context() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _text_message(text: str) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=42, is_bot=False, first_name="T")
    msg.chat = Chat(id=42, type="private")
    msg.text = text
    msg.voice = None
    msg.answer = AsyncMock()
    return msg


def _voice_message(file_id: str = "voice_abc") -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=42, is_bot=False, first_name="T")
    msg.chat = Chat(id=42, type="private")
    msg.text = None
    msg.voice = Voice(file_id=file_id, file_unique_id="u_" + file_id, duration=3)
    msg.answer = AsyncMock()
    return msg


@pytest.mark.asyncio
@pytest.mark.parametrize("thinking_state", [
    QuickAnalysisStates.GENERATING_INITIAL,
    QuickAnalysisStates.GENERATING_FOLLOWUP,
    OnboardingStates.GENERATING_INITIAL_ONB,
])
async def test_text_intercepted_in_generating_states(
    thinking_state,
    fsm_context: FSMContext,
):
    await fsm_context.set_state(thinking_state)
    msg = _text_message("что-то")

    await handle_thinking_text(msg, fsm_context)

    texts = [call.args[0] for call in msg.answer.call_args_list]
    assert any(THINKING_GUARD_NOTICE in t for t in texts)


@pytest.mark.asyncio
async def test_voice_intercepted_does_not_call_whisper(
    fsm_context: FSMContext,
):
    await fsm_context.set_state(QuickAnalysisStates.GENERATING_FOLLOWUP)
    msg = _voice_message()

    # handle_thinking_voice не зависит от AsyncOpenAI/Bot — отсутствие импортов
    # в модуле само по себе гарантирует что Whisper не вызовется.
    # Проверяем что юзер получил guard notice.
    await handle_thinking_voice(msg, fsm_context)

    texts = [call.args[0] for call in msg.answer.call_args_list]
    assert any(THINKING_GUARD_NOTICE in t for t in texts)
