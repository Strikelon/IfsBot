from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.start import handle_start
from adapters.telegram.states import OnboardingStates
from data.models import ExperienceLevel
from data.repositories import UserRepository


@pytest.fixture
def fsm_context() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _make_message(text: str = "/start", user_id: int = 42, first_name: str = "Катя") -> Message:
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=user_id, is_bot=False, first_name=first_name)
    message.chat = Chat(id=user_id, type="private")
    message.text = text
    message.answer = AsyncMock()
    return message


@pytest.mark.asyncio
async def test_start_creates_user_and_begins_onboarding_when_no_profile(
    test_session, schema, fsm_context: FSMContext
) -> None:
    message = _make_message()
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ) as mock_update_cmds:
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    # User создан
    repo = UserRepository(test_session)
    user = await repo.get_by_tg_id(42)
    assert user is not None
    assert user.telegram_user_id == 42
    assert user.display_name == "Катя"

    # FSM-state установлен — экран выбора типа обращения (спека B3)
    state = await fsm_context.get_state()
    assert state == OnboardingStates.CHOOSING_ENTRY_TYPE.state

    # Бот отправил ОДНО приветствие с выбором, что разбирать
    assert message.answer.await_count == 1
    intro = message.answer.await_args.args[0]
    assert "Здравствуйте" in intro
    assert "Катя" in intro
    assert "Выберите, что вы хотите разобрать" in intro
    # …и три inline-кнопки типов
    kb = message.answer.await_args.kwargs["reply_markup"]
    assert len(kb.inline_keyboard) == 3

    # sync_user_commands вызван — для not-yet-onboarded
    mock_update_cmds.assert_awaited_once()
    kwargs = mock_update_cmds.await_args.kwargs
    assert kwargs["user"].telegram_user_id == 42
    assert kwargs["onboarded"] is False


@pytest.mark.asyncio
async def test_start_skips_onboarding_when_profile_complete(
    test_session, schema, fsm_context: FSMContext
) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(
        user_id=user.id,
        display_name="Катя",
        profile_life_context="переезжаю",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()

    message = _make_message()
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ) as mock_update_cmds:
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    state = await fsm_context.get_state()
    assert state is None  # никакой онбординг не активен

    message.answer.assert_awaited_once()
    sent = message.answer.await_args.args[0]
    assert "с возвращением" in sent.lower()
    assert "Катя" in sent

    # sync_user_commands вызван с onboarded=True
    mock_update_cmds.assert_awaited_once()
    kwargs = mock_update_cmds.await_args.kwargs
    assert kwargs["user"].telegram_user_id == 42
    assert kwargs["onboarded"] is True
