"""L6: /profile правка life_context — режим «заменить целиком / дописать».

Ручная правка не должна молча сносить факты, накопленные живой памятью:
перед вводом текста юзер выбирает режим, режим хранится в FSM-data.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser

import adapters.telegram.handlers.profile as profile_module
from adapters.telegram.handlers.profile import (
    handle_edit_life_context,
    handle_edit_life_context_voice,
    handle_life_context_mode_callback,
    handle_life_context_mode_nudge,
    handle_profile_edit_callback,
)
from adapters.telegram.states import ProfileEditStates
from adapters.telegram.texts import ASK_NEW_LIFE_CONTEXT, IDLE_HINT
from data.repositories import UserRepository

EXISTING = "Живу в Грузии, двое детей."
ADDED = "Запустил новый проект."


def _msg(text: str | None, user_id: int = 42) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    m.chat = Chat(id=user_id, type="private")
    m.text = text
    m.answer = AsyncMock()
    return m


def _callback(data: str, user_id: int = 42) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=user_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


@pytest.fixture
def fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


@pytest_asyncio.fixture
async def onboarded_user(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context=EXISTING)
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()
    return user


# ---------------------------------------------------------------------------
# Вход в правку: показываем выбор режима, НЕ включаем ввод текста сразу
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_entry_shows_mode_keyboard(fsm, test_session) -> None:
    cb = _callback("profile:edit:life_context")
    await handle_profile_edit_callback(cb, state=fsm, session=test_session)

    # Режим ещё не выбран → ждём выбора режима (НЕ ввод текста и НЕ None:
    # None провалил бы стрей-текст в idle catch-all, молча теряя ввод).
    assert (
        await fsm.get_state() == ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE.state
    )

    cb.message.answer.assert_called_once()
    markup = cb.message.answer.await_args.kwargs.get("reply_markup")
    assert markup is not None
    callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert "profile:life:replace" in callbacks
    assert "profile:life:append" in callbacks


# ---------------------------------------------------------------------------
# Стрей-ввод ДО выбора режима: переспрашиваем с клавой, текст НЕ теряем в idle
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stray_text_before_mode_choice_reprompts_not_swallowed(
    test_session, onboarded_user, fsm
) -> None:
    """Regression: текст, набранный до выбора «заменить/дописать», не должен
    молча уходить в idle catch-all (IDLE_HINT) — бот переспрашивает режим."""
    # Вход в правку → бот ждёт выбор режима
    cb = _callback("profile:edit:life_context")
    await handle_profile_edit_callback(cb, state=fsm, session=test_session)
    assert (
        await fsm.get_state() == ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE.state
    )

    # Юзер печатает текст ДО тапа по кнопке режима
    msg = _msg(ADDED)
    await handle_life_context_mode_nudge(msg, state=fsm)

    # Контекст в БД не изменён (текст не применён и не потерян как «применённый»)
    await test_session.refresh(onboarded_user)
    assert onboarded_user.profile_life_context == EXISTING

    # Состояние не сброшено — следующий тап по кнопке продолжит флоу
    assert (
        await fsm.get_state() == ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE.state
    )

    # Ответ — не IDLE_HINT, а нудж с той же клавиатурой выбора режима
    msg.answer.assert_called_once()
    sent_text = msg.answer.await_args.args[0]
    assert sent_text != IDLE_HINT
    markup = msg.answer.await_args.kwargs.get("reply_markup")
    assert markup is not None
    callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert "profile:life:replace" in callbacks
    assert "profile:life:append" in callbacks


# ---------------------------------------------------------------------------
# Выбор режима: сохраняется в FSM-data + включается ввод текста
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mode_callback_append_sets_state_and_mode(fsm, test_session) -> None:
    cb = _callback("profile:life:append")
    await handle_life_context_mode_callback(cb, state=fsm)

    assert await fsm.get_state() == ProfileEditStates.EDITING_LIFE_CONTEXT.state
    data = await fsm.get_data()
    assert data.get("life_context_edit_mode") == "append"

    sent = cb.message.answer.await_args.args[0]
    assert ASK_NEW_LIFE_CONTEXT in sent or sent == ASK_NEW_LIFE_CONTEXT


@pytest.mark.asyncio
async def test_mode_callback_replace_sets_mode(fsm, test_session) -> None:
    cb = _callback("profile:life:replace")
    await handle_life_context_mode_callback(cb, state=fsm)

    assert await fsm.get_state() == ProfileEditStates.EDITING_LIFE_CONTEXT.state
    data = await fsm.get_data()
    assert data.get("life_context_edit_mode") == "replace"


# ---------------------------------------------------------------------------
# Применение (text): append сохраняет существующее, replace затирает
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_mode_preserves_existing(test_session, onboarded_user, fsm) -> None:
    await fsm.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)
    await fsm.update_data(life_context_edit_mode="append")

    await handle_edit_life_context(_msg(ADDED), state=fsm, session=test_session)

    await test_session.refresh(onboarded_user)
    assert onboarded_user.profile_life_context == f"{EXISTING}\n\n{ADDED}"
    assert await fsm.get_state() is None


@pytest.mark.asyncio
async def test_replace_mode_overwrites_existing(test_session, onboarded_user, fsm) -> None:
    await fsm.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)
    await fsm.update_data(life_context_edit_mode="replace")

    await handle_edit_life_context(_msg(ADDED), state=fsm, session=test_session)

    await test_session.refresh(onboarded_user)
    assert onboarded_user.profile_life_context == ADDED


@pytest.mark.asyncio
async def test_append_mode_with_empty_existing_writes_plain(test_session, fsm) -> None:
    """existing пустой → append схлопывается в обычную запись без ведущих \\n\\n."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()

    await fsm.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)
    await fsm.update_data(life_context_edit_mode="append")

    await handle_edit_life_context(_msg(ADDED), state=fsm, session=test_session)

    await test_session.refresh(user)
    assert user.profile_life_context == ADDED


# ---------------------------------------------------------------------------
# Применение (voice): режим уважается и голосовым путём
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_voice_append_mode_preserves_existing(
    test_session, onboarded_user, fsm, monkeypatch
) -> None:
    monkeypatch.setattr(
        profile_module, "_transcribe_profile_voice", AsyncMock(return_value=ADDED)
    )
    await fsm.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)
    await fsm.update_data(life_context_edit_mode="append")

    msg = _msg(None)
    await handle_edit_life_context_voice(
        msg, state=fsm, session=test_session,
        bot=AsyncMock(), openai_client=AsyncMock(), settings=AsyncMock(),
    )

    await test_session.refresh(onboarded_user)
    assert onboarded_user.profile_life_context == f"{EXISTING}\n\n{ADDED}"
