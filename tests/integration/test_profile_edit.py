"""Integration tests for /profile editing FSM (Task D.4).

/settings is removed — all tests here cover the new /profile edit flow.
Language removed from profile in Bug 2 fix.
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

from adapters.telegram.handlers.profile import (
    handle_edit_life_context,
    handle_life_context_mode_callback,
    handle_profile,
    handle_profile_edit_callback,
)
from adapters.telegram.states import ProfileEditStates
from adapters.telegram.texts import (
    ASK_NEW_LIFE_CONTEXT,
    NOT_ONBOARDED_PROMPT,
    PROFILE_LIFE_CONTEXT_UPDATED,
)
from data.repositories import UserRepository


def _msg(text: str, user_id: int = 42) -> Message:
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


def _make_state(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


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
    await repo.update_profile(
        user_id=user.id,
        profile_life_context="меняю работу",
    )
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()
    return user


# ---------------------------------------------------------------------------
# 1. Not-onboarded user
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_command_not_onboarded_shows_prompt(test_session) -> None:
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()

    msg = _msg("/profile")
    await handle_profile(msg, state=_make_state(), session=test_session)

    sent = msg.answer.await_args.args[0]
    assert NOT_ONBOARDED_PROMPT in sent or sent == NOT_ONBOARDED_PROMPT


# ---------------------------------------------------------------------------
# 2. Onboarded user sees card + edit buttons (no language)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_command_onboarded_shows_card_with_edit_buttons(
    test_session, onboarded_user
) -> None:
    msg = _msg("/profile")
    await handle_profile(msg, state=_make_state(), session=test_session)

    call_kwargs = msg.answer.await_args
    sent_text = call_kwargs.args[0]
    sent_markup = call_kwargs.kwargs.get("reply_markup")

    # Card text contains the "(из Telegram)" suffix on the name line
    assert "(из Telegram)" in sent_text
    assert "Катя" in sent_text

    # Language must NOT appear in the profile card
    assert "Язык" not in sent_text

    # Inline keyboard present
    assert sent_markup is not None
    all_button_texts = [
        btn.text
        for row in sent_markup.inline_keyboard
        for btn in row
    ]
    # Must NOT have an "Изменить имя" button
    assert not any("Изменить имя" in t for t in all_button_texts)
    # Must NOT have a language button
    assert not any("язык" in t.lower() for t in all_button_texts)
    # Must NOT have an experience button (removed)
    assert not any("опыт" in t.lower() for t in all_button_texts)
    # Must have the context edit button
    assert any("контекст" in t.lower() or "Изменить контекст" in t for t in all_button_texts)


# ---------------------------------------------------------------------------
# 3-4. Callback → state transitions
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_edit_callback_life_context_two_step_flow(fsm, test_session) -> None:
    """L6: сперва выбор режима (заменить/дописать), потом состояние ввода текста."""
    cb = _callback("profile:edit:life_context")
    await handle_profile_edit_callback(cb, state=fsm, session=test_session)

    # Шаг 1: ждём выбор режима (sub-state — чтобы стрей-текст не утёк в idle),
    # показана клава выбора режима
    assert (
        await fsm.get_state() == ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE.state
    )
    cb.message.answer.assert_called_once()
    markup = cb.message.answer.await_args.kwargs.get("reply_markup")
    callbacks = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert callbacks == ["profile:life:replace", "profile:life:append"]

    # Шаг 2: выбор режима → EDITING_LIFE_CONTEXT + приглашение к вводу
    cb_mode = _callback("profile:life:replace")
    await handle_life_context_mode_callback(cb_mode, state=fsm)

    assert await fsm.get_state() == ProfileEditStates.EDITING_LIFE_CONTEXT.state
    sent = cb_mode.message.answer.await_args.args[0]
    assert ASK_NEW_LIFE_CONTEXT in sent or sent == ASK_NEW_LIFE_CONTEXT


@pytest.mark.asyncio
async def test_profile_edit_callback_experience_is_noop(fsm, test_session) -> None:
    """Опыт убран из профиля → callback experience больше не обрабатывается."""
    cb = _callback("profile:edit:experience")
    await handle_profile_edit_callback(cb, state=fsm, session=test_session)

    assert await fsm.get_state() is None
    cb.message.answer.assert_not_called()


# ---------------------------------------------------------------------------
# 5. language callback no longer sets any state (unknown field, noop)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_profile_edit_callback_language_is_noop(fsm, test_session) -> None:
    cb = _callback("profile:edit:language")
    await handle_profile_edit_callback(cb, state=fsm, session=test_session)

    # State should remain None — unknown field, handler ignores it
    current = await fsm.get_state()
    assert current is None

    # No answer sent for unknown field
    cb.message.answer.assert_not_called()


# ---------------------------------------------------------------------------
# 6. Edit life_context — saves + clears state
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_edit_life_context_saves_and_clears_state(
    test_session, onboarded_user, fsm
) -> None:
    await fsm.set_state(ProfileEditStates.EDITING_LIFE_CONTEXT)

    msg = _msg("новый контекст жизни")
    await handle_edit_life_context(msg, state=fsm, session=test_session)

    # State cleared
    assert await fsm.get_state() is None

    # Confirmation sent
    sent = msg.answer.await_args.args[0]
    assert PROFILE_LIFE_CONTEXT_UPDATED in sent or sent == PROFILE_LIFE_CONTEXT_UPDATED

    # DB updated
    await test_session.refresh(onboarded_user)
    assert onboarded_user.profile_life_context == "новый контекст жизни"
