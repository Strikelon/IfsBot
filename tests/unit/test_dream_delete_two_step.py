"""Unit-тесты двухступенчатого удаления сна (Task 10, #3C)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Message
from aiogram.types import User as TgUser
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.dreams import (
    handle_dream_delete_callback,
    handle_dream_delete_cancel_callback,
    handle_dream_delete_confirm_callback,
    handle_dream_delete_final_callback,
)
from adapters.telegram.texts import (
    DREAM_DELETE_BTN_CANCEL,
    DREAM_DELETE_BTN_FINAL,
    DREAM_DELETE_BTN_YES,
    DREAM_DELETED_NOTICE,
)
from data.models import Base, ExperienceLevel
from data.repositories import DreamRepository, UserRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def user_and_dream(test_session, schema):
    """Онбординг-юзер (tg_id=42) с одним сном."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="меняю работу",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    from datetime import UTC, datetime
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="мне снился тёмный лес",
    )
    await test_session.commit()
    return user, dream


@pytest.fixture
def fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _fake_callback(data: str, user_id: int = 42) -> CallbackQuery:
    msg = AsyncMock(spec=Message)
    msg.edit_text = AsyncMock()
    cb = AsyncMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="X")
    cb.message = msg
    cb.answer = AsyncMock()
    return cb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_step_1_delete_shows_confirm_1(test_session, user_and_dream, fsm) -> None:
    """Тап dream_delete:<id> → edit_text с DREAM_DELETE_CONFIRM_1 + [Да, удалить | Отмена]."""
    user, dream = user_and_dream
    cb = _fake_callback(f"dream_delete:{dream.id}")

    await handle_dream_delete_callback(cb, fsm, session=test_session)

    cb.answer.assert_awaited_once()
    cb.message.edit_text.assert_awaited_once()
    call = cb.message.edit_text.call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "Удалить сон" in text

    markup = call.kwargs.get("reply_markup")
    assert markup is not None
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert DREAM_DELETE_BTN_YES in buttons
    assert DREAM_DELETE_BTN_CANCEL in buttons


@pytest.mark.asyncio
async def test_step_2_confirm_shows_confirm_2_with_warning(test_session, user_and_dream, fsm) -> None:
    """Тап dream_delete_confirm:<id> → edit_text с DREAM_DELETE_CONFIRM_2 + [Да, навсегда | Отмена]."""
    user, dream = user_and_dream
    cb = _fake_callback(f"dream_delete_confirm:{dream.id}")

    await handle_dream_delete_confirm_callback(cb, fsm, session=test_session)

    cb.answer.assert_awaited_once()
    cb.message.edit_text.assert_awaited_once()
    call = cb.message.edit_text.call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "навсегда" in text

    markup = call.kwargs.get("reply_markup")
    assert markup is not None
    buttons = [btn.text for row in markup.inline_keyboard for btn in row]
    assert DREAM_DELETE_BTN_FINAL in buttons
    assert DREAM_DELETE_BTN_CANCEL in buttons


@pytest.mark.asyncio
async def test_step_3_final_hard_deletes_dream(test_session, user_and_dream, fsm) -> None:
    """Тап dream_delete_final:<id> → dream удалён, edit_text = DREAM_DELETED_NOTICE."""
    user, dream = user_and_dream
    dream_id = dream.id
    cb = _fake_callback(f"dream_delete_final:{dream_id}")

    await handle_dream_delete_final_callback(cb, fsm, session=test_session)

    cb.answer.assert_awaited_once()
    cb.message.edit_text.assert_awaited_once()
    call = cb.message.edit_text.call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert text == DREAM_DELETED_NOTICE

    # Сон действительно удалён
    dream_repo = DreamRepository(test_session)
    deleted = await dream_repo.get_by_id(dream_id=dream_id, user_id=user.id)
    assert deleted is None


@pytest.mark.asyncio
async def test_cancel_restores_dream_card(test_session, user_and_dream, fsm) -> None:
    """Тап dream_delete_cancel:<id> → сон НЕ удалён, edit_text показывает карточку с кнопками [↩|🗑]."""
    user, dream = user_and_dream
    dream_id = dream.id
    cb = _fake_callback(f"dream_delete_cancel:{dream_id}")

    await handle_dream_delete_cancel_callback(cb, fsm, session=test_session)

    cb.answer.assert_awaited_once()
    cb.message.edit_text.assert_awaited_once()
    call = cb.message.edit_text.call_args
    text = call.args[0] if call.args else call.kwargs.get("text", "")
    assert "тёмный лес" in text

    markup = call.kwargs.get("reply_markup")
    assert markup is not None
    buttons_data = [btn.callback_data for row in markup.inline_keyboard for btn in row]
    assert f"dream_open:{dream_id}" in buttons_data
    assert f"dream_delete:{dream_id}" in buttons_data

    # Сон жив
    dream_repo = DreamRepository(test_session)
    still_there = await dream_repo.get_by_id(dream_id=dream_id, user_id=user.id)
    assert still_there is not None


@pytest.mark.asyncio
async def test_delete_rejects_other_users_dream(test_session, user_and_dream, schema, fsm) -> None:
    """User B тапает dream_delete_final:<dream_A.id> → no-op, сон A остаётся."""
    user_a, dream = user_and_dream
    dream_id = dream.id

    # Создать user B (tg_id=99) — нужен в БД чтобы get_by_tg_id вернул его
    user_repo = UserRepository(test_session)
    await user_repo.get_or_create_by_tg_id(telegram_user_id=99, display_name="Боб")
    await test_session.commit()

    cb = _fake_callback(f"dream_delete_final:{dream_id}", user_id=99)
    await handle_dream_delete_final_callback(cb, fsm, session=test_session)

    # edit_text НЕ вызван (fetch вернул None из-за несовпадения user_id)
    cb.message.edit_text.assert_not_awaited()

    # Сон A остался
    dream_repo = DreamRepository(test_session)
    still_there = await dream_repo.get_by_id(dream_id=dream_id, user_id=user_a.id)
    assert still_there is not None
