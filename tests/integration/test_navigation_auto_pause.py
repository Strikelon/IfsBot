from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.dreams import handle_dream_by_id, handle_dreams_list
from adapters.telegram.states import QuickAnalysisStates
from adapters.telegram.texts import AUTO_PAUSED_NOTICE
from data.models import (
    Base,
    ExperienceLevel,
    SessionStatus,
    SessionType,
)
from data.repositories import DreamRepository, SessionRepository, UserRepository


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    image_questions_enabled = False  # регресс без фичи: карточка как до image-questions


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def user_with_active_qa(test_session, schema):
    """Onboarded user with one dream and IN_PROGRESS Quick session."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="x",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await test_session.commit()
    return user, dream, sess


def _message(text: str, user_id: int = 42) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=user_id, type="private")
    msg.text = text
    msg.answer = AsyncMock()
    return msg


def _make_state(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


@pytest.mark.asyncio
async def test_dreams_list_pauses_active_session(test_session, user_with_active_qa):
    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message("/dreams")
    await handle_dreams_list(msg, session=test_session, state=state)
    await test_session.commit()

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert AUTO_PAUSED_NOTICE in sent


@pytest.mark.asyncio
async def test_dream_card_other_dream_pauses_active(test_session, user_with_active_qa):
    user, dream, sess = user_with_active_qa
    # second dream
    other = await DreamRepository(test_session).create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 11, tzinfo=UTC),
        transcription="другой сон",
    )
    await test_session.commit()

    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message(f"/dream {other.id}")
    await handle_dream_by_id(msg, session=test_session, state=state, settings=_NoLimitSettings())
    await test_session.commit()

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert AUTO_PAUSED_NOTICE in sent


@pytest.mark.asyncio
async def test_dream_card_same_dream_does_not_pause(test_session, user_with_active_qa):
    """`/dream_<same_id>` on the dream of the active Q&A — does NOT pause."""
    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=state, settings=_NoLimitSettings())
    await test_session.commit()

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert AUTO_PAUSED_NOTICE not in sent


@pytest.mark.asyncio
async def test_profile_pauses_active_quick(test_session, user_with_active_qa):
    from adapters.telegram.handlers.profile import handle_profile

    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message("/profile")
    await handle_profile(msg, session=test_session, state=state)
    await test_session.commit()

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert AUTO_PAUSED_NOTICE in sent


@pytest.mark.asyncio
async def test_profile_edit_callback_pauses_active_quick(test_session, user_with_active_qa):
    """Spec §4.2: profile:edit:* callback паузит активную Quick-сессию."""
    from aiogram.types import CallbackQuery

    from adapters.telegram.handlers.profile import handle_profile_edit_callback

    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    cb.data = "profile:edit:life_context"
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()

    await handle_profile_edit_callback(cb, state=state, session=test_session)
    await test_session.commit()

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    sent = " || ".join(
        call.args[0] for call in cb.message.answer.await_args_list if call.args
    )
    assert AUTO_PAUSED_NOTICE in sent


def _pause_notice_removes_keyboard(answer_mock) -> bool:
    """AUTO_PAUSED_NOTICE отправлен с ReplyKeyboardRemove (клава «Завершить» снята)."""
    from aiogram.types import ReplyKeyboardRemove

    for c in answer_mock.await_args_list:
        if c.args and c.args[0] == AUTO_PAUSED_NOTICE:
            return isinstance(c.kwargs.get("reply_markup"), ReplyKeyboardRemove)
    return False


@pytest.mark.asyncio
async def test_profile_removes_finish_keyboard_on_pause(test_session, user_with_active_qa):
    """Регрессия: /profile во время Q&A снимает reply-клаву «Завершить» (не висит)."""
    from adapters.telegram.handlers.profile import handle_profile

    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message("/profile")
    await handle_profile(msg, session=test_session, state=state)

    assert _pause_notice_removes_keyboard(msg.answer)


@pytest.mark.asyncio
async def test_dreams_list_removes_finish_keyboard_on_pause(test_session, user_with_active_qa):
    """Регрессия: /dreams во время Q&A снимает reply-клаву «Завершить»."""
    user, dream, sess = user_with_active_qa
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _message("/dreams")
    await handle_dreams_list(msg, session=test_session, state=state)

    assert _pause_notice_removes_keyboard(msg.answer)
