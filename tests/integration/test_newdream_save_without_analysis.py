"""End-to-end: /newdream → запись сна → «Это всё» → «Сохранить без разбора»
→ /dream_<id> показывает кнопку «Разобрать сейчас» (has_analysis=False).

Проверяет Task #4 — confirmation gate в /newdream flow.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, InlineKeyboardMarkup, Message
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.dreams import handle_dream_by_id
from adapters.telegram.handlers.newdream import (
    handle_newdream_confirm_analysis,
    handle_newdream_dream_loop,
)
from adapters.telegram.states import NewDreamStates
from adapters.telegram.texts import (
    DREAM_ACTION_START_ANALYSIS,
    DREAM_CONFIRM_ANALYZE_PROMPT,
    DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE,
    LOOP_ADD_BTN_DONE,
    SAVE_WITHOUT_ANALYSIS_BTN,
)
from data.models import Base, DreamStatus, ExperienceLevel
from data.models import Session as SessionModel
from data.repositories import DreamRepository, UserRepository


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


def _message(text: str, user_id: int = 42) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=user_id, type="private")
    msg.text = text
    msg.voice = None
    msg.answer = AsyncMock()
    return msg


def _make_fsm(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


class _Settings:
    """Лимит настроен, трат нет — spend_guard пропускает."""

    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0
    spend_limit_exempt_set = frozenset()


@pytest.mark.asyncio
async def test_save_without_analysis_then_dream_card_shows_analyze_button(
    test_session, schema
) -> None:
    """Полный flow:
    1. ASKING_DREAM → сон записан
    2. DREAM_LOOP «Это всё» → CONFIRMING_ANALYSIS (confirmation gate показан)
    3. «Сохранить без разбора» → FSM clear, dream.status=RECORDED
    4. /dream_<id> → карточка + кнопка «🔍 Разобрать сейчас» (нет разбора).
    """
    # Setup: онбординг-пройден юзер
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="текущий контекст",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    initial_dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про синий лес",
    )
    await test_session.commit()

    fsm = _make_fsm()
    await fsm.set_state(NewDreamStates.DREAM_LOOP)
    await fsm.update_data(newdream_dream_id=initial_dream.id)

    # Step 2: юзер тапает «Это всё» → confirmation gate (кнопка → классификатор не зовётся)
    msg_done = _message(LOOP_ADD_BTN_DONE)
    await handle_newdream_dream_loop(
        msg_done,
        fsm,
        session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini",
    )

    assert await fsm.get_state() == NewDreamStates.CONFIRMING_ANALYSIS.state
    texts_done = [call.args[0] for call in msg_done.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_done)

    # No Quick session created yet
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0

    # Step 3: юзер тапает «Сохранить без разбора»
    msg_save = _message(SAVE_WITHOUT_ANALYSIS_BTN)
    await handle_newdream_confirm_analysis(
        msg_save,
        fsm,
        session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini",
    )

    assert await fsm.get_state() is None
    texts_save = [call.args[0] for call in msg_save.answer.call_args_list]
    assert any(DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE in t for t in texts_save)

    # dream.status = RECORDED
    await test_session.refresh(initial_dream)
    assert initial_dream.status == DreamStatus.RECORDED

    # Still no sessions
    sessions_after = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions_after) == 0

    # Step 4: юзер открывает /dream_<id>
    msg_view = _message(f"/dream_{initial_dream.id}")
    view_fsm = _make_fsm()
    await handle_dream_by_id(msg_view, session=test_session, state=view_fsm, settings=_NoLimitSettings())

    # Бот отправил карточку с кнопкой «Разобрать сейчас»
    call_args_list = msg_view.answer.call_args_list
    assert len(call_args_list) >= 1

    # Найдём вызов с InlineKeyboardMarkup (кнопка «Разобрать»)
    inline_calls = [
        c for c in call_args_list
        if c.kwargs.get("reply_markup") is not None
        and isinstance(c.kwargs["reply_markup"], InlineKeyboardMarkup)
    ]
    assert len(inline_calls) >= 1, (
        f"Expected InlineKeyboardMarkup in one of {len(call_args_list)} answer calls"
    )
    # Текст кнопки должен быть «Разобрать сейчас»
    markup: InlineKeyboardMarkup = inline_calls[0].kwargs["reply_markup"]
    button_texts = [btn.text for row in markup.inline_keyboard for btn in row]
    assert DREAM_ACTION_START_ANALYSIS in button_texts, (
        f"Expected '{DREAM_ACTION_START_ANALYSIS}' in {button_texts}"
    )

    # Карточка без разбора → DreamCardStates.DECIDING_ANALYZE (свободный текст/голос
    # тоже запускает разбор, не только тап кнопки).
    from adapters.telegram.states import DreamCardStates

    assert await view_fsm.get_state() == DreamCardStates.DECIDING_ANALYZE.state
    assert (await view_fsm.get_data()).get("dream_card_dream_id") == initial_dream.id
