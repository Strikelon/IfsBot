"""End-to-end: во время thinking_status state установлен в GENERATING_*, после — восстановлен.

Проверяет что Task 7 wired thinking_status state-swap на каждом из 3 callsite'ов:
- quick.py:279  → GENERATING_INITIAL   (handle_branch_callback)
- quick.py:471  → GENERATING_FOLLOWUP  (_handle_qa_turn)
- onboarding.py:491 → GENERATING_INITIAL_ONB (_run_quick_for_onboarding)

Стратегия: патчим thinking_status и проверяем что handler вызывает его
с правильными state= и thinking_state= аргументами.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.onboarding import handle_per_session_context_onb
from adapters.telegram.handlers.quick import (
    handle_per_session_context_text,
    handle_quick_qa_text,
)
from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from data.models import Base, ExperienceLevel, RequestBranch, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Инжект памяти в промпты (Task 9) в этих тестах выключен.
    memory_inject_enabled = False
    # Флаг-офф: тест проверяет прямой callsite _run_initial_quick (обход offer-ветки).
    image_questions_enabled = False


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest.fixture
def fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


@pytest_asyncio.fixture
async def onboarded_user_and_dream(test_session, schema):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="меняю работу",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="мне снился тёмный лес",
    )
    await test_session.commit()
    return user, dream


@pytest_asyncio.fixture
async def onboarding_user_and_dream(test_session, schema):
    """User in onboarding flow (not yet onboarded)."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="ищу смысл",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    # NOT marking onboarded — this is the onboarding flow
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="я летел над городом",
    )
    await test_session.commit()
    return user, dream


def _callback(data: str) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _text_message(text: str) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock()
    return m


_MIN_CONTENT = 100  # must match generator._MIN_CONTENT_LEN


def _mock_or_response(text: str = "разбор...", p_tokens: int = 500, c_tokens: int = 100):
    padded = text if len(text) >= _MIN_CONTENT else text + " " + ("." * (_MIN_CONTENT - len(text)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=padded),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=p_tokens, completion_tokens=c_tokens),
        model_dump=lambda: {},
    )


# ---------------------------------------------------------------------------
# Helper: build fake thinking_status mock that captures kwargs
# ---------------------------------------------------------------------------


def _make_thinking_status_mock(captured: dict):
    """Returns an asynccontextmanager mock that records its call kwargs."""

    @asynccontextmanager
    async def _mock(message, **kwargs):
        captured.update(kwargs)
        yield message

    return _mock


# ---------------------------------------------------------------------------
# Test 1: quick.py:279 — handle_branch_callback → GENERATING_INITIAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_initial_callsite_swaps_to_generating_initial(
    test_session,
    onboarded_user_and_dream,
    fsm,
) -> None:
    """Branch tap → thinking_status called with state=fsm, thinking_state=GENERATING_INITIAL."""
    user, dream = onboarded_user_and_dream

    sess_repo = SessionRepository(test_session)
    q_sess = await sess_repo.create(
        user_id=user.id,
        dream_id=dream.id,
        type_=SessionType.QUICK,
    )
    await sess_repo.set_per_session_context(
        session_id=q_sess.id, context="вчера было сложное собрание"
    )
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.update_data(session_id=q_sess.id, dream_id=dream.id)

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(return_value=_mock_or_response())

    captured: dict = {}

    with patch(
        "adapters.telegram.handlers.quick.thinking_status",
        _make_thinking_status_mock(captured),
    ):
        msg = _text_message("вчера было сложное собрание")
        await handle_per_session_context_text(
            msg,
            fsm,
            session=test_session,
            openrouter_client=openrouter,
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    assert "thinking_state" in captured, "thinking_state not passed to thinking_status"
    assert captured["thinking_state"] == QuickAnalysisStates.GENERATING_INITIAL, (
        f"Expected GENERATING_INITIAL, got {captured.get('thinking_state')}"
    )
    assert captured.get("state") is fsm, "state param not passed (or wrong object)"


# ---------------------------------------------------------------------------
# Test 2: quick.py:471 — _handle_qa_turn (via handle_quick_qa_text) → GENERATING_FOLLOWUP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quick_followup_callsite_swaps_to_generating_followup(
    test_session,
    onboarded_user_and_dream,
    fsm,
) -> None:
    """Q&A followup → thinking_status called with thinking_state=GENERATING_FOLLOWUP."""
    user, dream = onboarded_user_and_dream

    sess_repo = SessionRepository(test_session)
    q_sess = await sess_repo.create(
        user_id=user.id,
        dream_id=dream.id,
        type_=SessionType.QUICK,
    )
    await sess_repo.set_request_branch(
        session_id=q_sess.id, branch=RequestBranch.FEEL_STATE
    )
    await sess_repo.set_per_session_context(
        session_id=q_sess.id, context="вчера было сложное собрание"
    )
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.update_data(session_id=q_sess.id, dream_id=dream.id)

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(return_value=_mock_or_response())
    classifier = AsyncMock()
    classifier.chat.completions.create = AsyncMock(return_value=_mock_or_response())

    captured: dict = {}

    with patch(
        "adapters.telegram.handlers.quick.thinking_status",
        _make_thinking_status_mock(captured),
    ):
        msg = _text_message("расскажи про тень")
        await handle_quick_qa_text(
            msg,
            fsm,
            session=test_session,
            bot=MagicMock(spec=Bot),
            openrouter_client=openrouter,
            openrouter_generator_model="google/gemini-flash",
            openrouter_classifier_client=classifier,
            openrouter_classifier_model="google/gemini-flash",
            openrouter_intent_model="openai/gpt-4o-mini",
            settings=_NoLimitSettings(),
        )

    assert "thinking_state" in captured, "thinking_state not passed to thinking_status"
    assert captured["thinking_state"] == QuickAnalysisStates.GENERATING_FOLLOWUP, (
        f"Expected GENERATING_FOLLOWUP, got {captured.get('thinking_state')}"
    )
    assert captured.get("state") is fsm, "state param not passed (or wrong object)"


# ---------------------------------------------------------------------------
# Test 3: onboarding.py:491 — _run_quick_for_onboarding → GENERATING_INITIAL_ONB
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_onboarding_initial_callsite_swaps_to_generating_initial_onb(
    test_session,
    onboarding_user_and_dream,
    fsm,
) -> None:
    """Onboarding branch tap → thinking_status called with thinking_state=GENERATING_INITIAL_ONB."""
    user, dream = onboarding_user_and_dream

    # The handler creates the session itself — we only pre-set FSM data with dream_id
    # and per_session_context (as the previous onboarding steps would have done).
    await fsm.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm.update_data(onboarding_dream_id=dream.id)

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(return_value=_mock_or_response())

    captured: dict = {}

    with patch(
        "adapters.telegram.handlers.onboarding.thinking_status",
        _make_thinking_status_mock(captured),
    ):
        msg = _text_message("было тревожно перед полётом")
        await handle_per_session_context_onb(
            msg,
            fsm,
            session=test_session,
            bot=MagicMock(spec=Bot),
            openai_client=AsyncMock(),
            openrouter_client=openrouter,
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    assert "thinking_state" in captured, "thinking_state not passed to thinking_status"
    assert captured["thinking_state"] == OnboardingStates.GENERATING_INITIAL_ONB, (
        f"Expected GENERATING_INITIAL_ONB, got {captured.get('thinking_state')}"
    )
    assert captured.get("state") is fsm, "state param not passed (or wrong object)"
