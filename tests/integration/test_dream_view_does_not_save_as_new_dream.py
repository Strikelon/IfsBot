"""Главный регресс для bug #2:
Записал сон → разобрал → «Завершить» → /dream_<id> → пишу текст «вопрос»
→ assert: новых dreams нет, FSM в IN_QA, текст ушёл бы в Q&A."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from adapters.telegram.handlers.dreams import handle_dream_by_id
from adapters.telegram.states import QuickAnalysisStates
from data.constants import QUICK_PHASE_INITIAL
from data.models import (
    Base,
    Dream,
    ExperienceLevel,
    LlmRole,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
)
from data.repositories import (
    DreamRepository,
    SessionMessageRepository,
    SessionRepository,
    UserRepository,
)


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Task 15: _resume_qa_for_dream теперь читает флаг фичи image questions.
    image_questions_enabled = True


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
    msg.answer = AsyncMock()
    return msg


def _make_state(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


@pytest_asyncio.fixture
async def onboarded_user_with_completed_quick(test_session: AsyncSession, schema):
    """Создаёт: onboarded user, dream, Quick-сессию PAUSED с initial-разбором.

    Returns (user, dream, paused_session).
    """
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
        transcription="мне снилась гора",
    )

    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)

    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="МОЙ-РАЗБОР-ПРО-ГОРУ",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    return user, dream, sess


@pytest.mark.asyncio
async def test_dream_view_with_analysis_resumes_qa_not_saves_new_dream(
    test_session: AsyncSession,
    onboarded_user_with_completed_quick,
) -> None:
    """При открытии /dream_<id> с разбором → FSM выставлен в IN_QA, новых снов нет."""
    user, dream, paused_session = onboarded_user_with_completed_quick

    dreams_before = (
        await test_session.execute(select(Dream).where(Dream.user_id == user.id))
    ).scalars().all()
    assert len(dreams_before) == 1

    # /dream_<id>
    message = _message(f"/dream_{dream.id}", user_id=user.telegram_user_id)
    state = _make_state(user_id=user.telegram_user_id)
    await handle_dream_by_id(message, state, test_session, settings=_NoLimitSettings())

    # ASSERT: FSM выставлен в IN_QA с правильными data
    fsm_state = await state.get_state()
    assert fsm_state == QuickAnalysisStates.IN_QA.state, f"expected IN_QA, got {fsm_state}"

    data = await state.get_data()
    assert data["dream_id"] == dream.id
    assert data["session_id"] == paused_session.id  # переиспользовали PAUSED-сессию

    # ASSERT: session снова IN_PROGRESS
    await test_session.refresh(paused_session)
    assert paused_session.status == SessionStatus.IN_PROGRESS

    # ASSERT: новых dreams не появилось
    dreams_after = (
        await test_session.execute(select(Dream).where(Dream.user_id == user.id))
    ).scalars().all()
    assert len(dreams_after) == 1
