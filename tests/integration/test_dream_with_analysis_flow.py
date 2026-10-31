"""End-to-end: запись сна → разбор → Q&A → «Завершить» → /dream_<id> → auto-resume.

Тест проверяет ключевой UX-flow phase4-lite — что после разбора юзер может
вернуться к сну через /dream_<id>, увидеть свой разбор, и автоматически
попасть в Q&A без промежуточного тапа кнопки.
"""

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

from adapters.telegram.handlers.dreams import handle_dream_by_id
from adapters.telegram.states import QuickAnalysisStates
from data.constants import QUICK_PHASE_INITIAL
from data.models import (
    Base,
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


@pytest.mark.asyncio
async def test_dream_with_analysis_full_flow(test_session, schema) -> None:
    """Юзер записал сон → сделал разбор → закрыл → возвращается через /dream_<id> →
    видит сон + разбор → FSM автоматически в IN_QA (без тапа кнопки)."""
    # Setup: onboarded user
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="x",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)

    # Симулируем: сон записан, разбор сделан, юзер тапнул Завершить (PAUSED)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="мне снилась гора",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
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

    # Step 1: user does /dream_<id>
    msg = _message(f"/dream {dream.id}")
    state = _make_state()
    await handle_dream_by_id(msg, session=test_session, state=state, settings=_NoLimitSettings())

    # Bot sends multiple messages: card + initial + hint
    sent_texts = [call.args[0] for call in msg.answer.await_args_list]
    assert any("10 мая" in t for t in sent_texts)
    assert any("МОЙ-РАЗБОР-ПРО-ГОРУ" in t for t in sent_texts)

    # FSM автоматически в IN_QA без тапа кнопки
    assert (await state.get_state()) == QuickAnalysisStates.IN_QA.state
    data = await state.get_data()
    assert data["session_id"] == sess.id
    assert data["dream_id"] == dream.id

    # Session is IN_PROGRESS
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
