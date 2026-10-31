"""Память v2: автоматика больше НЕ трогает profile_life_context (спека A2).

После удаления update_life_context профиль — только самоописание юзера:
- онбординг-разбор доставляется, но профиль остаётся байт-в-байт прежним;
- прямой запуск обычного разбора (_run_initial_quick) тоже не меняет профиль;
- usage-событий удалённого конвейера context.update.v1 не появляется.
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select

from adapters.telegram.handlers.onboarding import (
    _FSM_KEY_ONBOARDING_DREAM_ID,
    handle_per_session_context_onb,
)
from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from data.models import LlmUsageEvent, RequestBranch, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository

_DREAM_TEXT = "мне снился тёмный лес и волк"
_LIFE_CONTEXT = "меняю работу, нервничаю"
_PER_SESSION = "вчера был трудный день, поссорился с братом"
# Версия удалённого конвейера — литералом: константы больше не существует.
_LEGACY_CONTEXT_UPDATE_VERSION = "context.update.v1"


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_inject_enabled = False
    memory_write_enabled = False
    image_questions_enabled = False


@pytest_asyncio.fixture
async def fsm() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=42, user_id=42))


def _make_message(text: str) -> Message:
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    message.chat = Chat(id=42, type="private")
    message.text = text
    message.voice = None
    message.answer = AsyncMock(return_value=SimpleNamespace(message_id=888))
    return message


def _quick_llm_result() -> SimpleNamespace:
    return SimpleNamespace(
        text="quick output", input_tokens=100, output_tokens=50,
        error_category=None, model_used="google/gemini-3.5-flash",
    )


async def _make_user_with_dream(test_session):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context=_LIFE_CONTEXT)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription=_DREAM_TEXT
    )
    await test_session.commit()
    return user, dream


@pytest.mark.asyncio
async def test_onboarding_analysis_leaves_profile_untouched(test_session, fsm) -> None:
    user, dream = await _make_user_with_dream(test_session)
    await fsm.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm.update_data(**{_FSM_KEY_ONBOARDING_DREAM_ID: dream.id})

    message = _make_message(_PER_SESSION)
    with patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new=AsyncMock(return_value=_quick_llm_result()),
    ):
        await handle_per_session_context_onb(
            message, fsm, session=test_session, bot=AsyncMock(),
            openai_client=AsyncMock(), openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(), settings=_NoLimitSettings(),
        )

    # Поток жив: разбор доставлен, FSM в IN_QA.
    sent_texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("quick output" in t for t in sent_texts)
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # Профиль байт-в-байт прежний: per_session больше НЕ течёт в профиль.
    user_after = await UserRepository(test_session).get_by_tg_id(42)
    assert user_after.profile_life_context == _LIFE_CONTEXT

    # Ни одного usage-события удалённого конвейера.
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert [e for e in events if e.prompt_version == _LEGACY_CONTEXT_UPDATE_VERSION] == []


@pytest.mark.asyncio
async def test_quick_analysis_leaves_profile_untouched(test_session, fsm) -> None:
    from adapters.telegram.handlers.quick import _run_initial_quick

    user, dream = await _make_user_with_dream(test_session)
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context=_PER_SESSION)
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    await test_session.commit()

    message = _make_message("start")
    with patch(
        "adapters.telegram.handlers.quick._run_generator",
        new=AsyncMock(return_value=_quick_llm_result()),
    ):
        await _run_initial_quick(
            message=message, state=fsm, session=test_session, user_id=user.id,
            session_id=sess.id, openrouter_client=AsyncMock(),
            context_client=AsyncMock(), context_model="m",
            settings=_NoLimitSettings(), bot=None,
        )

    user_after = await UserRepository(test_session).get_by_tg_id(42)
    assert user_after.profile_life_context == _LIFE_CONTEXT
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert [e for e in events if e.prompt_version == _LEGACY_CONTEXT_UPDATE_VERSION] == []
