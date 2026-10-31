"""E2E onboarding flow tests (post-Quick-Only simplification).

Walks the full FSM from /start through inline Q&A and «Завершить» completion.
Phase 3: after Quick output FSM hands off to QuickAnalysisStates.IN_QA;
onboarding completes only on first «Завершить» tap.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.onboarding import (
    handle_dream_input,
    handle_dream_loop,
    handle_life_context_input,
    handle_life_context_loop,
    handle_per_session_context_onb,
)
from adapters.telegram.handlers.quick import handle_finish_quick_reply
from adapters.telegram.handlers.start import handle_start
from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import (
    END_OF_ONBOARDING_TEXT,
    FINISH_BUTTON_TEXT,
    LOOP_ADD_BTN_DONE,
)
from data.models import (
    Base,
    Dream,
    DreamStatus,
    RequestBranch,
    Session,
    SessionStatus,
    SessionType,
    User,
)


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Инжект памяти в промпты (Task 9) в этих тестах выключен.
    memory_inject_enabled = False
    # Флаг-офф: _build_quick_context (quick.v3.14) читает флаг фичи image questions.
    image_questions_enabled = False


# === Fixtures ===


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def fsm() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


# === Helpers ===


_TG_USER = SimpleNamespace(
    id=42,
    is_bot=False,
    first_name="Александр",
    username=None,
    last_name=None,
    language_code="ru",
    full_name="Александр",
)


def _make_message(text: str | None = None) -> Message:
    """AsyncMock Message with the given text. from_user/chat are stable."""
    message = AsyncMock(spec=Message)
    message.from_user = _TG_USER
    message.chat = Chat(id=42, type="private")
    message.text = text
    message.voice = None
    message.answer = AsyncMock()
    return message


def _make_callback(data: str) -> CallbackQuery:
    """AsyncMock CallbackQuery; `.message.answer` is also an AsyncMock."""
    cb = AsyncMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _TG_USER
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _quick_llm_result() -> SimpleNamespace:
    """Stand-in for GeneratorResult — same duck-typed fields."""
    return SimpleNamespace(text="quick output", input_tokens=100, output_tokens=50, error_category=None, model_used="google/gemini-3.5-flash")


# === Test: Quick-only onboarding path ===


@pytest.mark.asyncio
async def test_full_onboarding_quick_only_completes(
    test_session, schema, fsm: FSMContext
) -> None:
    """Full onboarding: /start → dream → life_context → per-session → Quick LLM
    → IN_QA handoff → Q&A → «Завершить» → onboarding_completed_at set + menu activated.

    Новый порядок: сначала просим сон, знакомство (life_context) — после сна.
    Phase 3: onboarding completes only on first «Завершить» tap, not inline after Quick.
    """
    bot = MagicMock(spec=Bot)
    bot.set_my_commands = AsyncMock()
    openai_client = AsyncMock()
    openrouter_client = AsyncMock()

    # --- Step 1: /start ---
    start_msg = _make_message("/start")
    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ) as start_cmds:
        await handle_start(start_msg, fsm, session=test_session, bot=bot)

    users = (await test_session.execute(select(User))).scalars().all()
    assert len(users) == 1
    user = users[0]
    assert user.display_name == "Александр"

    assert await fsm.get_state() == OnboardingStates.CHOOSING_ENTRY_TYPE.state
    start_cmds.assert_awaited_once()
    assert start_cmds.await_args.kwargs["onboarded"] is False

    # Новый экран выбора: симулируем тап «Сон» — дальше флоу сна БЕЗ изменений.
    from adapters.telegram.handlers.entry_choice import handle_entry_choice_callback

    cb = AsyncMock()
    cb.from_user = start_msg.from_user
    cb.data = "entry_type:dream"
    cb.message = AsyncMock()
    cb.message.chat = SimpleNamespace(id=42, type="private")
    cb.message.answer = AsyncMock()
    await handle_entry_choice_callback(cb, fsm, session=test_session, bot=None)
    assert await fsm.get_state() == OnboardingStates.ASKING_DREAM.state

    # --- Step 2: dream input (сначала сон) ---
    dream_msg = _make_message("приснилось, что лечу над лесом")
    await handle_dream_input(
        dream_msg, fsm, session=test_session, bot=bot, openai_client=openai_client,
        settings=_NoLimitSettings(),
    )
    dreams = (await test_session.execute(select(Dream))).scalars().all()
    assert len(dreams) == 1
    dream = dreams[0]
    assert dream.transcription == "приснилось, что лечу над лесом"
    assert await fsm.get_state() == OnboardingStates.DREAM_LOOP.state

    # --- Step 3: dream loop DONE → шаг знакомства ---
    dream_done = _make_message(LOOP_ADD_BTN_DONE)
    await handle_dream_loop(
        dream_done, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini",
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_LIFE_CONTEXT.state

    # --- Step 4: life_context input ---
    lc_msg = _make_message("меняю работу, нервничаю")
    await handle_life_context_input(
        lc_msg, fsm, session=test_session, bot=bot, openai_client=openai_client,
        settings=_NoLimitSettings(),
    )
    await test_session.refresh(user)
    assert user.profile_life_context == "меняю работу, нервничаю"
    assert await fsm.get_state() == OnboardingStates.LIFE_CONTEXT_LOOP.state

    # --- Step 5: life_context loop DONE → per-session вопрос перед разбором ---
    lc_done = _make_message(LOOP_ADD_BTN_DONE)
    await handle_life_context_loop(
        lc_done, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini",
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state

    # --- Step 7: per-session context → onboarding Quick LLM → handoff IN_QA (ветка убрана) ---
    psc_msg = _make_message("волнительный день")
    with patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new=AsyncMock(return_value=_quick_llm_result()),
    ):
        await handle_per_session_context_onb(
            psc_msg, fsm, session=test_session, bot=bot, openai_client=openai_client,
            openrouter_client=openrouter_client, openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    sess = sessions[0]
    assert sess.type == SessionType.QUICK
    assert sess.is_onboarding_session is True
    assert sess.request_branch == RequestBranch.UNCLEAR  # дефолт вместо вопроса
    assert sess.per_session_context == "волнительный день"

    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS

    # Phase 3: after Quick output → FSM = IN_QA (NOT cleared)
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state, (
        "After Quick output in onboarding, FSM should be IN_QA (not cleared)"
    )

    # onboarding NOT yet completed — only completes on first «Завершить»
    await test_session.refresh(user)
    assert user.onboarding_completed_at is None, (
        "onboarding_completed_at should be None until user taps «Завершить»"
    )

    # Session still IN_PROGRESS (not completed or paused yet)
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS

    # --- Step 9: user taps «Завершить» → onboarding completes ---
    finish_msg = _make_message(FINISH_BUTTON_TEXT)
    with patch(
        "adapters.telegram.handlers.quick.sync_user_commands",
        new=AsyncMock(),
    ) as menu_update:
        await handle_finish_quick_reply(
            finish_msg, fsm, session=test_session, bot=bot
        )

    # onboarding_completed_at is now set
    await test_session.refresh(user)
    assert user.onboarding_completed_at is not None, (
        "onboarding_completed_at should be set after «Завершить»"
    )

    # Menu was activated via sync_user_commands(onboarded=True)
    menu_update.assert_called_once()
    call_kwargs = menu_update.call_args.kwargs
    assert call_kwargs["onboarded"] is True
    assert call_kwargs["user"].telegram_user_id == user.telegram_user_id

    # Session = PAUSED
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED

    # FSM cleared
    assert await fsm.get_state() is None

    # Bot sent END_OF_ONBOARDING_TEXT (not SESSION_PAUSED_NOTICE)
    finish_sent_texts = [c.args[0] for c in finish_msg.answer.await_args_list]
    assert any(END_OF_ONBOARDING_TEXT in t for t in finish_sent_texts), (
        f"Expected END_OF_ONBOARDING_TEXT in sent messages, got: {finish_sent_texts}"
    )
