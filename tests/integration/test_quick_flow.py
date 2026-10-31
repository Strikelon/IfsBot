"""End-to-end Quick flow с моками внешних сервисов."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.quick import (
    handle_continue_quick_callback,
    handle_finish_quick_reply,
    handle_per_session_context_text,
    handle_quick_qa_text,
    handle_start_analysis_callback,
)
from adapters.telegram.states import QuickAnalysisStates
from data.models import (
    Base,
    DreamStatus,
    ExperienceLevel,
    LlmService,
    LlmUsageEvent,
    Session,
    SessionStatus,
    SessionType,
)
from data.repositories import DreamRepository, UserRepository


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Инжект памяти в промпты (Task 9) в этих тестах выключен.
    memory_inject_enabled = False
    # Флаг-офф: эти тесты фиксируют байт-в-байт поведение обычного потока до
    # фичи image questions (обход offer-ветки в _run_quick_after_context).
    image_questions_enabled = False


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def user_and_dream(test_session, schema):
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


@pytest.fixture
def fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _callback(data: str) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _text(text: str) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock()
    return m


_MIN_CONTENT = 100  # must match generator._MIN_CONTENT_LEN


def _mock_or_response(text: str, p_tokens: int = 500, c_tokens: int = 100):
    # Pad content to pass min-length validation in generator fallback chain.
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


@pytest.mark.asyncio
async def test_full_quick_flow_from_callback_to_closing(
    test_session, user_and_dream, fsm
) -> None:
    user, dream = user_and_dream

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        side_effect=[
            _mock_or_response("первичный разбор", 1000, 200),
            _mock_or_response("уточнение 1", 600, 100),
            _mock_or_response("закрывающее", 400, 80),
        ]
    )
    model = "google/gemini-3-pro"

    # 1. Юзер тапает «Разобрать сейчас»
    cb_start = _callback(f"start_analysis:{dream.id}")
    await handle_start_analysis_callback(cb_start, fsm, session=test_session, settings=_NoLimitSettings())
    assert await fsm.get_state() == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state

    # 2. Отвечает текстом → ветка убрана: сразу первичный разбор → IN_QA
    msg = _text("вчера было сложное собрание")
    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # 5. Юзер задаёт уточнение
    msg2 = _text("а если послушать тень?")
    classifier = AsyncMock()
    await handle_quick_qa_text(
        msg2,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=openrouter,
        openrouter_generator_model=model,
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )
    # quick.v2.1: qa_turns_used убран; источник истины — session_messages
    fsm_data = await fsm.get_data()
    assert "qa_turns_used" not in fsm_data

    # 6. Юзер нажимает reply-кнопку «Завершить»
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    sess_id = sessions[0].id

    finish_msg = _text("Завершить")
    await handle_finish_quick_reply(finish_msg, fsm, session=test_session, bot=MagicMock(spec=Bot))
    assert await fsm.get_state() is None

    # Финальное состояние БД:
    await test_session.refresh(sessions[0])
    await test_session.refresh(dream)
    assert sessions[0].status == SessionStatus.PAUSED
    assert sessions[0].request_branch is not None
    assert sessions[0].per_session_context == "вчера было сложное собрание"
    assert dream.status == DreamStatus.IN_ANALYSIS

    # Учёт: 2 OpenRouter-вызова × 2 события + 0 Whisper (всё текстом) = 4 события.
    # initial-разбор пишется как "quick.v3.15", Q&A — "quick.v3.13.followup".
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    or_events = [e for e in events if e.service == LlmService.OPENROUTER]
    assert len(or_events) == 4
    versions = {e.prompt_version for e in or_events}
    assert versions == {"quick.v3.15", "quick.v3.13.followup"}
    assert all(e.session_id == sess_id for e in or_events)
    # 3-й мок (закрывающее) не был использован — юзер нажал reply «Завершить» раньше
    assert openrouter.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_full_quick_flow_no_qa_limit(
    test_session, user_and_dream, fsm
) -> None:
    """quick.v2.1: QA_LIMIT снят — после 5 (и любого N) turn'ов сессия не закрывается."""
    user, dream = user_and_dream

    openrouter = AsyncMock()
    # 1 первичный + 5 follow-up = 6 ответов
    openrouter.chat.completions.create = AsyncMock(
        side_effect=[_mock_or_response(f"ответ #{i}", 500, 100) for i in range(6)]
    )
    model = "google/gemini-3-pro"
    classifier = AsyncMock()

    # 1-4: до IN_QA
    await handle_start_analysis_callback(
        _callback(f"start_analysis:{dream.id}"), fsm, session=test_session,
        settings=_NoLimitSettings(),
    )
    await handle_per_session_context_text(
        _text("контекст"), fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=classifier,
        settings=_NoLimitSettings(),
    )
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # 5 turn'ов QA
    for i in range(5):
        await handle_quick_qa_text(
            _text(f"вопрос {i}"),
            fsm,
            session=test_session,
            bot=MagicMock(spec=Bot),
            openrouter_client=openrouter,
            openrouter_generator_model=model,
            openrouter_classifier_client=classifier,
            openrouter_classifier_model="google/gemini-2.5-flash",
            openrouter_intent_model="openai/gpt-4o-mini",
            settings=_NoLimitSettings(),
        )

    # quick.v2.1: после 5-го turn'а сессия остаётся IN_PROGRESS, FSM в IN_QA
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert sessions[0].status == SessionStatus.IN_PROGRESS
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS

    # 1 первичный + 5 follow-up = 6 LLM-вызовов, × 2 события = 12 generator-событий
    # model recorded is now _QUICK_FALLBACK_CHAIN[0] (generator owns chain)
    from llm.generator import _QUICK_FALLBACK_CHAIN
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    or_events = [
        e for e in events
        if e.service == LlmService.OPENROUTER and e.model == _QUICK_FALLBACK_CHAIN[0]
    ]
    assert len(or_events) == 12


@pytest.mark.asyncio
async def test_reply_finish_button_pauses_session(
    test_session, user_and_dream, fsm
) -> None:
    """Reply-кнопка «Завершить» в IN_QA → sess.status=PAUSED, FSM очищен, dream.status=IN_ANALYSIS."""
    from aiogram.types import ReplyKeyboardRemove

    user, dream = user_and_dream

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        side_effect=[_mock_or_response("первичный разбор", 1000, 200)]
    )

    # 1. Доходим до IN_QA
    await handle_start_analysis_callback(
        _callback(f"start_analysis:{dream.id}"), fsm, session=test_session,
        settings=_NoLimitSettings(),
    )
    await handle_per_session_context_text(
        _text("контекст"), fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # 2. Юзер нажимает reply-кнопку «Завершить»
    finish_msg = _text("Завершить")
    await handle_finish_quick_reply(finish_msg, fsm, session=test_session, bot=MagicMock(spec=Bot))

    # FSM очищен
    assert await fsm.get_state() is None

    # Сессия PAUSED, dream.status остаётся IN_ANALYSIS
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    await test_session.refresh(sessions[0])
    await test_session.refresh(dream)
    assert sessions[0].status == SessionStatus.PAUSED
    assert dream.status == DreamStatus.IN_ANALYSIS

    # Бот ответил с ReplyKeyboardRemove
    finish_msg.answer.assert_awaited_once()
    call_kwargs = finish_msg.answer.await_args.kwargs
    assert isinstance(call_kwargs.get("reply_markup"), ReplyKeyboardRemove)


@pytest.mark.asyncio
async def test_continue_quick_resumes_paused_session(
    test_session, user_and_dream, fsm
) -> None:
    """callback continue_quick:<dream_id> при PAUSED сессии → sess.status=IN_PROGRESS, FSM=IN_QA."""
    from data.repositories import SessionRepository

    user, dream = user_and_dream

    # Создаём и сразу ставим сессию на PAUSED, имитируя что пользователь уже нажал «Завершить»
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    # Также нужно добавить initial bot message чтобы history не была пустой
    from adapters.telegram.session_log import QUICK_PHASE_INITIAL, log_bot_message
    await log_bot_message(
        test_session, session_id=sess.id, content="первичный разбор", phase_label=QUICK_PHASE_INITIAL
    )
    await test_session.commit()

    # Юзер возвращается через continue_quick
    cb = _callback(f"continue_quick:{dream.id}")
    await handle_continue_quick_callback(cb, fsm, session=test_session, settings=_NoLimitSettings())

    # FSM должен быть в IN_QA
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    fsm_data = await fsm.get_data()
    assert fsm_data.get("session_id") == sess.id

    # Сессия обратно IN_PROGRESS
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
