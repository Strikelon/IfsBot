"""Unit-тесты Quick-handlers. Голосовая часть — Whisper мокается."""

from __future__ import annotations

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
    handle_finish_quick_reply,
    handle_per_session_context_text,
    handle_per_session_context_voice,
    handle_quick_qa_text,
    handle_quick_qa_voice,
    handle_start_analysis_callback,
)
from adapters.telegram.states import QuickAnalysisStates
from adapters.telegram.texts import QUICK_ALREADY_ACTIVE
from data.models import (
    Base,
    DreamStatus,
    ExperienceLevel,
    LlmService,
    LlmUsageEvent,
    RequestBranch,
    Session,
    SessionStatus,
    SessionType,
    UsageMetric,
)
from data.models import DreamStatus as _DS
from data.repositories import DreamRepository, SessionRepository, UserRepository

_MIN_CONTENT = 100  # must match generator._MIN_CONTENT_LEN


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Экстрактор памяти на паузе (Task 6) в этих тестах выключен.
    memory_write_enabled = False
    # Инжект памяти в промпты (Task 9) в этих тестах выключен.
    memory_inject_enabled = False
    # Флаг-офф: эти тесты фиксируют байт-в-байт поведение обычного потока до
    # фичи image questions (обход offer-ветки в _run_quick_after_context).
    image_questions_enabled = False


def _llm_response(content: str, prompt_tokens: int = 500, completion_tokens: int = 100):
    """Mock LLM response that passes min-length check in fallback chain."""
    padded = content if len(content) >= _MIN_CONTENT else content + " " + ("." * (_MIN_CONTENT - len(content)))
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=padded),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        model_dump=lambda: {},
    )


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def onboarded_with_dream(test_session, schema):
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


def _callback(data: str, message_id: int = 100) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.message_id = message_id
    cb.message.answer = AsyncMock()
    cb.message.from_user = TgUser(id=1, is_bot=True, first_name="Bot")  # bot, not user
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


def _voice(file_id: str = "v1") -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = None
    m.voice = SimpleNamespace(file_id=file_id, duration=10, mime_type="audio/ogg")
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_build_quick_context_reads_custom_branch_from_session_state(
    test_session, onboarded_with_dream
) -> None:
    """branch=CUSTOM → request_branch_custom подтягивается из session_state."""
    from adapters.telegram.handlers.quick import CUSTOM_BRANCH_KEY, _build_quick_context
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.CUSTOM)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id, data={CUSTOM_BRANCH_KEY: "понять связь сна с разводом"}
    )
    await test_session.commit()
    # update_profile в фикстуре пишет core-UPDATE'ом и не синкает in-memory user
    # (в проде ветку выбирают отдельным update'ом со свежим SELECT). Перечитываем user.
    await test_session.refresh(user)

    ctx = await _build_quick_context(
        session=test_session, user_id=user.id, session_id=sess.id, settings=_NoLimitSettings()
    )
    assert ctx is not None
    assert ctx.request_branch == RequestBranch.CUSTOM
    assert ctx.request_branch_custom == "понять связь сна с разводом"
    # Флаг инжекта выключен в стабе → память не подставляется
    assert ctx.memory_block == ""


@pytest.mark.asyncio
async def test_build_quick_context_injects_memory_when_flag_on(
    test_session, onboarded_with_dream
) -> None:
    """memory_inject_enabled=True → активные единицы памяти собираются в memory_block."""
    from adapters.telegram.handlers.quick import _build_quick_context
    from data.repositories import UserMemoryItemRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await UserMemoryItemRepository(test_session).add(
        user_id=user.id,
        kind="part",
        text="Внутренний Критик — включается на работе",
        realm="reality",
        source="qa",
    )
    await test_session.commit()
    await test_session.refresh(user)

    class _InjectSettings(_NoLimitSettings):
        memory_inject_enabled = True

    ctx = await _build_quick_context(
        session=test_session, user_id=user.id, session_id=sess.id, settings=_InjectSettings()
    )
    assert ctx is not None
    assert "Внутренний Критик" in ctx.memory_block
    assert ctx.memory_block.startswith("ЧТО УЖЕ ИЗВЕСТНО")


@pytest.mark.asyncio
async def test_start_analysis_creates_session_and_asks_context(
    test_session, onboarded_with_dream, fsm
) -> None:
    user, dream = onboarded_with_dream
    cb = _callback(f"start_analysis:{dream.id}")

    await handle_start_analysis_callback(cb, fsm, session=test_session, settings=_NoLimitSettings())

    # Сессия создана, dream — in_analysis
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    assert sessions[0].dream_id == dream.id
    assert sessions[0].user_id == user.id
    assert sessions[0].status == SessionStatus.IN_PROGRESS
    # type ещё не выбран — null или quick по дефолту? По плану — выставляется при mode-callback.
    # Пока договариваемся: создаём с type=QUICK по дефолту (Phase 2 только Quick).
    assert sessions[0].type == SessionType.QUICK

    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS

    # FSM
    state = await fsm.get_state()
    assert state == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    data = await fsm.get_data()
    assert data["dream_id"] == dream.id
    assert data["session_id"] == sessions[0].id

    cb.answer.assert_awaited_once()  # «processed» ack для Telegram
    cb.message.answer.assert_awaited_once()
    sent = cb.message.answer.await_args.args[0]
    assert "что больше всего вас волновало" in sent.lower()


@pytest.mark.asyncio
async def test_start_analysis_rejected_when_active_session_exists(
    test_session, onboarded_with_dream, fsm
) -> None:
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    cb = _callback(f"start_analysis:{dream.id}")
    await handle_start_analysis_callback(cb, fsm, session=test_session, settings=_NoLimitSettings())

    # Не создаём вторую сессию
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1

    # FSM пустой
    assert await fsm.get_state() is None
    cb.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_per_session_context_text_runs_analysis(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Шаг ветки убран: per-session ответ → дефолт UNCLEAR + сразу разбор → IN_QA."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("разбор сна", prompt_tokens=800, completion_tokens=150)
    )

    msg = _text("вчера было сложное собрание")
    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    await test_session.refresh(sess)
    assert sess.per_session_context == "вчера было сложное собрание"
    assert sess.request_branch == RequestBranch.UNCLEAR  # дефолт вместо вопроса
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    openrouter.chat.completions.create.assert_awaited_once()
    msg.answer.assert_awaited()


@pytest.mark.asyncio
async def test_per_session_context_voice_uses_whisper(
    test_session, onboarded_with_dream, fsm
) -> None:
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    bot = AsyncMock()

    async def fake_dl(file, destination):
        destination.write(b"\x00" * 100)

    bot.download = AsyncMock(side_effect=fake_dl)
    openai = AsyncMock()
    openai.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="вчера трудный день", duration=8.0)
    )
    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("разбор", prompt_tokens=800, completion_tokens=150)
    )

    msg = _voice()
    await handle_per_session_context_voice(
        msg, fsm, session=test_session, bot=bot, openai_client=openai,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    await test_session.refresh(sess)
    assert sess.per_session_context == "вчера трудный день"
    assert sess.request_branch == RequestBranch.UNCLEAR
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    openrouter.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_per_session_runs_llm_records_usage_and_replies(
    test_session, onboarded_with_dream, fsm
) -> None:
    """per-session → разбор: записаны session_message + usage, юзеру прислан ответ + «Завершить»."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("Похоже, в сне…", prompt_tokens=1000, completion_tokens=200)
    )

    msg = _text("трудный день")
    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    from data.models import SessionMessage

    db_msgs = (
        (
            await test_session.execute(
                select(SessionMessage).order_by(SessionMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    assert len(db_msgs) == 1
    assert db_msgs[0].content.startswith("Похоже, в сне…")
    assert db_msgs[0].llm_phase_label == "initial"

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    or_events = [
        e for e in events
        if e.service == LlmService.OPENROUTER and e.prompt_version == "quick.v3.15"
    ]
    assert len(or_events) == 2
    by_metric = {e.usage_metric: e for e in or_events}
    assert by_metric[UsageMetric.INPUT_TOKENS].usage_amount == 1000
    assert by_metric[UsageMetric.OUTPUT_TOKENS].usage_amount == 200

    msg.answer.assert_awaited()
    last = msg.answer.await_args
    assert "Похоже, в сне…" in last.args[0]
    assert last.kwargs.get("reply_markup") is not None


@pytest.mark.asyncio
async def test_quick_qa_text_writes_pair_to_db(test_session, onboarded_with_dream, fsm) -> None:
    """quick.v2.1 (E2): Q&A пара пишется в session_messages, не в FSM."""
    from adapters.telegram.session_log import log_bot_message

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    # Initial разбор уже был — пишем в БД как conftest
    await log_bot_message(
        test_session, session_id=sess.id, content="первый ответ бота", phase_label="initial"
    )
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("следующий ответ", prompt_tokens=500, completion_tokens=100)
    )
    classifier = AsyncMock()
    msg = _text("а если послушать тень?")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=openrouter,
        openrouter_generator_model="google/gemini-3-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # qa_history больше не в FSM
    fsm_data = await fsm.get_data()
    assert "qa_history" not in fsm_data
    assert "qa_turns_used" not in fsm_data
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # А вот в session_messages — добавились user-вопрос и bot-ответ
    from data.models import SessionMessage, SessionMessageDirection

    db_msgs = (
        (
            await test_session.execute(
                select(SessionMessage).order_by(SessionMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    # initial + user-вопрос + bot-ответ = 3
    assert len(db_msgs) == 3
    assert db_msgs[1].direction == SessionMessageDirection.USER_TO_BOT
    assert db_msgs[1].content == "а если послушать тень?"
    assert db_msgs[2].direction == SessionMessageDirection.BOT_TO_USER
    # content may be padded to meet min-length check; check prefix
    assert db_msgs[2].content.startswith("следующий ответ")
    assert db_msgs[2].llm_phase_label == "qa"


@pytest.mark.asyncio
async def test_quick_qa_no_session_auto_close_after_N_turns(
    test_session, onboarded_with_dream, fsm
) -> None:
    """quick.v2.1: QA_LIMIT снят. На 5-м (и любом N-м) turn'е сессия остаётся IN_PROGRESS,
    кнопки «Перейти в глубокий разбор» / «Завершить» работают как и раньше."""
    from adapters.telegram.session_log import log_bot_message, log_user_message

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await log_bot_message(
        test_session, session_id=sess.id, content="первый", phase_label="initial"
    )
    # Записываем 4 уже произошедших пары
    for i in range(4):
        await log_user_message(test_session, session_id=sess.id, content=f"u{i}")
        await log_bot_message(
            test_session, session_id=sess.id, content=f"a{i}", phase_label="qa"
        )
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ-5", prompt_tokens=400, completion_tokens=80)
    )
    classifier = AsyncMock()

    msg = _text("5-й вопрос")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=openrouter,
        openrouter_generator_model="google/gemini-3-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # На 5-м turn'е сессия НЕ закрывается (это quick.v2.1 — лимит снят)
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state


def _intent_client(outcome: str) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f'{{"outcome": "{outcome}", "reason": "x"}}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=5),
        )
    )
    return client


@pytest.mark.asyncio
async def test_qa_finish_intent_pauses_session_without_llm(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Точка 4: свободная реплика «спасибо, всё» → классификатор finish → пауза, без generator."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock()
    classifier = _intent_client("finish")

    msg = _text("спасибо, на сегодня всё")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # Сессия на паузе, FSM очищен, generator НЕ вызывался
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert await fsm.get_state() is None
    generator.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_qa_intent_usage_is_recorded(test_session, onboarded_with_dream, fsm) -> None:
    """Q&A intent-вызов (gpt-4o-mini) попадает в llm_usage_events с версией intent.qa.v1."""
    from adapters.telegram.session_log import log_bot_message
    from llm.intent import INTENT_QA_VERSION

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await log_bot_message(
        test_session, session_id=sess.id, content="initial", phase_label="initial"
    )
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ", prompt_tokens=400, completion_tokens=80)
    )
    classifier = _intent_client("continue")  # вернёт токены 40/5

    await handle_quick_qa_text(
        _text("да, это про работу наверное"),
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    intent_events = [e for e in events if e.prompt_version == INTENT_QA_VERSION]
    assert len(intent_events) == 2  # input + output
    by_metric = {e.usage_metric: e for e in intent_events}
    assert by_metric[UsageMetric.INPUT_TOKENS].usage_amount == 40
    assert by_metric[UsageMetric.INPUT_TOKENS].model == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_qa_new_dream_intent_closes_current_and_starts_newdream(
    test_session, onboarded_with_dream, fsm
) -> None:
    """В IN_QA «давай новый сон» → текущая сессия на паузу, FSM → NewDream ASKING_DREAM."""
    from adapters.telegram.states import NewDreamStates

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock()
    classifier = _intent_client("new_dream")

    await handle_quick_qa_text(
        _text("а давай разберём новый сон"),
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # Текущая сессия закрыта (paused), генератор не вызывался, FSM → сбор нового сна
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    generator.chat.completions.create.assert_not_awaited()
    assert await fsm.get_state() == NewDreamStates.ASKING_DREAM.state


@pytest.mark.asyncio
async def test_qa_new_dream_intent_runs_memory_extraction_like_command(
    test_session, onboarded_with_dream, fsm, monkeypatch
) -> None:
    """Паритет с /newdream: свободная фраза «новый сон» тоже запускает экстрактор
    памяти на авто-паузе. Регрессия: ручной вызов handle_newdream_command из
    _handle_qa_turn терял openrouter_classifier_client → экстрактор молча скипался."""
    from adapters.telegram.states import NewDreamStates

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    class _MemoryOnSettings(_NoLimitSettings):
        memory_write_enabled = True
        openrouter_context_model = "openai/gpt-4o-mini"
        memory_compact_enabled = True
        memory_compact_budget_chars = 1600

    extract_mock = AsyncMock()
    monkeypatch.setattr(
        "adapters.telegram.handlers.quick.run_memory_extraction_on_pause", extract_mock
    )

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock()
    classifier = _intent_client("new_dream")

    await handle_quick_qa_text(
        _text("а давай разберём новый сон"),
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_MemoryOnSettings(),
    )

    # Поведение как у команды: пауза + переход к сбору нового сна…
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert await fsm.get_state() == NewDreamStates.ASKING_DREAM.state
    # …И экстрактор памяти реально вызван с живым клиентом классификатора —
    # сказанное в сессии сохраняется так же, как на пути через /newdream.
    extract_mock.assert_awaited_once()
    assert extract_mock.call_args.kwargs["client"] is classifier
    assert extract_mock.call_args.kwargs["session_id"] == sess.id
    assert extract_mock.call_args.kwargs["user_id"] == user.id


@pytest.mark.asyncio
async def test_qa_new_dream_during_onboarding_completes_onboarding(
    test_session, schema, fsm
) -> None:
    """Регрессия: «новый сон» в IN_QA во время НЕзавершённого онбординга не должно
    упираться в NOT_ONBOARDED тупик — сначала завершаем онбординг, потом сбор нового сна."""
    from datetime import UTC, datetime

    from adapters.telegram.states import NewDreamStates

    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context="я в эмиграции")
    # НЕ onboarded: onboarding_completed_at остаётся None
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime(2026, 6, 1, tzinfo=UTC),
        transcription="мне снился поезд",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK,
        is_onboarding_session=True,
    )
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await test_session.commit()
    await test_session.refresh(user)
    assert user.onboarding_completed_at is None

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock()
    classifier = _intent_client("new_dream")

    await handle_quick_qa_text(
        _text("давай разберём другой сон"),
        fsm,
        session=test_session,
        bot=AsyncMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # Онбординг завершён + перешли к сбору нового сна (не NOT_ONBOARDED тупик)
    await test_session.refresh(user)
    assert user.onboarding_completed_at is not None
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    generator.chat.completions.create.assert_not_awaited()
    assert await fsm.get_state() == NewDreamStates.ASKING_DREAM.state


@pytest.mark.asyncio
async def test_finish_reply_button_pauses_session(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Reply «Завершить» → PAUSED (не COMPLETED), dream.status остаётся IN_ANALYSIS."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _text("Завершить")
    await handle_finish_quick_reply(msg, fsm, session=test_session, bot=MagicMock(spec=Bot))

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    await test_session.refresh(dream)
    # dream.status НЕ меняется при pause — остаётся как есть (RECORDED в данном тесте)
    assert dream.status == _DS.RECORDED
    assert await fsm.get_state() is None


@pytest.mark.asyncio
async def test_dream_card_text_analyze_starts_session(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Свободный текст на карточке сна (classify=analyze) → запускается разбор."""
    from adapters.telegram.handlers.quick import handle_dream_card_text
    from adapters.telegram.states import DreamCardStates

    user, dream = onboarded_with_dream
    await fsm.set_state(DreamCardStates.DECIDING_ANALYZE)
    await fsm.set_data({"dream_card_dream_id": dream.id})

    classifier = _intent_client("analyze")
    await handle_dream_card_text(
        _text("да, разбери сейчас"),
        fsm,
        session=test_session,
        settings=_NoLimitSettings(),
        openrouter_classifier_client=classifier,
        openrouter_intent_model="openai/gpt-4o-mini",
    )

    assert await fsm.get_state() == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    data = await fsm.get_data()
    assert data.get("session_id") is not None
    assert data.get("dream_id") == dream.id


@pytest.mark.asyncio
async def test_dream_card_text_save_clears_state(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Свободный текст на карточке сна (classify=save) → state очищается, разбор не стартует."""
    from adapters.telegram.handlers.quick import handle_dream_card_text
    from adapters.telegram.states import DreamCardStates

    user, dream = onboarded_with_dream
    await fsm.set_state(DreamCardStates.DECIDING_ANALYZE)
    await fsm.set_data({"dream_card_dream_id": dream.id})

    classifier = _intent_client("save")
    msg = _text("нет, потом")
    await handle_dream_card_text(
        msg,
        fsm,
        session=test_session,
        settings=_NoLimitSettings(),
        openrouter_classifier_client=classifier,
        openrouter_intent_model="openai/gpt-4o-mini",
    )

    assert await fsm.get_state() is None
    msg.answer.assert_awaited()


@pytest.mark.asyncio
async def test_quick_qa_handles_llm_failure_gracefully(
    test_session, onboarded_with_dream, fsm
) -> None:
    """OpenRouter падает → юзер видит сообщение, сессия НЕ закрывается, FSM остаётся в IN_QA."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
    classifier = AsyncMock()

    msg = _text("вопрос")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=openrouter,
        openrouter_generator_model="google/gemini-3-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    msg.answer.assert_awaited()
    assert "попробуйте" in msg.answer.await_args.args[0].lower()

    # quick.v2.1 (pending-user fix): user-message НЕ должен быть в БД, если LLM
    # упала — иначе load_quick_qa_history молча игнорирует «висячий» вопрос,
    # и он теряется из контекста следующих turn'ов.
    from data.models import SessionMessage, SessionMessageDirection

    db_msgs = (
        (
            await test_session.execute(
                select(SessionMessage).where(
                    SessionMessage.session_id == sess.id,
                    SessionMessage.direction == SessionMessageDirection.USER_TO_BOT,
                )
            )
        )
        .scalars()
        .all()
    )
    assert db_msgs == [], (
        "User-сообщение записалось в БД несмотря на сбой LLM — это создаст "
        "«висячий» вопрос без ответа"
    )


@pytest.mark.asyncio
async def test_quick_qa_voice_transcribes_and_advances(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Voice in IN_QA: Whisper transcribes, records Whisper usage, then LLM call."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    bot = AsyncMock()

    async def fake_dl(file, destination):
        destination.write(b"\x00" * 100)

    bot.download = AsyncMock(side_effect=fake_dl)
    openai = AsyncMock()
    openai.audio.transcriptions.create = AsyncMock(
        return_value=SimpleNamespace(text="а если послушать тень?", duration=5.0)
    )

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ", prompt_tokens=500, completion_tokens=100)
    )

    classifier = AsyncMock()
    msg = _voice()
    await handle_quick_qa_voice(
        msg,
        fsm,
        session=test_session,
        bot=bot,
        openai_client=openai,
        openrouter_client=openrouter,
        openrouter_generator_model="google/gemini-3-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # quick.v2.1 (E2): user-voice сохранён в session_messages как USER_TO_BOT с
    # content_type=VOICE и audio_file_id. qa_history в FSM больше нет.
    from data.models import (
        SessionMessage,
        SessionMessageContentType,
        SessionMessageDirection,
    )

    db_msgs = (
        (
            await test_session.execute(
                select(SessionMessage).order_by(SessionMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    user_msgs = [m for m in db_msgs if m.direction == SessionMessageDirection.USER_TO_BOT]
    assert len(user_msgs) == 1
    assert user_msgs[0].content == "а если послушать тень?"
    assert user_msgs[0].content_type == SessionMessageContentType.VOICE
    assert user_msgs[0].audio_file_id == "v1"
    # Bot ответ тоже в БД (последнее сообщение):
    assert db_msgs[-1].direction == SessionMessageDirection.BOT_TO_USER
    # content may be padded to meet min-length check; check prefix
    assert db_msgs[-1].content.startswith("ответ")
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state

    # Whisper was called:
    openai.audio.transcriptions.create.assert_awaited_once()
    # OpenRouter was called:
    openrouter.chat.completions.create.assert_awaited_once()

    # Whisper usage was recorded (audio_seconds, ceil(5.0)=5):
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    whisper_events = [e for e in events if e.service == LlmService.WHISPER]
    assert len(whisper_events) == 1
    assert whisper_events[0].usage_amount == 5
    assert whisper_events[0].session_id == sess.id


# ===================================================================
# quick.v2.1: Continue / New / To-Deep — UX возврата к разобранному сну
# ===================================================================


@pytest.mark.asyncio
async def test_start_analysis_shows_already_active_when_dream_has_inactive_quick(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Если на сне уже есть inactive Quick-сессия — handle_start_analysis_callback
    не создаёт новую сессию (UX «новый разбор» удалён в Task 3), а отвечает
    QUICK_ALREADY_ACTIVE. Этот путь срабатывает только при stale-callback из
    старого сообщения — карточка /dream_<id> для уже разобранного сна теперь
    показывает кнопку «Задать вопрос», а не «Разобрать сейчас»."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    old_sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await sess_repo.complete(session_id=old_sess.id)
    await test_session.commit()

    cb = _callback(f"start_analysis:{dream.id}")
    from adapters.telegram.handlers.quick import handle_start_analysis_callback

    await handle_start_analysis_callback(cb, fsm, session=test_session, settings=_NoLimitSettings())

    # Новая сессия НЕ создана
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1  # только старая completed

    # FSM не продвинулся в ASKING_PER_SESSION_CONTEXT
    assert await fsm.get_state() is None

    # Юзеру отправили QUICK_ALREADY_ACTIVE без клавиатуры
    cb.message.answer.assert_awaited_once()
    args = cb.message.answer.await_args.args
    assert args[0] == QUICK_ALREADY_ACTIVE


@pytest.mark.asyncio
async def test_continue_quick_reopens_session_and_shows_last_reply(
    test_session, onboarded_with_dream, fsm
) -> None:
    """handle_continue_quick_callback: status completed → in_progress, dream
    in_analysis, FSM в IN_QA с показом последней реплики бота."""
    from adapters.telegram.handlers.quick import handle_continue_quick_callback
    from adapters.telegram.session_log import log_bot_message, log_user_message

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    # Записываем историю: initial + 2 QA пары
    await log_bot_message(
        test_session, session_id=sess.id, content="первый разбор", phase_label="initial"
    )
    await log_user_message(test_session, session_id=sess.id, content="q1")
    await log_bot_message(test_session, session_id=sess.id, content="a1", phase_label="qa")
    await log_user_message(test_session, session_id=sess.id, content="q2")
    await log_bot_message(
        test_session, session_id=sess.id, content="последний ответ бота", phase_label="qa"
    )
    await sess_repo.complete(session_id=sess.id)
    # Также пометим dream как ANALYZED — после complete'а
    dream.status = _DS.ANALYZED
    await test_session.commit()

    cb = _callback(f"continue_quick:{dream.id}")
    await handle_continue_quick_callback(cb, fsm, session=test_session, settings=_NoLimitSettings())

    # Сессия reopened
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    assert sess.finished_at is None
    await test_session.refresh(dream)
    assert dream.status == _DS.IN_ANALYSIS

    # FSM в IN_QA + ссылки на сессию
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    data = await fsm.get_data()
    assert data["dream_id"] == dream.id
    assert data["session_id"] == sess.id

    # Бот написал intro + последнюю реплику
    assert cb.message.answer.await_count >= 2
    intro_call = cb.message.answer.await_args_list[0]
    assert "Возвращаюсь к прошлому разбору" in intro_call.args[0]
    # Task 8: reply-клавы «Завершить» НЕТ на intro — она теперь inline под разбором
    assert intro_call.kwargs.get("reply_markup") is None
    # Последний answer — содержит last_bot_reply с inline-клавой (Task 8)
    last_call = cb.message.answer.await_args_list[-1]
    assert "последний ответ бота" in last_call.args[0]
    assert last_call.kwargs.get("reply_markup") is not None


@pytest.mark.asyncio
async def test_qa_turn_triggers_rolling_summary_at_threshold(
    test_session, onboarded_with_dream, fsm
) -> None:
    """quick.v2.1 опция C: на 9-м вопросе (8 пар уже в БД) запускается Flash-
    суммаризация, summary сохраняется в session_state, и при построении
    follow-up messages summary инжектится в system."""
    from adapters.telegram.session_log import log_bot_message, log_user_message
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="контекст")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    # 1 initial + 7 QA-пар = 8 «пар» (с [initial] на первой позиции). Триггер при >= 8.
    await log_bot_message(
        test_session, session_id=sess.id, content="первый разбор", phase_label="initial"
    )
    for i in range(7):
        await log_user_message(test_session, session_id=sess.id, content=f"q{i}")
        await log_bot_message(
            test_session, session_id=sess.id, content=f"a{i}", phase_label="qa"
        )
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    # Generator (Pro) — отвечает на 8-й вопрос юзера
    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock(
        return_value=_llm_response("свежий ответ", prompt_tokens=900, completion_tokens=300)
    )

    # Classifier (Flash) — генерит summary
    classifier = AsyncMock()
    classifier.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="сжатое содержание прошлых 4 turn'ов")
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1500, completion_tokens=180),
        )
    )

    msg = _text("вопрос на пороге summary")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # Classifier-клиент вызван дважды: intent-классификатор (точка 4) + суммаризация.
    # Последний вызов — суммаризация (на модели flash).
    assert classifier.chat.completions.create.await_count == 2
    flash_call = classifier.chat.completions.create.await_args
    assert flash_call.kwargs["model"] == "google/gemini-2.5-flash"

    # Summary сохранён в session_state.data.quick_qa_summary
    state_repo = SessionStateRepository(test_session)
    saved = await state_repo.get_data(session_id=sess.id)
    assert saved is not None
    assert saved["quick_qa_summary"] == "сжатое содержание прошлых 4 turn'ов"

    # В generator messages summary попал в system (через _system_followup)
    gen_call = generator.chat.completions.create.await_args
    sys_content = gen_call.kwargs["messages"][0]["content"]
    assert "сжатое содержание прошлых 4 turn'ов" in sys_content
    # qa_history в generator-call обрезан до RECENT_KEEP=4 пар + текущий вопрос
    # messages = [system, (user, assistant) × 4, user_followup] → 1 + 8 + 1 = 10
    assert len(gen_call.kwargs["messages"]) == 10

    # В БД появилось ещё две записи (user-вопрос + bot-ответ)
    from data.models import SessionMessage

    rows = (
        (
            await test_session.execute(
                select(SessionMessage).order_by(SessionMessage.created_at.asc())
            )
        )
        .scalars()
        .all()
    )
    # До: 1 initial + 7*(user+bot) = 15. После: +2 = 17
    assert len(rows) == 17
    assert rows[-2].content == "вопрос на пороге summary"
    # content may be padded to meet min-length check; check prefix
    assert rows[-1].content.startswith("свежий ответ")


@pytest.mark.asyncio
async def test_qa_turn_skips_summary_below_threshold(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Ниже SUMMARY_THRESHOLD Flash НЕ вызывается, summary не считается."""
    from adapters.telegram.session_log import log_bot_message

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await log_bot_message(
        test_session, session_id=sess.id, content="initial", phase_label="initial"
    )
    # Только 1 пара (initial) — намного ниже порога 8.
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    generator = AsyncMock()
    generator.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ", prompt_tokens=400, completion_tokens=80)
    )
    classifier = AsyncMock()
    classifier.chat.completions.create = AsyncMock()

    msg = _text("первый вопрос")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=generator,
        openrouter_generator_model="google/gemini-2.5-pro",
        openrouter_classifier_client=classifier,
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    # Суммаризация НЕ запускалась (ниже порога): summary в session_state не появился.
    # (classifier-клиент мог быть вызван 1 раз — это intent-классификатор точки 4, не summary.)
    from data.repositories import SessionStateRepository as _SSR

    saved = await _SSR(test_session).get_data(session_id=sess.id)
    assert saved is None or "quick_qa_summary" not in saved
    assert classifier.chat.completions.create.await_count <= 1


@pytest.mark.asyncio
async def test_initial_analysis_stores_options_and_inline_kb(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Хвост из генератора → labels в session_state + inline-клава под разбором."""
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY
    from data.repositories import SessionStateRepository
    from llm.prompts.quick import QA_OPTIONS_SENTINEL

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    raw = (
        "Глубокий разбор сна про лес.\n\n**Если интересно копнуть глубже:**\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"choice","options":["Линия про страх","Линия про опору"]}'
    )
    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response(raw, prompt_tokens=800, completion_tokens=150)
    )

    msg = _text("вчера было сложное собрание")
    msg.answer = AsyncMock(return_value=SimpleNamespace(message_id=777))

    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    # labels+kind в session_state
    state_repo = SessionStateRepository(test_session)
    data = await state_repo.get_data(session_id=sess.id)
    assert data[QA_OPTIONS_KEY]["labels"] == ["Линия про страх", "Линия про опору"]
    assert data[QA_OPTIONS_KEY]["kind"] == "choice"

    # kb_html тоже сохранён — непустая строка с прозой
    kb_html = data[QA_OPTIONS_KEY].get("kb_html")
    assert isinstance(kb_html, str) and len(kb_html) > 0

    # клава прикреплена; хвост юзеру НЕ виден
    kwargs = msg.answer.await_args.kwargs
    assert "reply_markup" in kwargs
    sent_text = msg.answer.await_args.args[0]
    assert QA_OPTIONS_SENTINEL not in sent_text
    assert "Линия про страх" not in sent_text  # вариантов в прозе нет

    # message_id клавы в FSM
    assert (await fsm.get_data())["qa_options_msg_id"] == 777


@pytest.mark.asyncio
async def test_initial_analysis_no_tail_falls_back_to_no_chips(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("разбор без хвоста", prompt_tokens=800, completion_tokens=150)
    )
    msg = _text("текст")
    msg.answer = AsyncMock(return_value=SimpleNamespace(message_id=1))

    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )
    state_repo = SessionStateRepository(test_session)
    data = await state_repo.get_data(session_id=sess.id)
    assert data[QA_OPTIONS_KEY]["labels"] == []  # fallback: без чипов


# ===================================================================
# Task 7: qa:* callback handler
# ===================================================================


@pytest.mark.asyncio
async def test_qa_callback_finish_pauses_session(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import handle_qa_callback

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    cb = _callback("qa:finish")
    bot = AsyncMock(spec=Bot)
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=bot,
        openrouter_client=AsyncMock(), openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert await fsm.get_state() is None  # handle_finish чистит FSM


@pytest.mark.asyncio
async def test_qa_callback_custom_prompts_and_stays_in_qa(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import handle_qa_callback
    from adapters.telegram.texts import QA_CUSTOM_QUESTION_PROMPT

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    cb = _callback("qa:custom")
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=AsyncMock(), openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )
    cb.message.answer.assert_awaited()
    call = cb.message.answer.await_args
    sent = call.args[0] if call.args else call.kwargs.get("text", "")
    assert QA_CUSTOM_QUESTION_PROMPT in sent
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_qa_callback_chip_runs_followup_with_label(
    test_session, onboarded_with_dream, fsm
) -> None:
    """qa:0 → followup = labels[0] → генератор follow-up (история пополняется)."""
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from adapters.telegram.session_log import load_quick_qa_history
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {"labels": ["Линия про страх", "Линия про опору"], "kind": "choice"}},
    )
    # initial-сообщение, чтобы followup имел контекст
    from adapters.telegram.session_log import log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    await log_bot_message(test_session, session_id=sess.id, content="первый разбор", phase_label=QUICK_PHASE_INITIAL)
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("follow-up ответ", prompt_tokens=300, completion_tokens=80)
    )
    cb = _callback("qa:0")
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )
    # user-followup записан как лейбл дословно
    _, pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert pairs and pairs[-1][0] == "Линия про страх"


@pytest.mark.asyncio
async def test_qa_callback_chip_finish_intent_pauses_session(
    test_session, onboarded_with_dream, fsm, monkeypatch
) -> None:
    """Ранее (Task 7): чип-тап qa:0 с лейблом, который классификатор считал 'finish',
    должен был завершать сессию. Это было ОШИБКОЙ: чип = детерминированный explore,
    классификатор не должен влиять на его исход.

    После фикса chip-bypass (classify_intent=False): тап чипа ВСЕГДА идёт в генератор
    (continue), независимо от того, что вернул бы классификатор. Сессия остаётся IN_QA,
    классификатор не вызывается.

    Тест переписан: проверяем исправленное поведение — bypass работает.
    """
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from adapters.telegram.session_log import load_quick_qa_history, log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository
    from llm.intent import IntentResult

    # Классификатор замокан на "finish" — до фикса это паузило сессию.
    # После фикса (bypass) этот мок вообще не должен вызываться.
    mock_classifier = AsyncMock(
        return_value=IntentResult(outcome="finish", input_tokens=0, output_tokens=0)
    )
    monkeypatch.setattr(
        "adapters.telegram.handlers.quick.classify_qa_intent",
        mock_classifier,
    )

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    # Записываем чип-лейблы в session_state
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {"labels": ["Завершить разбор"], "kind": "choice"}},
    )
    await log_bot_message(
        test_session, session_id=sess.id, content="первый разбор", phase_label=QUICK_PHASE_INITIAL
    )
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("продолжение разбора", prompt_tokens=300, completion_tokens=80)
    )
    cb = _callback("qa:0")
    bot = AsyncMock(spec=Bot)
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=bot,
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    # Классификатор НЕ должен быть вызван (bypass)
    mock_classifier.assert_not_awaited()

    # Генератор ВЫЗВАН — чип = explore, идём в follow-up
    openrouter.chat.completions.create.assert_awaited()

    # Сессия остаётся IN_QA (не paused!)
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA, (
        "Сессия должна оставаться IN_QA после чип-тапа (bypass классификатора)"
    )
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS, (
        "Сессия не должна уходить в PAUSED при тапе чипа"
    )

    # Лейбл чипа записан как реплика юзера
    _, pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert pairs and pairs[-1][0] == "Завершить разбор"


# ===================================================================
# Task 8: возврат к сну восстанавливает inline-клаву из session_state
# ===================================================================


@pytest.mark.asyncio
async def test_resume_reattaches_inline_kb_from_session_state(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import (
        QA_OPTIONS_KEY,
        handle_continue_quick_callback,
    )
    from adapters.telegram.session_log import log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await log_bot_message(test_session, session_id=sess.id, content="разбор", phase_label=QUICK_PHASE_INITIAL)
    await sess_repo.pause(session_id=sess.id)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {"labels": ["Линия А"], "kind": "choice"}},
    )
    await test_session.commit()

    cb = _callback(f"continue_quick:{dream.id}")
    cb.message.answer = AsyncMock(return_value=SimpleNamespace(message_id=999))
    await handle_continue_quick_callback(
        cb, fsm, session=test_session, settings=_NoLimitSettings()
    )
    # последняя отправка несёт inline-клаву с чипом
    kwargs = cb.message.answer.await_args.kwargs
    kb = kwargs.get("reply_markup")
    assert kb is not None
    flat = [b for row in kb.inline_keyboard for b in row]
    assert flat[0].text == "Линия А" and flat[0].callback_data == "qa:0"
    # FSM должна содержать qa_options_msg_id отправленной клавы
    assert (await fsm.get_data()).get("qa_options_msg_id") == 999


# ===================================================================
# Choice-trail: chip tap edits the message to show «Ваш выбор: …»
# ===================================================================


@pytest.mark.asyncio
async def test_qa_callback_chip_edits_message_with_choice(
    test_session, onboarded_with_dream, fsm
) -> None:
    """qa:0 тап → bot.edit_message_text вызван с message_id=777 и 'Ваш выбор:' в тексте."""
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from adapters.telegram.session_log import log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {
            "labels": ["Линия про страх", "Линия про опору"],
            "kind": "choice",
            "kb_html": "…Куда заглянем дальше?",
        }},
    )
    await log_bot_message(test_session, session_id=sess.id, content="первый разбор", phase_label=QUICK_PHASE_INITIAL)
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id, "qa_options_msg_id": 777})

    bot = AsyncMock(spec=Bot)
    bot.edit_message_text = AsyncMock()

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("follow-up ответ", prompt_tokens=300, completion_tokens=80)
    )
    cb = _callback("qa:0", message_id=777)
    cb.message.chat = MagicMock()
    cb.message.chat.id = 42

    await handle_qa_callback(
        cb, fsm, session=test_session, bot=bot,
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    # edit_message_text вызван ровно один раз
    bot.edit_message_text.assert_awaited_once()
    call_kwargs = bot.edit_message_text.await_args.kwargs
    assert call_kwargs["message_id"] == 777
    assert "Куда заглянем дальше?" in call_kwargs["text"]
    assert "Ваш выбор:" in call_kwargs["text"]
    assert "Линия про страх" in call_kwargs["text"]


@pytest.mark.asyncio
async def test_qa_callback_custom_edits_message_with_svy_vopros(
    test_session, onboarded_with_dream, fsm
) -> None:
    """qa:custom → edit_message_text содержит 'Ваш выбор:' и 'свой вопрос'."""
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {
            "labels": ["Линия А"],
            "kind": "choice",
            "kb_html": "…вопрос?",
        }},
    )
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id, "qa_options_msg_id": 888})

    bot = AsyncMock(spec=Bot)
    bot.edit_message_text = AsyncMock()

    cb = _callback("qa:custom", message_id=888)
    cb.message.chat = MagicMock()
    cb.message.chat.id = 42

    await handle_qa_callback(
        cb, fsm, session=test_session, bot=bot,
        openrouter_client=AsyncMock(), openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    bot.edit_message_text.assert_awaited_once()
    text = bot.edit_message_text.await_args.kwargs["text"]
    assert "Ваш выбор:" in text
    assert "свой вопрос" in text


@pytest.mark.asyncio
async def test_qa_callback_chip_missing_kb_html_falls_back_gracefully(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Если kb_html нет в session_state — edit не вызывается, но _handle_qa_turn запускается."""
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from adapters.telegram.session_log import load_quick_qa_history, log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    state_repo = SessionStateRepository(test_session)
    # NO kb_html — only labels
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {"labels": ["Вопрос А", "Вопрос Б"], "kind": "choice"}},
    )
    await log_bot_message(test_session, session_id=sess.id, content="начальный разбор", phase_label=QUICK_PHASE_INITIAL)
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id, "qa_options_msg_id": 555})

    bot = AsyncMock(spec=Bot)
    bot.edit_message_text = AsyncMock()
    bot.edit_message_reply_markup = AsyncMock()

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ на вопрос", prompt_tokens=300, completion_tokens=80)
    )
    cb = _callback("qa:0", message_id=555)
    cb.message.chat = MagicMock()
    cb.message.chat.id = 42

    await handle_qa_callback(
        cb, fsm, session=test_session, bot=bot,
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    # edit_message_text НЕ вызван (нет kb_html), но qa_turn выполнился
    bot.edit_message_text.assert_not_awaited()
    # Проверяем что follow-up действительно прошёл — user-message записан в БД
    _, pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert pairs and pairs[-1][0] == "Вопрос А"


# ===================================================================
# Task 4: карта беседы — persist + feed-back
# ===================================================================


@pytest.mark.asyncio
async def test_initial_delivery_persists_map_and_clarify_kb(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import QA_MAP_KEY
    from data.repositories import SessionStateRepository
    from llm.prompts.quick import QA_OPTIONS_SENTINEL

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    raw = (
        "Разбор.\n\n**Тигр был агрессивным или добрым?**\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"clarify","options":[],"map":{"focus":"тигр","type":"unknown","stage":"вглубь","covered":[]}}'
    )
    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response(raw, prompt_tokens=800, completion_tokens=150)
    )
    msg = _text("контекст")
    msg.answer = AsyncMock(return_value=SimpleNamespace(message_id=1))

    await handle_per_session_context_text(
        msg, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    state_repo = SessionStateRepository(test_session)
    data = await state_repo.get_data(session_id=sess.id)
    assert data[QA_MAP_KEY]["focus"] == "тигр"
    assert data[QA_MAP_KEY]["stage"] == "вглубь"
    # clarify-клава: Пропустить + Завершить (без чипов)
    kb = msg.answer.await_args.kwargs["reply_markup"]
    cbs = [b.callback_data for row in kb.inline_keyboard for b in row]
    assert cbs == ["qa:skip", "qa:finish"]


@pytest.mark.asyncio
async def test_followup_reads_map_into_generator_messages(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Карта из session_state попадает в system-сообщение генератора."""
    from adapters.telegram.handlers.quick import QA_MAP_KEY
    from adapters.telegram.session_log import log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    await log_bot_message(test_session, session_id=sess.id, content="первый разбор", phase_label=QUICK_PHASE_INITIAL)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_MAP_KEY: {"focus": "тигр", "type": "manager", "stage": "вглубь", "covered": ["лес"]}},
    )
    await test_session.commit()
    await test_session.refresh(user)
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    captured = {}
    async def _capture(**kwargs):
        captured["messages"] = kwargs["messages"]
        return _llm_response("ответ", prompt_tokens=200, completion_tokens=50)
    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(side_effect=_capture)

    msg = _text("да")
    msg.answer = AsyncMock(return_value=SimpleNamespace(message_id=2))
    await handle_quick_qa_text(
        msg, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )
    system_msg = captured["messages"][0]["content"]
    assert "Карта беседы" in system_msg and "тигр" in system_msg


# ===================================================================
# Task 5: qa:skip callback
# ===================================================================


@pytest.mark.asyncio
async def test_qa_callback_skip_runs_turn_with_skip_signal(
    test_session, onboarded_with_dream, fsm
) -> None:
    from adapters.telegram.handlers.quick import QA_SKIP_FOLLOWUP, handle_qa_callback
    from adapters.telegram.session_log import load_quick_qa_history, log_bot_message
    from data.constants import QUICK_PHASE_INITIAL

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    await log_bot_message(test_session, session_id=sess.id, content="разбор", phase_label=QUICK_PHASE_INITIAL)
    await test_session.commit()
    await test_session.refresh(user)
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("интерпретация без давления", prompt_tokens=200, completion_tokens=50)
    )
    cb = _callback("qa:skip")
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )
    # followup-сигнал пропуска записан как реплика юзера
    _, pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert pairs and pairs[-1][0] == QA_SKIP_FOLLOWUP


# ===================================================================
# Chip-bypass: тап чипа/Пропустить не идёт через intent-классификатор
# ===================================================================


@pytest.mark.asyncio
async def test_qa_chip_tap_bypasses_intent_classifier(
    test_session, onboarded_with_dream, fsm, monkeypatch
) -> None:
    """Тап чипа qa:N — classify_qa_intent НЕ вызывается.

    Чип = детерминированный «explore» — классификатор нужен только для свободного
    текста/голоса. Если классификатор всё равно вызвать и он вернёт new_dream,
    handler вызовет handle_newdream_command с message.from_user = бот → «не onboarded».
    Фикс: classify_intent=False на chip-tap → классификатор пропущен,
    генератор выполнен, сессия остаётся IN_QA.
    """
    from adapters.telegram.handlers.quick import QA_OPTIONS_KEY, handle_qa_callback
    from adapters.telegram.session_log import load_quick_qa_history, log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from data.repositories import SessionStateRepository
    from llm.intent import IntentResult

    # Классификатор замокан на new_dream — если его вызовут, тест это обнаружит.
    mock_classifier = AsyncMock(
        return_value=IntentResult(outcome="new_dream", input_tokens=0, output_tokens=0)
    )
    monkeypatch.setattr("adapters.telegram.handlers.quick.classify_qa_intent", mock_classifier)

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    state_repo = SessionStateRepository(test_session)
    await state_repo.upsert(
        session_id=sess.id,
        data={QA_OPTIONS_KEY: {"labels": ["Исчезновение маяка", "Другая линия"], "kind": "choice"}},
    )
    await log_bot_message(
        test_session, session_id=sess.id, content="первый разбор", phase_label=QUICK_PHASE_INITIAL
    )
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ про маяк", prompt_tokens=300, completion_tokens=80)
    )
    cb = _callback("qa:0")
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    # 1. Классификатор НЕ был вызван (bypass)
    mock_classifier.assert_not_awaited()

    # 2. Генератор ВЫЗВАНhttps://
    openrouter.chat.completions.create.assert_awaited()

    # 3. Сессия остаётся IN_QA (не paused, не new_dream)
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA

    # 4. Лейбл чипа записан как реплика юзера
    _, pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert pairs and pairs[-1][0] == "Исчезновение маяка"


@pytest.mark.asyncio
async def test_qa_skip_bypasses_intent_classifier(
    test_session, onboarded_with_dream, fsm, monkeypatch
) -> None:
    """Тап qa:skip — classify_qa_intent НЕ вызывается.

    Пропустить = детерминированный continue, классификатор не нужен.
    """
    from adapters.telegram.handlers.quick import handle_qa_callback
    from adapters.telegram.session_log import log_bot_message
    from data.constants import QUICK_PHASE_INITIAL
    from llm.intent import IntentResult

    mock_classifier = AsyncMock(
        return_value=IntentResult(outcome="finish", input_tokens=0, output_tokens=0)
    )
    monkeypatch.setattr("adapters.telegram.handlers.quick.classify_qa_intent", mock_classifier)

    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.UNCLEAR)
    await log_bot_message(
        test_session, session_id=sess.id, content="разбор", phase_label=QUICK_PHASE_INITIAL
    )
    await test_session.commit()
    await test_session.refresh(user)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(
        return_value=_llm_response("ответ без вопроса", prompt_tokens=200, completion_tokens=50)
    )
    cb = _callback("qa:skip")
    await handle_qa_callback(
        cb, fsm, session=test_session, bot=AsyncMock(spec=Bot),
        openrouter_client=openrouter, openrouter_generator_model="g",
        openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="c",
        openrouter_intent_model="i", settings=_NoLimitSettings(),
    )

    # 1. Классификатор НЕ был вызван
    mock_classifier.assert_not_awaited()

    # 2. Генератор ВЫЗВАН
    openrouter.chat.completions.create.assert_awaited()

    # 3. Сессия остаётся IN_QA
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA


def test_is_real_finish_guards_long_answers():
    """Длинная реплика с outcome=finish НЕ считается завершением (ложное срабатывание)."""
    from adapters.telegram.handlers.quick import _QA_FINISH_MAX_CHARS, _is_real_finish

    # Короткий явный финиш — засчитывается.
    assert _is_real_finish("finish", "хватит, спасибо") is True
    assert _is_real_finish("finish", "  достаточно на сегодня  ") is True

    # Длинный содержательный ответ, ошибочно классифицированный как finish, — НЕ финиш.
    long_answer = (
        "Я проходил его тренинг физикл, пришёл туда по рекомендации, и при весе 85 кг "
        "и росте 170 см смог благодаря его методике подсчёта калорий и физической "
        "активности похудеть на 20 кг, я уже 3,5 года нахожусь в его проекте, очень ему "
        "благодарен за такой крутой продукт, он не инфоциган и реально помогает людям "
        "похудеть. Благодаря проекту Олега я стал больше верить в себя и что благодаря "
        "ежедневным действиям можно добиться хороших результатов. Я ему очень "
        "благодарен, что помог мне стать новой версией меня."
    )
    assert len(long_answer.strip()) > _QA_FINISH_MAX_CHARS
    assert _is_real_finish("finish", long_answer) is False

    # continue / new_dream — никогда не финиш, независимо от длины.
    assert _is_real_finish("continue", "хватит") is False
    assert _is_real_finish("new_dream", "давай новый сон") is False


def _blocked_response():
    """OpenRouter-ответ «фильтр зарубил запрос» (Gemini отдаёт это как 400)."""
    _msg = "Gemini blocked the request: PROHIBITED_CONTENT"
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        error=SimpleNamespace(code=400, message=_msg),
        model_dump=lambda: {
            "provider": "test-provider",
            "error": {"code": 400, "message": _msg},
        },
    )


@pytest.mark.asyncio
async def test_initial_analysis_content_filter_exits_dream_mode(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Все модели зарубили сон → человеческий текст, сессия на паузе, FSM очищен.

    Регрессия инцидента 2026-07-20: юзер читал «Что-то сломалось в коде бота»
    и оставался в состоянии без выхода.
    """
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(return_value=_blocked_response())

    msg = _text("трудный день")
    await handle_per_session_context_text(
        msg, fsm, session=test_session,
        openrouter_client=openrouter, openrouter_classifier_client=AsyncMock(),
        settings=_NoLimitSettings(),
    )

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert await fsm.get_state() is None
    sent = " ".join(str(c.args[0]) for c in msg.answer.await_args_list)
    assert "безопасный режим" in sent
    assert "/newdream" in sent
    assert "сломалось в коде" not in sent


@pytest.mark.asyncio
async def test_qa_turn_content_filter_keeps_user_in_qa(
    test_session, onboarded_with_dream, fsm
) -> None:
    """Фильтр на круге Q&A → текст, но юзер остаётся в IN_QA и сессия жива."""
    user, dream = onboarded_with_dream
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await sess_repo.set_per_session_context(session_id=sess.id, context="c")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await test_session.commit()
    await test_session.refresh(sess)

    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    openrouter = AsyncMock()
    openrouter.chat.completions.create = AsyncMock(return_value=_blocked_response())

    msg = _text("мой ответ")
    await handle_quick_qa_text(
        msg,
        fsm,
        session=test_session,
        bot=MagicMock(spec=Bot),
        openrouter_client=openrouter,
        openrouter_generator_model="google/gemini-3-pro",
        openrouter_classifier_client=AsyncMock(),
        openrouter_classifier_model="google/gemini-2.5-flash",
        openrouter_intent_model="openai/gpt-4o-mini",
        settings=_NoLimitSettings(),
    )

    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    sent = " ".join(str(c.args[0]) for c in msg.answer.await_args_list)
    assert "безопасный режим" in sent
    assert "/newdream" not in sent
