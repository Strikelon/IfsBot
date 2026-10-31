"""Тесты для adapters/telegram/session_log.py — persistence Q&A в session_messages."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
import pytest_asyncio
from sqlalchemy import select

from adapters.telegram.session_log import (
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_QA,
    load_dream_qa_for_display,
    load_quick_qa_history,
    log_bot_message,
    log_user_message,
)
from data.models import (
    ExperienceLevel,
    LlmRole,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionType,
)
from data.repositories import (
    DreamRepository,
    SessionMessageRepository,
    SessionRepository,
    UserRepository,
)


@pytest_asyncio.fixture
async def quick_session_id(test_session) -> int:
    """Сетап: онбордженый юзер + сон + Quick-сессия."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="меняю работу",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await user_repo.mark_onboarded(user_id=user.id)
    dream = await DreamRepository(test_session).create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="мне снился тёмный лес",
    )
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await test_session.commit()
    return sess.id


@pytest.mark.asyncio
async def test_log_bot_message_writes_correct_row(test_session, quick_session_id) -> None:
    await log_bot_message(
        test_session,
        session_id=quick_session_id,
        content="развёрнутый IFS-разбор сна",
        phase_label=QUICK_PHASE_INITIAL,
    )
    await test_session.commit()

    rows = (await test_session.execute(select(SessionMessage))).scalars().all()
    assert len(rows) == 1
    row = rows[0]
    assert row.session_id == quick_session_id
    assert row.direction == SessionMessageDirection.BOT_TO_USER
    assert row.content_type == SessionMessageContentType.TEXT
    assert row.content == "развёрнутый IFS-разбор сна"
    assert row.llm_phase_label == QUICK_PHASE_INITIAL
    assert row.llm_role_used == LlmRole.GENERATOR


@pytest.mark.asyncio
async def test_log_user_message_text_default(test_session, quick_session_id) -> None:
    await log_user_message(
        test_session, session_id=quick_session_id, content="а если послушать тень?"
    )
    await test_session.commit()
    row = (await test_session.execute(select(SessionMessage))).scalars().one()
    assert row.direction == SessionMessageDirection.USER_TO_BOT
    assert row.content_type == SessionMessageContentType.TEXT
    assert row.content == "а если послушать тень?"
    assert row.audio_file_id is None
    assert row.llm_phase_label is None
    assert row.llm_role_used is None


@pytest.mark.asyncio
async def test_log_user_message_voice_with_file_id(test_session, quick_session_id) -> None:
    await log_user_message(
        test_session,
        session_id=quick_session_id,
        content="транскрипция голоса",
        content_type=SessionMessageContentType.VOICE,
        audio_file_id="voice-abc-123",
    )
    await test_session.commit()
    row = (await test_session.execute(select(SessionMessage))).scalars().one()
    assert row.content_type == SessionMessageContentType.VOICE
    assert row.audio_file_id == "voice-abc-123"


@pytest.mark.asyncio
async def test_load_quick_qa_history_empty(test_session, quick_session_id) -> None:
    initial, pairs = await load_quick_qa_history(test_session, session_id=quick_session_id)
    assert initial is None
    assert pairs == []


@pytest.mark.asyncio
async def test_load_quick_qa_history_initial_only(test_session, quick_session_id) -> None:
    await log_bot_message(
        test_session,
        session_id=quick_session_id,
        content="первый разбор",
        phase_label=QUICK_PHASE_INITIAL,
    )
    await test_session.commit()

    initial, pairs = await load_quick_qa_history(test_session, session_id=quick_session_id)
    assert initial == "первый разбор"
    assert pairs == []


@pytest.mark.asyncio
async def test_load_quick_qa_history_initial_plus_pairs(test_session, quick_session_id) -> None:
    await log_bot_message(
        test_session,
        session_id=quick_session_id,
        content="первый разбор",
        phase_label=QUICK_PHASE_INITIAL,
    )
    for i in range(3):
        await log_user_message(test_session, session_id=quick_session_id, content=f"вопрос {i}")
        await log_bot_message(
            test_session,
            session_id=quick_session_id,
            content=f"ответ {i}",
            phase_label=QUICK_PHASE_QA,
        )
    await test_session.commit()

    initial, pairs = await load_quick_qa_history(test_session, session_id=quick_session_id)
    assert initial == "первый разбор"
    assert pairs == [
        ("вопрос 0", "ответ 0"),
        ("вопрос 1", "ответ 1"),
        ("вопрос 2", "ответ 2"),
    ]


@pytest.mark.asyncio
async def test_load_quick_qa_history_drops_pending_user_without_answer(
    test_session, quick_session_id
) -> None:
    """Если у последнего user-сообщения ещё нет bot-ответа (LLM упала / в процессе)
    — этот «висячий» вопрос НЕ попадает в qa_pairs. По возвращению модель ещё раз
    его получит как `user_followup`, и пара запишется только после ответа."""
    await log_bot_message(
        test_session,
        session_id=quick_session_id,
        content="initial",
        phase_label=QUICK_PHASE_INITIAL,
    )
    await log_user_message(test_session, session_id=quick_session_id, content="q1")
    await log_bot_message(
        test_session, session_id=quick_session_id, content="a1", phase_label=QUICK_PHASE_QA
    )
    # «Висячий» вопрос — без последующего bot-сообщения
    await log_user_message(test_session, session_id=quick_session_id, content="pending question")
    await test_session.commit()

    initial, pairs = await load_quick_qa_history(test_session, session_id=quick_session_id)
    assert initial == "initial"
    assert pairs == [("q1", "a1")]


@pytest.mark.asyncio
async def test_load_quick_qa_history_preserves_chronological_order(
    test_session, quick_session_id
) -> None:
    """Записи возвращаются в порядке created_at.asc(), без перемешивания."""
    await log_bot_message(
        test_session,
        session_id=quick_session_id,
        content="i",
        phase_label=QUICK_PHASE_INITIAL,
    )
    await log_user_message(test_session, session_id=quick_session_id, content="первый")
    await log_bot_message(
        test_session, session_id=quick_session_id, content="A1", phase_label=QUICK_PHASE_QA
    )
    await log_user_message(test_session, session_id=quick_session_id, content="второй")
    await log_bot_message(
        test_session, session_id=quick_session_id, content="A2", phase_label=QUICK_PHASE_QA
    )
    await test_session.commit()

    _, pairs = await load_quick_qa_history(test_session, session_id=quick_session_id)
    assert [q for q, _ in pairs] == ["первый", "второй"]
    assert [a for _, a in pairs] == ["A1", "A2"]


@pytest.mark.asyncio
async def test_load_dream_qa_for_display_no_session_returns_none(
    test_session, schema
) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон",
    )
    await test_session.commit()

    result = await load_dream_qa_for_display(test_session, dream_id=dream.id)
    assert result is None


@pytest.mark.asyncio
async def test_load_dream_qa_for_display_returns_initial_only(
    test_session, schema
) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
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
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="мой разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    result = await load_dream_qa_for_display(test_session, dream_id=dream.id)
    assert result is not None
    initial, pairs = result
    assert initial == "мой разбор"
    assert pairs == []


@pytest.mark.asyncio
async def test_load_dream_qa_for_display_returns_initial_and_pairs(
    test_session, schema
) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
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
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=SessionMessageContentType.TEXT,
        content="а что значит этот образ?",
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="это Менеджер",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label="qa",
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    result = await load_dream_qa_for_display(test_session, dream_id=dream.id)
    assert result is not None
    initial, pairs = result
    assert initial == "разбор"
    assert pairs == [("а что значит этот образ?", "это Менеджер")]


@pytest.mark.asyncio
async def test_load_dream_qa_for_display_uses_latest_session(
    test_session, schema
) -> None:
    """Если у сна несколько inactive сессий — возвращаем самую свежую по last_active_at."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон",
    )
    sess_repo = SessionRepository(test_session)
    old = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=old.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="старый разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=old.id)
    await test_session.commit()

    new_ = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await msg_repo.append(
        session_id=new_.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="новый разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=new_.id)
    await test_session.commit()

    result = await load_dream_qa_for_display(test_session, dream_id=dream.id)
    assert result is not None
    initial, _ = result
    assert initial == "новый разбор"
