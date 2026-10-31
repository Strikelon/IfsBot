"""Integration tests for idle catch-all handler (Phase 4).

User without active session sending non-command text or voice → IDLE_HINT.
"""
from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message, Voice
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.idle import handle_idle_text, handle_idle_voice
from adapters.telegram.texts import IDLE_HINT
from data.models import Base, Dream, ExperienceLevel, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def onboarded_user(test_session, schema):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(
        user_id=user.id,
        profile_life_context="x",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()
    return user


@pytest.fixture
def fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _make_message(text: str, user_id: int = 42) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=user_id, type="private")
    msg.text = text
    msg.voice = None
    msg.answer = AsyncMock()
    return msg


def _make_voice_message(user_id: int = 42) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=user_id, type="private")
    msg.text = None
    msg.voice = AsyncMock(spec=Voice)
    msg.voice.file_id = "fake_voice_file_id"
    msg.answer = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_idle_text_triggers_hint(test_session, onboarded_user, fsm) -> None:
    """User with no active session sends text without / → IDLE_HINT."""
    msg = _make_message(text="привет")

    await handle_idle_text(msg, fsm, session=test_session)

    msg.answer.assert_called_once()
    sent_text = msg.answer.call_args.args[0]
    assert sent_text == IDLE_HINT

    # Critical: free text must NOT create a new Dream
    dreams = (await test_session.execute(select(Dream))).scalars().all()
    assert dreams == [], "free text должен НЕ создавать новый Dream"


@pytest.mark.asyncio
async def test_idle_voice_triggers_hint(test_session, onboarded_user, fsm) -> None:
    """Voice without active session → IDLE_HINT. NOT transcribed."""
    voice_msg = _make_voice_message()

    await handle_idle_voice(voice_msg, fsm, session=test_session)

    voice_msg.answer.assert_called_once()
    sent_text = voice_msg.answer.call_args.args[0]
    assert sent_text == IDLE_HINT
    # Critical: no Whisper/STT call — handler takes no openai_client param


@pytest.mark.asyncio
async def test_idle_handler_with_active_session_still_sends_hint(
    test_session, onboarded_user, fsm, schema
) -> None:
    """If user has IN_PROGRESS session — handler still sends IDLE_HINT (safe default)."""
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="тёмный лес",
    )
    sess_repo = SessionRepository(test_session)
    await sess_repo.create(
        dream_id=dream.id,
        user_id=onboarded_user.id,
        type_=SessionType.QUICK,
        is_onboarding_session=False,
    )
    await test_session.commit()

    msg = _make_message(text="привет")

    await handle_idle_text(msg, fsm, session=test_session)

    # Handler still responds with IDLE_HINT as a safe default
    msg.answer.assert_called_once()
    sent_text = msg.answer.call_args.args[0]
    assert sent_text == IDLE_HINT


@pytest.mark.asyncio
async def test_idle_text_with_only_paused_session_triggers_hint(
    test_session, onboarded_user, fsm, schema
) -> None:
    """User with only PAUSED sessions (no IN_PROGRESS) → treated as idle, gets IDLE_HINT."""
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="тёмный лес",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id,
        user_id=onboarded_user.id,
        type_=SessionType.QUICK,
        is_onboarding_session=False,
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    msg = _make_message(text="привет")
    await handle_idle_text(msg, fsm, session=test_session)

    msg.answer.assert_called_once()
    assert msg.answer.call_args.args[0] == IDLE_HINT
