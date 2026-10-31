from datetime import UTC, datetime

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers._common import auto_pause_active_quick
from adapters.telegram.states import QuickAnalysisStates
from data.models import (
    Base,
    ExperienceLevel,
    SessionStatus,
    SessionType,
)
from data.repositories import DreamRepository, SessionRepository, UserRepository


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def setup_user_with_session(test_session, schema):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await user_repo.update_profile(
        user_id=user.id,
        profile_life_context="x",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
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
    await test_session.commit()
    return user, dream, sess


def _make_state(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


@pytest.mark.asyncio
async def test_auto_pause_pauses_in_progress(test_session, setup_user_with_session):
    user, dream, sess = setup_user_with_session
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    paused = await auto_pause_active_quick(
        session=test_session, state=state, user_id=user.id
    )
    await test_session.commit()

    assert paused is not None and paused.session_id == sess.id
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert (await state.get_state()) is None  # FSM cleared


@pytest.mark.asyncio
async def test_auto_pause_skips_when_excluded_dream_matches(
    test_session, setup_user_with_session
):
    """Если excluded_dream_id == активная dream — не паузим."""
    user, dream, sess = setup_user_with_session
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    paused = await auto_pause_active_quick(
        session=test_session,
        state=state,
        user_id=user.id,
        excluded_dream_id=dream.id,
    )
    await test_session.commit()

    assert paused is None
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.IN_PROGRESS
    assert (await state.get_state()) == "QuickAnalysisStates:IN_QA"


@pytest.mark.asyncio
async def test_auto_pause_pauses_when_excluded_is_other_dream(
    test_session, setup_user_with_session
):
    """excluded_dream_id != активная dream — паузим."""
    user, dream, sess = setup_user_with_session
    state = _make_state()
    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream.id, "session_id": sess.id})

    paused = await auto_pause_active_quick(
        session=test_session,
        state=state,
        user_id=user.id,
        excluded_dream_id=999,
    )
    await test_session.commit()

    assert paused is not None and paused.session_id == sess.id
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED


@pytest.mark.asyncio
async def test_auto_pause_noop_when_no_active(test_session, schema):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()
    state = _make_state()

    paused = await auto_pause_active_quick(
        session=test_session, state=state, user_id=user.id
    )

    assert paused is None
    assert (await state.get_state()) is None
