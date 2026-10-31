from datetime import UTC, datetime
from decimal import Decimal

import pytest

from data.models import (
    Dream,
    DreamStatus,
    ExperienceLevel,
    LlmService,
    LlmUsageEvent,
    RequestBranch,
    Session,
    SessionStatus,
    SessionType,
    UsageMetric,
    User,
)
from data.repositories import DreamRepository, UserRepository


@pytest.mark.asyncio
async def test_user_can_be_created(test_session, schema) -> None:
    user = User(telegram_user_id=42, display_name="Катя", locale="ru")
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    assert user.id is not None
    assert user.telegram_user_id == 42
    assert user.locale == "ru"
    assert isinstance(user.created_at, datetime)


@pytest.mark.asyncio
async def test_dream_links_to_user(test_session, schema) -> None:
    user = User(telegram_user_id=42)
    test_session.add(user)
    await test_session.flush()

    dream = Dream(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="мне снилась тёмная фигура в лесу",
        title="Сон от 10 мая",
        status=DreamStatus.RECORDED,
    )
    test_session.add(dream)
    await test_session.commit()
    await test_session.refresh(dream)

    assert dream.id is not None
    assert dream.user_id == user.id
    assert dream.status == DreamStatus.RECORDED


@pytest.mark.asyncio
async def test_session_links_to_dream_and_user(test_session, schema) -> None:
    user = User(telegram_user_id=42)
    test_session.add(user)
    await test_session.flush()
    dream = Dream(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="...",
        status=DreamStatus.RECORDED,
    )
    test_session.add(dream)
    await test_session.flush()

    sess = Session(
        dream_id=dream.id,
        user_id=user.id,
        type=SessionType.QUICK,
        status=SessionStatus.IN_PROGRESS,
    )
    test_session.add(sess)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.id is not None
    assert sess.dream_id == dream.id
    assert sess.user_id == user.id
    assert sess.type == SessionType.QUICK


@pytest.mark.asyncio
async def test_enum_columns_store_lowercase_values(test_session, schema) -> None:
    """Регрессия: enum-колонки должны сохранять value (lowercase), а не name.

    Если этот тест падает, миграция Phase 6 в Postgres сломает INSERT.
    """
    from sqlalchemy import text

    user = User(telegram_user_id=99)
    test_session.add(user)
    await test_session.flush()
    dream = Dream(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="t",
        status=DreamStatus.RECORDED,
    )
    test_session.add(dream)
    await test_session.flush()
    sess = Session(
        dream_id=dream.id,
        user_id=user.id,
        type=SessionType.QUICK,
        status=SessionStatus.IN_PROGRESS,
    )
    test_session.add(sess)
    await test_session.commit()

    stored_dream_status = (
        await test_session.execute(
            text("SELECT status FROM dreams WHERE id = :id"), {"id": dream.id}
        )
    ).scalar()
    stored_session_type = (
        await test_session.execute(
            text("SELECT type FROM sessions WHERE id = :id"), {"id": sess.id}
        )
    ).scalar()
    stored_session_status = (
        await test_session.execute(
            text("SELECT status FROM sessions WHERE id = :id"), {"id": sess.id}
        )
    ).scalar()

    assert stored_dream_status == "recorded"
    assert stored_session_type == "quick"
    assert stored_session_status == "in_progress"


@pytest.mark.asyncio
async def test_user_can_have_profile_fields(test_session, schema) -> None:
    user = User(
        telegram_user_id=42,
        display_name="Катя",
        locale="ru",
        profile_life_context="переезжаю в новый город",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    assert user.profile_life_context == "переезжаю в новый город"
    assert user.profile_experience_level == ExperienceLevel.NEWBIE


@pytest.mark.asyncio
async def test_user_profile_fields_default_none(test_session, schema) -> None:
    """Phase 0 users (без онбординга) должны корректно создаваться без profile_*"""
    user = User(telegram_user_id=42)
    test_session.add(user)
    await test_session.commit()
    await test_session.refresh(user)

    assert user.profile_life_context is None
    assert user.profile_experience_level is None


@pytest.mark.asyncio
async def test_dream_stores_raw_audio_file_id(test_session, schema) -> None:
    user = User(telegram_user_id=42)
    test_session.add(user)
    await test_session.flush()

    dream = Dream(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про лес",
        raw_audio_file_id="CQACAgIAAxkBAAEC...",
        status=DreamStatus.RECORDED,
    )
    test_session.add(dream)
    await test_session.commit()
    await test_session.refresh(dream)

    assert dream.raw_audio_file_id == "CQACAgIAAxkBAAEC..."


@pytest.mark.asyncio
async def test_llm_usage_event_can_be_created(test_session, schema) -> None:
    user = User(telegram_user_id=42)
    test_session.add(user)
    await test_session.flush()

    event = LlmUsageEvent(
        user_id=user.id,
        session_id=None,
        service=LlmService.WHISPER,
        model="whisper-1",
        usage_metric=UsageMetric.AUDIO_SECONDS,
        usage_amount=87,
        cost_usd=Decimal("0.0087"),
        prompt_version=None,
    )
    test_session.add(event)
    await test_session.commit()
    await test_session.refresh(event)

    assert event.id is not None
    assert event.service == LlmService.WHISPER
    assert event.usage_amount == 87
    assert event.cost_usd == Decimal("0.0087")


@pytest.mark.asyncio
async def test_session_request_branch_persists(test_session) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="t",
    )
    await test_session.commit()

    s = Session(
        dream_id=dream.id,
        user_id=user.id,
        type=SessionType.QUICK,
        status=SessionStatus.IN_PROGRESS,
        per_session_context="меняю работу",
        request_branch=RequestBranch.FEEL_STATE,
    )
    test_session.add(s)
    await test_session.commit()
    await test_session.refresh(s)

    assert s.per_session_context == "меняю работу"
    assert s.request_branch == RequestBranch.FEEL_STATE


def test_request_branch_values_match_spec() -> None:
    """6 веток из spec §7 + custom (свободный запрос юзера), в том же порядке."""
    assert [b.value for b in RequestBranch] == [
        "feel_state",
        "decision",
        "conflict",
        "calm",
        "new_about_self",
        "unclear",
        "custom",
    ]


@pytest.mark.asyncio
async def test_dream_image_can_be_created(test_session, schema):
    from data.models import Dream, DreamImage, IfsCategory, ImageShape, User

    user = User(telegram_user_id=11, display_name="x")
    test_session.add(user)
    await test_session.flush()
    dream = Dream(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 1, tzinfo=UTC),
        transcription="t",
    )
    test_session.add(dream)
    await test_session.flush()

    img = DreamImage(
        dream_id=dream.id,
        name="Тёмная фигура",
        ifs_category=IfsCategory.FIREFIGHTER,
        shape=ImageShape.HUMAN_LIKE,
    )
    test_session.add(img)
    await test_session.flush()

    assert img.id is not None
    assert img.analyzed_in_session_id is None


@pytest.mark.asyncio
async def test_session_state_one_to_one(test_session, schema):
    from data.models import Dream, Session, SessionState, SessionType, User

    user = User(telegram_user_id=12, display_name="x")
    test_session.add(user)
    await test_session.flush()
    dream = Dream(user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t")
    test_session.add(dream)
    await test_session.flush()
    sess = Session(dream_id=dream.id, user_id=user.id, type=SessionType.DEEP)
    test_session.add(sess)
    await test_session.flush()

    state = SessionState(session_id=sess.id, data={"target_part_name": "x", "shape": "object"})
    test_session.add(state)
    await test_session.flush()

    assert state.data["target_part_name"] == "x"


@pytest.mark.asyncio
async def test_session_message_persists(test_session, schema):
    from data.models import (
        Dream,
        LlmRole,
        Session,
        SessionMessage,
        SessionMessageContentType,
        SessionMessageDirection,
        SessionType,
        User,
    )

    user = User(telegram_user_id=13, display_name="x")
    test_session.add(user)
    await test_session.flush()
    dream = Dream(user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t")
    test_session.add(dream)
    await test_session.flush()
    sess = Session(dream_id=dream.id, user_id=user.id, type=SessionType.DEEP)
    test_session.add(sess)
    await test_session.flush()

    msg = SessionMessage(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="Где в теле замечаете?",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label="P2_FOCUS",
    )
    test_session.add(msg)
    await test_session.flush()

    assert msg.id is not None


@pytest.mark.asyncio
async def test_session_has_phase3_columns(test_session, schema):
    from data.models import Dream, P4Substep, Session, SessionPhase, SessionType, User

    user = User(telegram_user_id=14, display_name="x")
    test_session.add(user)
    await test_session.flush()
    dream = Dream(user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t")
    test_session.add(dream)
    await test_session.flush()
    sess = Session(
        dream_id=dream.id,
        user_id=user.id,
        type=SessionType.DEEP,
        current_phase=SessionPhase.P4,
        p4_substep=P4Substep.ORIGIN,
        pivot_count=1,
    )
    test_session.add(sess)
    await test_session.flush()

    assert sess.current_phase == SessionPhase.P4
    assert sess.p4_substep == P4Substep.ORIGIN
    assert sess.pivot_count == 1
    assert sess.target_image_id is None
