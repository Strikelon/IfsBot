from datetime import UTC, datetime
from decimal import Decimal

import pytest

from data.constants import QUICK_PHASE_INITIAL
from data.models import (
    DreamStatus,
    ExperienceLevel,
    IfsCategory,
    ImageShape,
    LlmRole,
    LlmService,
    P4Substep,
    RequestBranch,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionPhase,
    SessionStatus,
    SessionType,
    UsageMetric,
)
from data.repositories import (
    DreamImageRepository,
    DreamRepository,
    SessionMessageRepository,
    SessionRepository,
    SessionStateRepository,
    UsageEventRepository,
    UserRepository,
)


@pytest.mark.asyncio
async def test_user_repo_creates_on_first_call(test_session, schema) -> None:
    repo = UserRepository(test_session)

    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()

    assert user.id is not None
    assert user.telegram_user_id == 42
    assert user.display_name == "Катя"
    assert user.profile_life_context is None
    assert user.profile_experience_level is None


@pytest.mark.asyncio
async def test_user_repo_returns_existing(test_session, schema) -> None:
    repo = UserRepository(test_session)

    u1 = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    u2 = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Иной")

    assert u1.id == u2.id
    assert u2.display_name == "Катя"  # уже созданного юзера не перезаписываем


@pytest.mark.asyncio
async def test_user_repo_update_profile(test_session, schema) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()

    await repo.update_profile(
        user_id=user.id,
        display_name="Екатерина",
        profile_life_context="меняю работу",
        profile_experience_level=ExperienceLevel.SOME_EXPERIENCE,
    )
    await test_session.commit()
    await test_session.refresh(user)

    assert user.display_name == "Екатерина"
    assert user.profile_life_context == "меняю работу"
    assert user.profile_experience_level == ExperienceLevel.SOME_EXPERIENCE


@pytest.mark.asyncio
async def test_user_repo_is_onboarded_flag(test_session, schema) -> None:
    # Semantics changed: completion is now tracked via onboarding_completed_at,
    # not derived from profile fields. Test updated accordingly.
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()

    assert repo.is_onboarded(user) is False

    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()
    await test_session.refresh(user)

    assert repo.is_onboarded(user) is True


@pytest.mark.asyncio
async def test_dream_repo_create_and_get(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="мне снился лес",
        raw_audio_file_id="abc",
    )
    await test_session.commit()
    await test_session.refresh(dream)

    fetched = await dream_repo.get_by_id(dream.id, user_id=user.id)
    assert fetched is not None
    assert fetched.transcription == "мне снился лес"
    assert fetched.status == DreamStatus.RECORDED


@pytest.mark.asyncio
async def test_dream_repo_get_by_id_respects_user_isolation(test_session, schema) -> None:
    """Другой юзер не должен получить чужой сон."""
    user_repo = UserRepository(test_session)
    a = await user_repo.get_or_create_by_tg_id(telegram_user_id=1)
    b = await user_repo.get_or_create_by_tg_id(telegram_user_id=2)
    await test_session.commit()

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=a.id,
        recorded_at=datetime.now(UTC),
        transcription="принадлежит A",
    )
    await test_session.commit()

    # B не видит сон A:
    assert await dream_repo.get_by_id(dream.id, user_id=b.id) is None
    # A видит:
    assert await dream_repo.get_by_id(dream.id, user_id=a.id) is not None


@pytest.mark.asyncio
async def test_dream_repo_list_by_user_reverse_chronological(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()

    dream_repo = DreamRepository(test_session)
    d1 = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="первый",
    )
    d2 = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 12, tzinfo=UTC),
        transcription="второй",
    )
    d3 = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 11, tzinfo=UTC),
        transcription="третий",
    )
    await test_session.commit()

    dreams = await dream_repo.list_by_user(user_id=user.id, limit=20)
    assert [d.id for d in dreams] == [d2.id, d3.id, d1.id]


@pytest.mark.asyncio
async def test_usage_event_repo_record(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    await test_session.commit()

    repo = UsageEventRepository(test_session)
    event = await repo.record(
        user_id=user.id,
        service=LlmService.WHISPER,
        model="whisper-1",
        usage_metric=UsageMetric.AUDIO_SECONDS,
        usage_amount=87,
        cost_usd=Decimal("0.008700"),
    )
    await test_session.commit()
    await test_session.refresh(event)

    assert event.id is not None
    assert event.user_id == user.id
    assert event.service == LlmService.WHISPER
    assert event.cost_usd == Decimal("0.008700")
    assert event.session_id is None


@pytest.mark.asyncio
async def test_session_repo_creates_quick_session(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    await test_session.commit()

    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.id is not None
    assert sess.dream_id == dream.id
    assert sess.user_id == user.id
    assert sess.type == SessionType.QUICK
    assert sess.status == SessionStatus.IN_PROGRESS
    assert sess.per_session_context is None
    assert sess.request_branch is None
    assert sess.finished_at is None


@pytest.mark.asyncio
async def test_session_repo_create_defaults_is_onboarding_false(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    await test_session.commit()

    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.is_onboarding_session is False


@pytest.mark.asyncio
async def test_session_repo_create_with_is_onboarding_true(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    await test_session.commit()

    repo = SessionRepository(test_session)
    sess = await repo.create(
        dream_id=dream.id,
        user_id=user.id,
        type_=SessionType.QUICK,
        is_onboarding_session=True,
    )
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.is_onboarding_session is True


@pytest.mark.asyncio
async def test_session_repo_get_active_for_user_dream(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    await test_session.commit()

    repo = SessionRepository(test_session)
    assert await repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id) is None

    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()
    found = await repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id)
    assert found is not None
    assert found.id == sess.id


@pytest.mark.asyncio
async def test_session_repo_complete_clears_active(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await repo.complete(session_id=sess.id)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.status == SessionStatus.COMPLETED
    assert sess.finished_at is not None
    # И больше не считается активной:
    assert await repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id) is None


@pytest.mark.asyncio
async def test_session_repo_reopen_returns_to_in_progress(test_session, schema) -> None:
    """reopen() возвращает completed-сессию в IN_PROGRESS и очищает finished_at.

    Используется в quick.v2.1 UX continue: юзер тапает «Продолжить тот разбор»
    на сне с completed Quick — старая сессия становится in_progress, qa_history
    из session_messages подгружается."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await repo.complete(session_id=sess.id)
    await test_session.commit()
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.COMPLETED
    assert sess.finished_at is not None

    await repo.reopen(session_id=sess.id)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.status == SessionStatus.IN_PROGRESS
    assert sess.finished_at is None


@pytest.mark.asyncio
async def test_session_repo_set_context_and_branch(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="t"
    )
    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK)
    await test_session.commit()

    await repo.set_per_session_context(session_id=sess.id, context="меняю работу")
    await repo.set_request_branch(session_id=sess.id, branch=RequestBranch.DECISION)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.per_session_context == "меняю работу"
    assert sess.request_branch == RequestBranch.DECISION


@pytest.mark.asyncio
async def test_session_repo_get_by_id_respects_user(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    a = await user_repo.get_or_create_by_tg_id(telegram_user_id=1)
    b = await user_repo.get_or_create_by_tg_id(telegram_user_id=2)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(user_id=a.id, recorded_at=datetime.now(UTC), transcription="t")
    repo = SessionRepository(test_session)
    sess = await repo.create(dream_id=dream.id, user_id=a.id, type_=SessionType.QUICK)
    await test_session.commit()

    assert await repo.get_by_id(sess.id, user_id=a.id) is not None
    assert await repo.get_by_id(sess.id, user_id=b.id) is None


@pytest.mark.asyncio
async def test_dream_image_repository_bulk_create_and_list(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=100, display_name="x")
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    await test_session.commit()

    repo = DreamImageRepository(test_session)
    images = await repo.bulk_create(
        dream_id=dream.id,
        images=[
            {"name": "Тёмная фигура", "ifs_category": IfsCategory.FIREFIGHTER, "shape": ImageShape.HUMAN_LIKE},
            {"name": "Туман", "ifs_category": IfsCategory.UNCLEAR, "shape": ImageShape.PHENOMENON},
        ],
    )
    assert len(images) == 2
    assert all(img.id is not None for img in images)

    listed = await repo.list_by_dream(dream_id=dream.id)
    assert len(listed) == 2
    assert {i.name for i in listed} == {"Тёмная фигура", "Туман"}


@pytest.mark.asyncio
async def test_dream_image_repository_get_by_id_isolated_by_user(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    u1 = await user_repo.get_or_create_by_tg_id(telegram_user_id=200, display_name="u1")
    u2 = await user_repo.get_or_create_by_tg_id(telegram_user_id=201, display_name="u2")
    d = await DreamRepository(test_session).create(
        user_id=u1.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    await test_session.commit()
    repo = DreamImageRepository(test_session)
    [img] = await repo.bulk_create(
        dream_id=d.id,
        images=[{"name": "X", "ifs_category": IfsCategory.MANAGER, "shape": ImageShape.OBJECT}],
    )
    await test_session.commit()

    assert (await repo.get_by_id_for_user(image_id=img.id, user_id=u1.id)) is not None
    assert (await repo.get_by_id_for_user(image_id=img.id, user_id=u2.id)) is None


@pytest.mark.asyncio
async def test_dream_image_repository_mark_analyzed(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=300, display_name="x")
    d = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    await test_session.commit()
    img_repo = DreamImageRepository(test_session)
    [img] = await img_repo.bulk_create(
        dream_id=d.id,
        images=[{"name": "X", "ifs_category": IfsCategory.MANAGER, "shape": ImageShape.OBJECT}],
    )
    sess = await SessionRepository(test_session).create(
        dream_id=d.id, user_id=user.id, type_=SessionType.DEEP
    )
    await test_session.commit()

    await img_repo.mark_analyzed(image_id=img.id, session_id=sess.id)
    await test_session.commit()
    await test_session.refresh(img)
    assert img.analyzed_in_session_id == sess.id


@pytest.mark.asyncio
async def test_session_state_repository_upsert_and_get(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=400, display_name="x")
    d = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    sess = await SessionRepository(test_session).create(
        dream_id=d.id, user_id=user.id, type_=SessionType.DEEP
    )
    await test_session.commit()

    repo = SessionStateRepository(test_session)
    await repo.upsert(session_id=sess.id, data={"target_part_name": "Тень", "shape": "human-like"})
    await test_session.commit()

    fetched = await repo.get_data(session_id=sess.id)
    assert fetched == {"target_part_name": "Тень", "shape": "human-like"}

    await repo.upsert(session_id=sess.id, data={"target_part_name": "Тень", "shape": "human-like", "part_role": "охраняет"})
    await test_session.commit()
    fetched2 = await repo.get_data(session_id=sess.id)
    assert fetched2["part_role"] == "охраняет"


@pytest.mark.asyncio
async def test_session_message_repository_append_and_list(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=500, display_name="x")
    d = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    sess = await SessionRepository(test_session).create(
        dream_id=d.id, user_id=user.id, type_=SessionType.DEEP
    )
    await test_session.commit()

    repo = SessionMessageRepository(test_session)
    await repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="Где в теле?",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label="P2_FOCUS",
    )
    await repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=SessionMessageContentType.TEXT,
        content="в груди давит",
    )
    await test_session.commit()

    msgs = await repo.list_for_resume(session_id=sess.id, limit=8)
    assert len(msgs) == 2
    assert msgs[0].content == "Где в теле?"
    assert msgs[1].direction == SessionMessageDirection.USER_TO_BOT


@pytest.mark.asyncio
async def test_session_repository_phase3_setters(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=600, display_name="x")
    d = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 1, tzinfo=UTC), transcription="t"
    )
    sess = await SessionRepository(test_session).create(
        dream_id=d.id, user_id=user.id, type_=SessionType.DEEP
    )
    img_repo = DreamImageRepository(test_session)
    [img] = await img_repo.bulk_create(
        dream_id=d.id,
        images=[{"name": "X", "ifs_category": IfsCategory.MANAGER, "shape": ImageShape.OBJECT}],
    )
    await test_session.commit()

    repo = SessionRepository(test_session)
    await repo.set_target_image_id(session_id=sess.id, image_id=img.id)
    await repo.set_current_phase(session_id=sess.id, phase=SessionPhase.P4)
    await repo.set_p4_substep(session_id=sess.id, substep=P4Substep.ORIGIN)
    await repo.increment_pivot_count(session_id=sess.id)
    await repo.increment_pivot_count(session_id=sess.id)
    await test_session.commit()
    await test_session.refresh(sess)

    assert sess.target_image_id == img.id
    assert sess.current_phase == SessionPhase.P4
    assert sess.p4_substep == P4Substep.ORIGIN
    assert sess.pivot_count == 2


# ---------------------------------------------------------------------------
# UserRepository — new onboarding_completed_at methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_onboarded_returns_false_for_new_user(test_session, schema) -> None:
    """Fresh user has onboarding_completed_at=None → is_onboarded is False,
    even if profile fields ARE set."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=700)
    await test_session.commit()

    # Populate profile fields — old semantics would have returned True here.
    await repo.update_profile(
        user_id=user.id,
        profile_life_context="some context",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await test_session.commit()
    await test_session.refresh(user)

    assert user.onboarding_completed_at is None
    assert repo.is_onboarded(user) is False


@pytest.mark.asyncio
async def test_is_onboarded_returns_true_after_mark_onboarded(test_session, schema) -> None:
    """ORM mutation: in-memory user reflects the change immediately, no refresh needed."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=701)
    await test_session.commit()

    assert repo.is_onboarded(user) is False

    await repo.mark_onboarded(user_id=user.id)

    # In-memory object must already reflect the change — no flush/refresh required.
    assert repo.is_onboarded(user) is True

    await test_session.commit()
    await test_session.refresh(user)
    assert repo.is_onboarded(user) is True


@pytest.mark.asyncio
async def test_mark_onboarded_sets_recent_timestamp(test_session, schema) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=702)
    await test_session.commit()

    before = datetime.now(UTC)
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()
    after = datetime.now(UTC)
    await test_session.refresh(user)

    assert user.onboarding_completed_at is not None
    ts = user.onboarding_completed_at
    # SQLite may return naive datetimes — normalise for comparison.
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    assert before <= ts <= after


@pytest.mark.asyncio
async def test_update_display_name_if_changed_returns_false_when_same(
    test_session, schema
) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=703, display_name="Катя")
    await test_session.commit()

    result = await repo.update_display_name_if_changed(user_id=user.id, display_name="Катя")

    assert result is False
    # In-memory object unchanged
    assert user.display_name == "Катя"


@pytest.mark.asyncio
async def test_update_display_name_if_changed_returns_true_when_different(
    test_session, schema
) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=704, display_name="Катя")
    await test_session.commit()

    result = await repo.update_display_name_if_changed(user_id=user.id, display_name="Екатерина")

    assert result is True
    # In-memory object is already updated (ORM mutation, no flush needed to read)
    assert user.display_name == "Екатерина"

    await test_session.flush()
    await test_session.refresh(user)
    assert user.display_name == "Екатерина"


@pytest.mark.asyncio
async def test_update_display_name_if_changed_truncates_to_128(test_session, schema) -> None:
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=705, display_name="x")
    await test_session.commit()

    long_name = "А" * 200
    result = await repo.update_display_name_if_changed(user_id=user.id, display_name=long_name)

    assert result is True
    assert len(user.display_name) == 128
    assert user.display_name == "А" * 128


@pytest.mark.asyncio
async def test_update_display_name_if_changed_no_user_returns_false(
    test_session, schema
) -> None:
    repo = UserRepository(test_session)

    result = await repo.update_display_name_if_changed(
        user_id=999999, display_name="Фантом"
    )

    assert result is False


# ---------------------------------------------------------------------------
# DreamRepository.dream_has_analysis
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_has_analysis_false_when_no_session(test_session, schema) -> None:
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон",
    )
    await test_session.commit()

    assert await dream_repo.dream_has_analysis(dream_id=dream.id) is False


@pytest.mark.asyncio
async def test_dream_has_analysis_false_when_session_without_initial(
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
    await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    # session exists but no initial message
    await test_session.commit()

    assert await dream_repo.dream_has_analysis(dream_id=dream.id) is False


@pytest.mark.asyncio
async def test_dream_has_analysis_true_after_initial(test_session, schema) -> None:
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
    await test_session.commit()

    assert await dream_repo.dream_has_analysis(dream_id=dream.id) is True
