from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from data.models import User
from data.repositories import UserRepository


@pytest.mark.asyncio
async def test_user_has_activity_columns(test_session):
    u = User(
        telegram_user_id=123,
        display_name="Аня",
        username="anya",
        last_seen_at=datetime(2026, 6, 8, tzinfo=UTC),
        is_active=True,
    )
    test_session.add(u)
    await test_session.flush()
    fetched = (await test_session.execute(select(User).where(User.telegram_user_id == 123))).scalar_one()
    assert fetched.username == "anya"
    assert fetched.is_active is True
    assert fetched.blocked_at is None
    assert fetched.last_seen_at == datetime(2026, 6, 8, tzinfo=UTC)


@pytest.mark.asyncio
async def test_is_active_defaults_true(test_session):
    u = User(telegram_user_id=456, display_name="Боб")
    test_session.add(u)
    await test_session.flush()
    assert u.is_active is True


@pytest.mark.asyncio
async def test_touch_activity_sets_fields(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=7, display_name="X")
    now = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)
    await repo.touch_activity(user_id=user.id, username="newuser", now=now)
    await test_session.flush()
    assert user.username == "newuser"
    assert user.last_seen_at == now
    assert user.is_active is True


@pytest.mark.asyncio
async def test_touch_activity_reactivates_blocked(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=8)
    await repo.set_blocked(user_id=user.id, blocked=True, now=datetime(2026, 6, 7, tzinfo=UTC))
    await test_session.flush()
    assert user.is_active is False
    await repo.touch_activity(user_id=user.id, username=None, now=datetime(2026, 6, 8, tzinfo=UTC))
    await test_session.flush()
    assert user.is_active is True


@pytest.mark.asyncio
async def test_set_blocked_records_timestamp(test_session):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=9)
    ts = datetime(2026, 6, 8, tzinfo=UTC)
    await repo.set_blocked(user_id=user.id, blocked=True, now=ts)
    await test_session.flush()
    assert user.is_active is False
    assert user.blocked_at == ts


@pytest.mark.asyncio
async def test_dream_create_bumps_counter(test_session):
    from data.repositories import DreamRepository, UserRepository
    urepo = UserRepository(test_session)
    user = await urepo.get_or_create_by_tg_id(telegram_user_id=1, display_name="A")
    await test_session.flush()
    drepo = DreamRepository(test_session)
    await drepo.create(user_id=user.id, recorded_at=datetime(2026, 6, 8, tzinfo=UTC), transcription="x")
    await test_session.flush()
    assert user.dreams_total == 1
