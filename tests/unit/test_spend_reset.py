"""Точка отсчёта трат (spend_reset_at): лимит считается только после неё."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from adapters.telegram.spend_guard import is_over_limit
from data.models import LlmService, LlmUsageEvent, UsageMetric, User
from data.repositories import UsageEventRepository


class _Settings:
    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0
    spend_limit_exempt_set = frozenset()


async def _spend_at(session, user_id: int, amount: str, when: datetime) -> None:
    """Событие трат с явным временем: record() ставит now(), а нам нужна история."""
    session.add(
        LlmUsageEvent(
            user_id=user_id,
            service=list(LlmService)[0],
            model="m",
            usage_metric=list(UsageMetric)[0],
            usage_amount=1,
            cost_usd=Decimal(amount),
            created_at=when,
        )
    )
    await session.flush()


@pytest.mark.asyncio
async def test_old_spend_ignored_after_reset(test_session):
    now = datetime.now(UTC)
    user = User(telegram_user_id=901, display_name="A", spend_reset_at=now)
    test_session.add(user)
    await test_session.flush()
    await _spend_at(test_session, user.id, "5.00", now - timedelta(days=3))
    assert await is_over_limit(test_session, user, _Settings()) is False


@pytest.mark.asyncio
async def test_spend_after_reset_counts(test_session):
    now = datetime.now(UTC)
    user = User(telegram_user_id=902, display_name="B", spend_reset_at=now - timedelta(hours=1))
    test_session.add(user)
    await test_session.flush()
    await _spend_at(test_session, user.id, "5.00", now - timedelta(days=3))
    await _spend_at(test_session, user.id, "1.00", now)
    assert await is_over_limit(test_session, user, _Settings()) is True


@pytest.mark.asyncio
async def test_without_reset_behaviour_unchanged(test_session):
    now = datetime.now(UTC)
    user = User(telegram_user_id=903, display_name="C")
    test_session.add(user)
    await test_session.flush()
    await _spend_at(test_session, user.id, "1.00", now - timedelta(days=30))
    assert await is_over_limit(test_session, user, _Settings()) is True


@pytest.mark.asyncio
async def test_total_cost_full_history_still_available(test_session):
    """Админка показывает ВСЁ потраченное — вызов без since не меняется."""
    now = datetime.now(UTC)
    user = User(telegram_user_id=904, display_name="D", spend_reset_at=now)
    test_session.add(user)
    await test_session.flush()
    await _spend_at(test_session, user.id, "0.40", now - timedelta(days=2))
    await _spend_at(test_session, user.id, "0.30", now + timedelta(minutes=1))
    repo = UsageEventRepository(test_session)
    assert await repo.total_cost_for_user(user.id) == Decimal("0.70")
    assert await repo.total_cost_for_user(user.id, since=now) == Decimal("0.30")
