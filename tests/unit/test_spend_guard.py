from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from adapters.telegram.spend_guard import is_over_limit, user_limit_usd
from data.models import LlmService, UsageMetric, User
from data.repositories import UsageEventRepository
from domain.access import TIER_EARLY_TESTER, TIER_FREE_INTERVIEW, TIER_NEW


class _Settings:
    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0  # старые тесты писались до тиров: одинаковый порог
    spend_limit_exempt_set = frozenset({"owner_user"})


async def _spend(session, user_id, amount):
    await UsageEventRepository(session).record(
        user_id=user_id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal(amount),
    )


@pytest.mark.asyncio
async def test_under_limit_not_blocked(test_session):
    u = User(telegram_user_id=1, display_name="A", username="bob")
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "0.50")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _Settings()) is False


@pytest.mark.asyncio
async def test_at_or_over_limit_blocked(test_session):
    u = User(telegram_user_id=2, display_name="B", username="bob")
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "1.00")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _Settings()) is True


@pytest.mark.asyncio
async def test_exempt_username_never_blocked(test_session):
    u = User(telegram_user_id=3, display_name="C", username="OwNer_User")  # case-insensitive
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "9.99")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _Settings()) is False


# --- тиры доступа (2026-08-01) ---


class _TieredSettings:
    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 0.5
    spend_limit_exempt_set = frozenset({"owner_user"})


def _user(tg_id: int, tier: str, *, username: str | None = None, until=None) -> User:
    return User(
        telegram_user_id=tg_id,
        display_name="X",
        username=username,
        access_tier=tier,
        access_tier_until=until,
    )


def test_user_limit_early_tester_is_one_dollar():
    assert user_limit_usd(_user(10, TIER_EARLY_TESTER), _TieredSettings()) == Decimal("1.0")


def test_user_limit_new_is_half_dollar():
    assert user_limit_usd(_user(11, TIER_NEW), _TieredSettings()) == Decimal("0.5")


def test_user_limit_active_interview_is_none():
    until = datetime.now(UTC) + timedelta(days=30)
    assert user_limit_usd(_user(12, TIER_FREE_INTERVIEW, until=until), _TieredSettings()) is None


def test_user_limit_expired_interview_drops_to_new():
    until = datetime.now(UTC) - timedelta(days=1)
    assert user_limit_usd(
        _user(13, TIER_FREE_INTERVIEW, until=until), _TieredSettings()
    ) == Decimal("0.5")


def test_user_limit_exempt_username_beats_tier():
    assert user_limit_usd(_user(14, TIER_NEW, username="OwNer_User"), _TieredSettings()) is None


@pytest.mark.asyncio
async def test_new_tier_blocked_at_half_dollar(test_session):
    u = _user(15, TIER_NEW)
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "0.50")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _TieredSettings()) is True


@pytest.mark.asyncio
async def test_early_tester_still_ok_at_half_dollar(test_session):
    u = _user(16, TIER_EARLY_TESTER)
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "0.60")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _TieredSettings()) is False


@pytest.mark.asyncio
async def test_active_interview_never_blocked(test_session):
    u = _user(17, TIER_FREE_INTERVIEW, until=datetime.now(UTC) + timedelta(days=30))
    test_session.add(u)
    await test_session.flush()
    await _spend(test_session, u.id, "9.99")
    await test_session.flush()
    assert await is_over_limit(test_session, u, _TieredSettings()) is False
