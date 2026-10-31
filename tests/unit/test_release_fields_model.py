"""Новые поля users из батча 2026-08-18: дефолты и nullability."""

from datetime import UTC, datetime

import pytest

from data.models import User
from data.repositories import UserRepository
from domain.access import INTERVIEW_MONTHS


@pytest.mark.asyncio
async def test_new_fields_default_to_none(test_session):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=777)
    assert user.spend_reset_at is None
    assert user.paywall_shown_at is None
    assert user.interview_passed_at is None
    assert user.tier_before_interview is None


@pytest.mark.asyncio
async def test_fields_are_writable(test_session):
    user = User(telegram_user_id=778, display_name="X")
    user.spend_reset_at = datetime(2026, 8, 18, tzinfo=UTC)
    user.paywall_shown_at = datetime(2026, 8, 10, tzinfo=UTC)
    user.interview_passed_at = datetime(2026, 8, 12, tzinfo=UTC)
    user.tier_before_interview = "early_tester"
    test_session.add(user)
    await test_session.flush()
    assert user.tier_before_interview == "early_tester"


def test_interview_months_is_two():
    assert INTERVIEW_MONTHS == 2
