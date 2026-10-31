"""Две кнопки статуса: интервью на 2 месяца и возврат в исторический тир."""

from datetime import UTC, datetime

import pytest

from admin.routers.subscribers import apply_tier_change
from data.models import User
from domain.access import TIER_EARLY_TESTER, TIER_FREE_INTERVIEW, TIER_NEW

NOW = datetime(2026, 8, 18, 12, 0, tzinfo=UTC)


def _user(tier: str = TIER_NEW) -> User:
    return User(telegram_user_id=1, display_name="A", access_tier=tier)


def test_interview_grants_two_months_and_remembers_previous():
    user = _user(TIER_EARLY_TESTER)
    apply_tier_change(user, "interview", now=NOW)
    assert user.access_tier == TIER_FREE_INTERVIEW
    assert user.access_tier_until == datetime(2026, 10, 18, 12, 0, tzinfo=UTC)
    assert user.interview_passed_at == NOW
    assert user.tier_before_interview == TIER_EARLY_TESTER


def test_second_interview_extends_but_keeps_first_date():
    user = _user(TIER_EARLY_TESTER)
    apply_tier_change(user, "interview", now=NOW)
    later = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    apply_tier_change(user, "interview", now=later)
    assert user.access_tier_until == datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
    assert user.interview_passed_at == NOW           # дата ПЕРВОГО интервью
    assert user.tier_before_interview == TIER_EARLY_TESTER  # не затирается на free_interview


def test_regular_restores_historical_tier():
    user = _user(TIER_EARLY_TESTER)
    apply_tier_change(user, "interview", now=NOW)
    apply_tier_change(user, "regular", now=NOW)
    assert user.access_tier == TIER_EARLY_TESTER     # бейджик «1» вернулся
    assert user.access_tier_until is None
    assert user.tier_before_interview is None
    assert user.interview_passed_at == NOW           # факт интервью остаётся


def test_regular_without_history_falls_back_to_new():
    user = _user(TIER_FREE_INTERVIEW)
    apply_tier_change(user, "regular", now=NOW)
    assert user.access_tier == TIER_NEW


def test_unknown_action_raises():
    with pytest.raises(ValueError):
        apply_tier_change(_user(), "founder", now=NOW)
