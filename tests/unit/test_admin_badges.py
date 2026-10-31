"""Три значка у аватара не конфликтуют: тир, пейвол, интервью."""

from datetime import UTC, datetime

from admin.utils import access_badge, interview_badge, paywall_badge
from domain.access import TIER_EARLY_TESTER, TIER_NEW

DAY = datetime(2026, 8, 12, tzinfo=UTC)


def test_paywall_badge_present_only_when_shown():
    assert paywall_badge(None) is None
    badge = paywall_badge(DAY)
    assert badge["symbol"] == "$"
    assert badge["corner"] == "bottom-right"
    assert "12.08.2026" in badge["title"]


def test_interview_badge_present_only_when_passed():
    assert interview_badge(None) is None
    badge = interview_badge(DAY)
    assert badge["symbol"] == "✓"
    assert badge["corner"] == "top-left"
    assert "12.08.2026" in badge["title"]


def test_tier_badge_keeps_top_right_corner():
    badge = access_badge(TIER_EARLY_TESTER, None)
    assert badge["symbol"] == "1"
    assert badge["corner"] == "top-right"


def test_three_badges_occupy_three_corners():
    corners = {
        access_badge(TIER_EARLY_TESTER, None)["corner"],
        paywall_badge(DAY)["corner"],
        interview_badge(DAY)["corner"],
    }
    assert len(corners) == 3


def test_no_badges_for_plain_user():
    assert access_badge(TIER_NEW, None) is None
    assert paywall_badge(None) is None
    assert interview_badge(None) is None
