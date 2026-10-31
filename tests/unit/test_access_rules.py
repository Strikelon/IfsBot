from datetime import UTC, datetime, timedelta
from decimal import Decimal

from domain.access import (
    ALL_TIERS,
    TIER_EARLY_TESTER,
    TIER_FREE_INTERVIEW,
    TIER_NEW,
    interview_active,
    is_exempt,
    normalize_exempt,
    resolve_limit_usd,
)

NOW = datetime(2026, 8, 1, 12, 0, tzinfo=UTC)
EXEMPT = frozenset({"owner_user", "partner_user"})


def _limit(**kwargs):
    base = dict(
        username=None,
        access_tier=TIER_NEW,
        access_tier_until=None,
        exempt=EXEMPT,
        early_tester_limit_usd=1.0,
        new_user_limit_usd=0.5,
        now=NOW,
    )
    base.update(kwargs)
    return resolve_limit_usd(**base)


def test_all_tiers_listed():
    assert ALL_TIERS == (TIER_NEW, TIER_EARLY_TESTER, TIER_FREE_INTERVIEW)


def test_normalize_exempt_strips_at_case_and_blanks():
    assert normalize_exempt("@Owner_user, partner_user ,") == EXEMPT
    assert normalize_exempt("") == frozenset()
    assert normalize_exempt(None) == frozenset()


def test_is_exempt_case_insensitive_and_at_tolerant():
    assert is_exempt("OwNer_User", EXEMPT) is True
    assert is_exempt("@owner_user", EXEMPT) is True
    assert is_exempt("someone", EXEMPT) is False
    assert is_exempt(None, EXEMPT) is False


def test_interview_active_none_means_unlimited():
    assert interview_active(None, now=NOW) is True


def test_interview_active_future_and_past():
    assert interview_active(NOW + timedelta(days=1), now=NOW) is True
    assert interview_active(NOW - timedelta(seconds=1), now=NOW) is False


def test_interview_active_treats_naive_as_utc():
    naive_future = (NOW + timedelta(days=1)).replace(tzinfo=None)
    naive_past = (NOW - timedelta(days=1)).replace(tzinfo=None)
    assert interview_active(naive_future, now=NOW) is True
    assert interview_active(naive_past, now=NOW) is False


def test_exempt_username_beats_every_tier():
    assert _limit(username="OwNer_User", access_tier=TIER_NEW) is None
    assert _limit(username="owner_user", access_tier=TIER_EARLY_TESTER) is None


def test_active_interview_has_no_limit():
    assert _limit(
        access_tier=TIER_FREE_INTERVIEW,
        access_tier_until=NOW + timedelta(days=90),
    ) is None


def test_expired_interview_falls_back_to_new_limit():
    assert _limit(
        access_tier=TIER_FREE_INTERVIEW,
        access_tier_until=NOW - timedelta(days=1),
    ) == Decimal("0.5")


def test_early_tester_limit():
    assert _limit(access_tier=TIER_EARLY_TESTER) == Decimal("1.0")


def test_new_tier_limit():
    assert _limit(access_tier=TIER_NEW) == Decimal("0.5")


def test_unknown_tier_falls_back_to_new_limit():
    assert _limit(access_tier="whatever") == Decimal("0.5")


def test_now_defaults_to_utc_now():
    # без явного now активный интервью-срок в будущем всё ещё безлимит
    far_future = datetime.now(UTC) + timedelta(days=1)
    assert resolve_limit_usd(
        username=None,
        access_tier=TIER_FREE_INTERVIEW,
        access_tier_until=far_future,
        exempt=EXEMPT,
        early_tester_limit_usd=1.0,
        new_user_limit_usd=0.5,
    ) is None
