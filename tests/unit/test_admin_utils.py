from datetime import UTC, datetime, timedelta
from decimal import Decimal

from admin.utils import (
    access_badge,
    access_line,
    add_months,
    avatar_color,
    avatar_initials,
    format_in_tz,
    format_relative_date,
    format_usd,
)

_NOW = datetime(2026, 6, 8, 12, 0, tzinfo=UTC)


def test_format_usd():
    assert format_usd(Decimal("0.0042")) == "$0.0042"
    assert format_usd(Decimal("0")) == "$0.0000"
    assert format_usd(None) == "$0.0000"


def test_format_relative_date():
    assert format_relative_date(_NOW, now=_NOW) == "Сегодня"
    assert format_relative_date(_NOW - timedelta(days=1), now=_NOW) == "Вчера"
    assert format_relative_date(_NOW - timedelta(days=3), now=_NOW) == "3 дн. назад"
    assert format_relative_date(None, now=_NOW) == "—"


def test_avatar_initials():
    assert avatar_initials("Анна Каренина") == "АК"
    assert avatar_initials("Боб") == "Б"
    assert avatar_initials(None) == "?"
    assert avatar_initials("   ") == "?"


def test_avatar_color_is_stable():
    assert avatar_color("Аня") == avatar_color("Аня")
    assert avatar_color(None)  # must not raise, returns some color


def test_format_in_tz_moscow():
    dt = datetime(2026, 6, 8, 14, 37, tzinfo=UTC)  # 14:37 UTC -> 17:37 Moscow
    assert format_in_tz(dt, tz="Europe/Moscow", fmt="%d.%m.%Y %H:%M") == "08.06.2026 17:37"


def test_format_in_tz_none():
    assert format_in_tz(None, tz="Europe/Moscow", fmt="%H:%M") == ""


def test_format_in_tz_naive_assumed_utc():
    # naive datetimes (e.g. from SQLite) are assumed to be UTC
    dt = datetime(2026, 6, 8, 14, 37)
    assert format_in_tz(dt, tz="Europe/Moscow", fmt="%H:%M") == "17:37"


# --- тиры доступа (2026-08-01) ---

_TIER_NOW = datetime(2026, 8, 1, tzinfo=UTC)


def test_badge_none_for_plain_user():
    assert access_badge("new", None, now=_TIER_NOW) is None
    assert access_badge(None, None, now=_TIER_NOW) is None


def test_badge_gold_one_for_early_tester():
    b = access_badge("early_tester", None, now=_TIER_NOW)
    assert b["symbol"] == "1"
    assert b["bg"] == "bg-amber-400"
    assert b["title"] == "Первый тестер · лимит $1"


def test_badge_emerald_infinity_for_active_interview():
    b = access_badge("free_interview", datetime(2026, 11, 1, tzinfo=UTC), now=_TIER_NOW)
    assert b["symbol"] == "∞"
    assert b["bg"] == "bg-emerald-500"
    assert b["title"] == "Безлимит до 01.11.2026"


def test_badge_gray_for_expired_interview():
    b = access_badge("free_interview", datetime(2026, 7, 1, tzinfo=UTC), now=_TIER_NOW)
    assert b["symbol"] == "∞"
    assert b["bg"] == "bg-gray-400"
    assert b["title"] == "Безлимит истёк 01.07.2026"


def test_badge_handles_naive_datetime_from_db():
    b = access_badge("free_interview", datetime(2026, 11, 1), now=_TIER_NOW)
    assert b["bg"] == "bg-emerald-500"


def test_line_for_founder():
    assert access_line(
        "new", None, exempt=True, early_limit=1.0, new_limit=0.5, now=_TIER_NOW
    ) == "Основатель, без лимита"


def test_line_for_early_tester():
    assert access_line(
        "early_tester", None, exempt=False, early_limit=1.0, new_limit=0.5, now=_TIER_NOW
    ) == "Первый тестер, лимит $1.00"


def test_line_for_active_interview():
    assert access_line(
        "free_interview",
        datetime(2026, 11, 1, tzinfo=UTC),
        exempt=False,
        early_limit=1.0,
        new_limit=0.5,
        now=_TIER_NOW,
    ) == "Интервью, безлимит до 01.11.2026"


def test_line_for_expired_interview():
    assert access_line(
        "free_interview",
        datetime(2026, 7, 1, tzinfo=UTC),
        exempt=False,
        early_limit=1.0,
        new_limit=0.5,
        now=_TIER_NOW,
    ) == "Интервью истекло 01.07.2026, лимит $0.50"


def test_line_for_new():
    assert access_line(
        "new", None, exempt=False, early_limit=1.0, new_limit=0.5, now=_TIER_NOW
    ) == "Обычный, лимит $0.50"


def test_add_months_plain():
    assert add_months(datetime(2026, 8, 1, tzinfo=UTC), 3) == datetime(2026, 11, 1, tzinfo=UTC)


def test_add_months_crosses_year():
    assert add_months(datetime(2026, 11, 15, tzinfo=UTC), 3) == datetime(2027, 2, 15, tzinfo=UTC)


def test_add_months_clamps_short_month():
    assert add_months(datetime(2026, 11, 30, tzinfo=UTC), 3) == datetime(2027, 2, 28, tzinfo=UTC)


def test_add_months_clamps_to_leap_day():
    assert add_months(datetime(2027, 11, 30, tzinfo=UTC), 3) == datetime(2028, 2, 29, tzinfo=UTC)


def test_add_months_keeps_time_and_tz():
    src = datetime(2026, 5, 31, 14, 30, tzinfo=UTC)
    assert add_months(src, 3) == datetime(2026, 8, 31, 14, 30, tzinfo=UTC)
