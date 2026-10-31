"""Шаблоны админки рендерятся с новыми колонками и значками (батч 2026-08-18).

Тест ловит опечатки в Jinja — их не видно ни в юнит-тестах запросов, ни в линтере,
а падают они уже в браузере.
"""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from admin.queries import CardSummary, SubscriberRow
from admin.utils import (
    access_badge,
    access_line,
    avatar_color,
    avatar_initials,
    format_relative_date,
    format_usd,
    interview_badge,
    paywall_badge,
)

_TEMPLATES = Path(__file__).resolve().parents[2] / "src" / "admin" / "templates"

DAY = datetime(2026, 8, 12, tzinfo=UTC)


def _env() -> Environment:
    return Environment(loader=FileSystemLoader(_TEMPLATES), autoescape=True)


def _row(**over) -> SubscriberRow:
    base = dict(
        user_id=1, telegram_user_id=42, display_name="Катя", username="katya",
        dreams_count=2, questions_count=7, cost_usd=Decimal("0.83"),
        last_seen_at=DAY, created_at=DAY, is_active=True,
        access_tier="early_tester", paywall_shown_at=DAY, interview_passed_at=DAY,
    )
    base.update(over)
    return SubscriberRow(**base)


def _card(**over) -> CardSummary:
    base = dict(
        user_id=1, telegram_user_id=42, display_name="Катя", username="katya",
        dreams_count=2, questions_count=7, cost_usd=Decimal("1.20"),
        last_seen_at=DAY, created_at=DAY, is_active=True,
        access_tier="free_interview", access_tier_until=datetime(2026, 10, 12, tzinfo=UTC),
        paywall_shown_at=DAY, interview_passed_at=DAY,
        spend_reset_at=DAY, cost_since_reset=Decimal("0.35"),
    )
    base.update(over)
    return CardSummary(**base)


def _render_subscribers(rows) -> str:
    return _env().get_template("subscribers.html").render(
        rows=rows, total=len(rows), page=1, per_page=50, search="", sort="paywall", desc=True,
        fmt_usd=format_usd, fmt_date=format_relative_date, initials=avatar_initials,
        color=avatar_color, access_badge=access_badge, paywall_badge=paywall_badge,
        interview_badge=interview_badge, request=None,
    )


def test_subscribers_table_shows_new_columns():
    html = _render_subscribers([_row()])
    assert "Пейвол" in html
    assert "Интервью" in html
    assert "sort=paywall" in html and "sort=interview" in html
    assert "12.08.2026" in html


def test_subscribers_row_draws_three_badges():
    html = _render_subscribers([_row()])
    assert "Первый тестер · лимит $1" in html
    assert "Видел пейвол 12.08.2026" in html
    assert "Интервью пройдено 12.08.2026" in html
    # три разных угла — значки не перетирают друг друга
    assert "-top-0.5 -right-0.5" in html
    assert "-bottom-0.5 -right-0.5" in html
    assert "-top-0.5 -left-0.5" in html


def test_subscribers_plain_user_shows_dashes():
    html = _render_subscribers([_row(access_tier="new", paywall_shown_at=None, interview_passed_at=None)])
    assert "Видел пейвол" not in html
    assert "—" in html


def test_user_card_renders_two_tier_buttons_and_reset_line():
    card = _card()
    html = _env().get_template("user_detail.html").render(
        u=card, dialog=[], fmt_usd=format_usd, fmt_date=format_relative_date,
        initials=avatar_initials, color=avatar_color, access_badge=access_badge,
        paywall_badge=paywall_badge, interview_badge=interview_badge,
        access=access_line(card.access_tier, card.access_tier_until, exempt=False,
                           early_limit=1.0, new_limit=1.0),
        request=None,
    )
    assert "Обычный тестер" in html
    assert "setTier(&#39;regular&#39;)" in html or "setTier('regular')" in html
    assert "Первый тестер</button>" not in html      # третьей кнопки больше нет
    assert "безлимит на 2 месяца" in html
    assert "с момента" in html and "0.35" in html
