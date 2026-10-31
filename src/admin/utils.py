from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from zoneinfo import ZoneInfo

from domain.access import (
    TIER_EARLY_TESTER,
    TIER_FREE_INTERVIEW,
    add_months,
    interview_active,
)

_PALETTE = ["#ef4444", "#f59e0b", "#10b981", "#3b82f6", "#8b5cf6", "#ec4899", "#14b8a6"]


def format_usd(value: Decimal | float | None) -> str:
    if value is None:
        value = Decimal("0")
    return f"${Decimal(str(value)):.4f}"


def format_relative_date(dt: datetime | None, *, now: datetime | None = None) -> str:
    if dt is None:
        return "—"
    now = now or datetime.now(UTC)
    days = (now.date() - dt.date()).days
    if days <= 0:
        return "Сегодня"
    if days == 1:
        return "Вчера"
    return f"{days} дн. назад"


def avatar_initials(name: str | None) -> str:
    if not name or not name.strip():
        return "?"
    parts = name.strip().split()
    if len(parts) == 1:
        return parts[0][0].upper()
    return (parts[0][0] + parts[1][0]).upper()


def avatar_color(name: str | None) -> str:
    key = name or "?"
    return _PALETTE[sum(ord(c) for c in key) % len(_PALETTE)]


def format_in_tz(dt: datetime | None, *, tz: str, fmt: str) -> str:
    """Format an aware (or naive-assumed-UTC) datetime in `tz`."""
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(ZoneInfo(tz)).strftime(fmt)


def _fmt_day(dt: datetime) -> str:
    """Дата в dd.mm.yyyy; наивную из БД трактуем как UTC."""
    return (dt if dt.tzinfo else dt.replace(tzinfo=UTC)).strftime("%d.%m.%Y")


def access_badge(
    tier: str | None, until: datetime | None, *, now: datetime | None = None
) -> dict | None:
    """Значок тира у аватара: {'symbol', 'bg', 'corner', 'title'} или None.

    Правый верхний угол — за тиром; пейвол и интервью занимают свои углы, чтобы
    три значка не перетирали друг друга (батч 2026-08-18).
    """
    if tier == TIER_EARLY_TESTER:
        return {
            "symbol": "1", "bg": "bg-amber-400", "corner": "top-right",
            "title": "Первый тестер · лимит $1",
        }
    if tier == TIER_FREE_INTERVIEW:
        if interview_active(until, now=now):
            suffix = f" до {_fmt_day(until)}" if until else " бессрочно"
            return {
                "symbol": "∞", "bg": "bg-emerald-500", "corner": "top-right",
                "title": f"Безлимит{suffix}",
            }
        return {
            "symbol": "∞", "bg": "bg-gray-400", "corner": "top-right",
            "title": f"Безлимит истёк {_fmt_day(until)}",
        }
    return None


def paywall_badge(shown_at: datetime | None) -> dict | None:
    """Значок «видел пейвол» — правый нижний угол, чтобы не спорить с тиром."""
    if shown_at is None:
        return None
    return {
        "symbol": "$",
        "bg": "bg-rose-500",
        "corner": "bottom-right",
        "title": f"Видел пейвол {_fmt_day(shown_at)}",
    }


def interview_badge(passed_at: datetime | None) -> dict | None:
    """Значок «интервью пройдено» — левый верхний угол; переживает истечение срока."""
    if passed_at is None:
        return None
    return {
        "symbol": "✓",
        "bg": "bg-sky-500",
        "corner": "top-left",
        "title": f"Интервью пройдено {_fmt_day(passed_at)}",
    }


def access_line(
    tier: str | None,
    until: datetime | None,
    *,
    exempt: bool,
    early_limit: float,
    new_limit: float,
    now: datetime | None = None,
) -> str:
    """Строка «Доступ» для карточки юзера."""
    if exempt:
        return "Основатель, без лимита"
    if tier == TIER_FREE_INTERVIEW:
        if interview_active(until, now=now):
            return (
                f"Интервью, безлимит до {_fmt_day(until)}"
                if until
                else "Интервью, безлимит бессрочно"
            )
        return f"Интервью истекло {_fmt_day(until)}, лимит ${new_limit:.2f}"
    if tier == TIER_EARLY_TESTER:
        return f"Первый тестер, лимит ${early_limit:.2f}"
    return f"Обычный, лимит ${new_limit:.2f}"


# add_months переехал в domain/access.py вместе с apply_tier_change (2026-08-24):
# те же правила зовёт команда /interview в боте. Здесь — реэкспорт для шаблонов
# и старых импортов.
__all__ = [*globals().get("__all__", []), "add_months"]
