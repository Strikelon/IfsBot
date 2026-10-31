"""Чистые правила тиров доступа.

Без ORM и без Settings: один и тот же расчёт зовут бот (`spend_guard`) и админка
(карточка юзера). Любое изменение правил делается здесь, а не в двух местах.
"""

from __future__ import annotations

import calendar
from datetime import UTC, datetime
from decimal import Decimal
from typing import Protocol

TIER_NEW = "new"
TIER_EARLY_TESTER = "early_tester"
TIER_FREE_INTERVIEW = "free_interview"

ALL_TIERS: tuple[str, ...] = (TIER_NEW, TIER_EARLY_TESTER, TIER_FREE_INTERVIEW)

# Срок бесплатного безлимита за интервью. Одна константа на админку и на текст
# пейвола: обещание юзеру и срок в БД обязаны совпадать.
INTERVIEW_MONTHS = 2


def normalize_exempt(raw: str | None) -> frozenset[str]:
    """Comma-separated юзернеймы -> нормализованный набор (lowercase, без @, без пустых)."""
    if not raw:
        return frozenset()
    return frozenset(
        part.strip().lstrip("@").lower()
        for part in raw.split(",")
        if part.strip().lstrip("@")
    )


def is_exempt(username: str | None, exempt: frozenset[str]) -> bool:
    """True, если юзернейм в списке освобождённых от лимита (основатели)."""
    if not username:
        return False
    return username.lstrip("@").lower() in exempt


def _as_utc(dt: datetime) -> datetime:
    """Наивную дату из БД трактуем как UTC: иначе сравнение с aware-now бросит TypeError."""
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)


def interview_active(until: datetime | None, *, now: datetime | None = None) -> bool:
    """Активен ли бесплатный период интервью.

    `until is None` – бессрочно: через админку так не поставить, но ручной SQL
    не должен молча означать «уже истёк».
    """
    if until is None:
        return True
    return _as_utc(until) > (now or datetime.now(UTC))


def resolve_limit_usd(
    *,
    username: str | None,
    access_tier: str | None,
    access_tier_until: datetime | None,
    exempt: frozenset[str],
    early_tester_limit_usd: float,
    new_user_limit_usd: float,
    now: datetime | None = None,
) -> Decimal | None:
    """Порог накопленных трат для юзера. None – лимита нет вовсе.

    Порядок приоритетов зафиксирован спекой: основатели -> активное интервью ->
    первые тестеры -> все остальные (включая интервью с истёкшим сроком).
    """
    if is_exempt(username, exempt):
        return None
    if access_tier == TIER_FREE_INTERVIEW and interview_active(access_tier_until, now=now):
        return None
    if access_tier == TIER_EARLY_TESTER:
        return Decimal(str(early_tester_limit_usd))
    return Decimal(str(new_user_limit_usd))


def add_months(dt: datetime, months: int) -> datetime:
    """Прибавить календарные месяцы с зажимом дня (30.11 + 3 = 28.02).

    Не timedelta(days=90): человеку обещаны «два месяца», и срок в БД должен
    совпадать с обещанием в тексте.
    """
    total = dt.month - 1 + months
    year = dt.year + total // 12
    month = total % 12 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)


class TierUser(Protocol):
    """Минимум полей юзера, который нужен смене тира.

    Protocol, а не ORM-модель: правила тиров живут в domain и не тянут сюда data.
    """

    access_tier: str
    access_tier_until: datetime | None
    tier_before_interview: str | None
    interview_passed_at: datetime | None


TIER_ACTIONS: tuple[str, ...] = ("regular", "interview")


def apply_tier_change(user: TierUser, action: str, *, now: datetime) -> None:
    """Применить смену статуса к юзеру (без коммита).

    `interview` — безлимит на INTERVIEW_MONTHS месяцев от сегодня; повторное
    назначение продлевает срок, но дата ПЕРВОГО интервью не переписывается.
    `regular` — вернуть исторический тир: иначе снятие интервью стирало бы
    бейджик «1» у первых тестеров.

    Живёт в domain, потому что зовут из двух мест: кнопка в админке и команда
    /interview в боте (2026-08-24). Две копии правил разъехались бы.
    """
    if action == "interview":
        if user.access_tier != TIER_FREE_INTERVIEW:
            user.tier_before_interview = user.access_tier
        user.access_tier = TIER_FREE_INTERVIEW
        user.access_tier_until = add_months(now, INTERVIEW_MONTHS)
        if user.interview_passed_at is None:
            user.interview_passed_at = now
        return
    if action == "regular":
        user.access_tier = user.tier_before_interview or TIER_NEW
        user.access_tier_until = None
        user.tier_before_interview = None
        return
    raise ValueError(f"unknown tier action: {action}")
