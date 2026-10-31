"""UI-утилиты, разделяемые между handlers."""

from __future__ import annotations

from datetime import datetime

_MONTHS_RU = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


def format_date_ru(dt: datetime) -> str:
    """`10 мая` — дата в русском human-readable формате. Phase 5 учтёт user.locale."""
    return f"{dt.day} {_MONTHS_RU[dt.month]}"
