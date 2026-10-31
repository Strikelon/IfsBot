"""Расчёт retention-матриц по календарным корзинам (cohort × period).

Чистые функции: на вход — список UserActivity (сырые даты на юзера),
на выход — матрицы % удержания. Никакого IO/БД (тестируется на фикстурах).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

_METRIC_FIELD = {
    "launch": "active_days",
    "analysis": "analysis_days",
    "qa": "qa_days",
    "return": "return_days",
}


@dataclass
class UserActivity:
    user_id: int
    registered_at: datetime
    source: str | None
    active_days: list[date]
    analysis_days: list[date]
    qa_days: list[date]
    return_days: list[date]
    spend_by_day: dict[date, float]


@dataclass
class RetentionRow:
    cohort: date
    source: str
    cohort_size: int
    values: list[float]              # % retained per period
    dollars: list[float] = field(default_factory=list)  # spend metric only


@dataclass
class RetentionResult:
    metric: str
    granularity: str
    periods: int
    rows: list[RetentionRow]


def cohort_key(registered_at: datetime, granularity: str) -> date:
    d = registered_at.date()
    if granularity == "day":
        return d
    if granularity == "week":
        return d - timedelta(days=d.weekday())  # Monday start
    if granularity == "month":
        return d.replace(day=1)
    raise ValueError(f"unknown granularity: {granularity}")


def period_index(registered_at: datetime, when: date, granularity: str) -> int:
    base = cohort_key(registered_at, granularity)
    if granularity == "day":
        return (when - base).days
    if granularity == "week":
        return (when - base).days // 7
    if granularity == "month":
        return (when.year - base.year) * 12 + (when.month - base.month)
    raise ValueError(f"unknown granularity: {granularity}")


def _days_for(user: UserActivity, metric: str) -> list[date]:
    return getattr(user, _METRIC_FIELD[metric])


def retention_matrix(
    users: list[UserActivity],
    *,
    metric: str,
    granularity: str,
    periods: int,
    by_source: bool = False,
) -> RetentionResult:
    # Группируем юзеров по (cohort[, source]).
    groups: dict[tuple[date, str], list[UserActivity]] = {}
    for u in users:
        ck = cohort_key(u.registered_at, granularity)
        src = (u.source or "(unknown)") if by_source else "(all)"
        groups.setdefault((ck, src), []).append(u)

    rows: list[RetentionRow] = []
    for (ck, src), members in sorted(groups.items()):
        size = len(members)
        counts = [0] * periods
        dollars = [0.0] * periods
        for u in members:
            if metric == "spend":
                hit = [False] * periods
                for day, amount in u.spend_by_day.items():
                    p = period_index(u.registered_at, day, granularity)
                    if 0 <= p < periods:
                        dollars[p] += amount
                        hit[p] = True
                for p in range(periods):
                    if hit[p]:
                        counts[p] += 1
            else:
                seen: set[int] = set()
                for day in _days_for(u, metric):
                    p = period_index(u.registered_at, day, granularity)
                    if 0 <= p < periods:
                        seen.add(p)
                for p in seen:
                    counts[p] += 1
        values = [round(100.0 * c / size, 1) if size else 0.0 for c in counts]
        rows.append(
            RetentionRow(
                cohort=ck,
                source=src,
                cohort_size=size,
                values=values,
                dollars=[round(d, 4) for d in dollars] if metric == "spend" else [],
            )
        )
    return RetentionResult(
        metric=metric, granularity=granularity, periods=periods, rows=rows
    )
