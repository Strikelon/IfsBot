"""Расчёт воронки онбординга по когортам (чистые функции, без БД)."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from analytics.retention import cohort_key

# Порядок шагов воронки = порядок онбординга: сначала сон, затем знакомство.
FUNNEL_STEPS: list[str] = [
    "first_launch",
    "onb_dream_told",
    "onb_life_context",
    "onb_intent_given",
    "onb_img_q_offered",
    "onb_img_q_done",
    "onb_analysis_delivered",
    "onb_qa_answered",
    "onb_completed",
]


@dataclass
class UserFunnel:
    user_id: int
    registered_at: datetime
    source: str | None
    reached: set[str]


@dataclass
class FunnelRow:
    cohort: object
    source: str
    cohort_size: int
    step_counts: dict[str, int]
    step_pct: dict[str, float]


def funnel_by_cohort(
    users: list[UserFunnel], *, granularity: str, by_source: bool = False
) -> list[FunnelRow]:
    groups: dict[tuple[object, str], list[UserFunnel]] = {}
    for u in users:
        ck = cohort_key(u.registered_at, granularity)
        src = (u.source or "(unknown)") if by_source else "(all)"
        groups.setdefault((ck, src), []).append(u)

    rows: list[FunnelRow] = []
    for (ck, src), members in sorted(groups.items()):
        size = len(members)
        counts = {step: 0 for step in FUNNEL_STEPS}
        for u in members:
            for step in FUNNEL_STEPS:
                if step in u.reached:
                    counts[step] += 1
        pct = {
            step: (round(100.0 * counts[step] / size, 1) if size else 0.0)
            for step in FUNNEL_STEPS
        }
        rows.append(
            FunnelRow(
                cohort=ck, source=src, cohort_size=size,
                step_counts=counts, step_pct=pct,
            )
        )
    return rows
