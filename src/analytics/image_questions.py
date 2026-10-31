"""Метрики фичи «вопросы по образам» (чистые функции, без БД).

Append-события дедупятся на чтении по metadata.session_id (философия
analytics_events: write — append-only, дедуп per-session — read-side).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ImgQEvent:
    user_id: int
    event_type: str
    metadata: dict


def _sid(e: ImgQEvent):
    return e.metadata.get("session_id")


def split_by_entry_type(events: list[ImgQEvent]) -> dict[str, list[ImgQEvent]]:
    """Разложить события по типу обращения (metadata.entry_type).

    Имена событий одни на все типы (append-only, на них завязан отчёт с июля),
    поэтому разрез идёт по метаданным. События БЕЗ `entry_type` — вся история до
    этой фичи — считаются снами: тогда других типов просто не существовало.
    Незнакомое значение тоже сводим к сну (fail-safe, как entry_config).
    """
    from domain.entry_types import ALL_KINDS, KIND_DREAM

    buckets: dict[str, list[ImgQEvent]] = {}
    for e in events:
        kind = e.metadata.get("entry_type")
        if kind not in ALL_KINDS:
            kind = KIND_DREAM
        buckets.setdefault(kind, []).append(e)
    return buckets


def offer_choice_split(events: list[ImgQEvent]) -> dict:
    offered: dict = {}
    choices: dict = {}
    for e in events:
        if e.event_type == "img_q_offered" and _sid(e) is not None:
            offered.setdefault(_sid(e), bool(e.metadata.get("is_onboarding")))
        elif e.event_type == "img_q_choice" and _sid(e) is not None:
            choices.setdefault(_sid(e), e.metadata)  # первый выбор сессии
    by_onb: dict = {True: {"offered": 0, "answer": 0, "decline": 0},
                    False: {"offered": 0, "answer": 0, "decline": 0}}
    by_via: dict = {}
    answer = decline = 0
    for sid, is_onb in offered.items():
        by_onb[is_onb]["offered"] += 1
        meta = choices.get(sid)
        if meta is None:
            continue
        choice = meta.get("choice")
        if choice in ("answer", "decline"):
            by_onb[is_onb][choice] += 1
            if choice == "answer":
                answer += 1
            else:
                decline += 1
            via = meta.get("via", "?")
            by_via[via] = by_via.get(via, 0) + 1
    return {
        "offered": len(offered), "answer": answer, "decline": decline,
        "no_choice": len(offered) - answer - decline,
        "by_onboarding": by_onb, "by_via": by_via,
    }


def answered_histogram(events: list[ImgQEvent]) -> dict[tuple[int, int], int]:
    seen: set = set()
    hist: dict[tuple[int, int], int] = {}
    for e in events:
        if e.event_type != "img_q_flow_done" or _sid(e) in seen or _sid(e) is None:
            continue
        seen.add(_sid(e))
        key = (int(e.metadata.get("answered", 0)), int(e.metadata.get("total", 0)))
        hist[key] = hist.get(key, 0) + 1
    return hist


def drop_by_question(events: list[ImgQEvent]) -> dict[int, int]:
    """Сессии с choice=answer БЕЗ flow_done → номер вопроса (1-based), на котором
    бросили: max(idx)+2 по answered/skipped (следующий не взятый), без событий — 1."""
    started = {_sid(e) for e in events
               if e.event_type == "img_q_choice" and e.metadata.get("choice") == "answer"}
    finished = {_sid(e) for e in events if e.event_type == "img_q_flow_done"}
    max_idx: dict = {}
    for e in events:
        if e.event_type in ("img_q_answered", "img_q_skipped") and _sid(e) in started:
            idx = int(e.metadata.get("idx", -1))
            max_idx[_sid(e)] = max(max_idx.get(_sid(e), -1), idx)
    out: dict[int, int] = {}
    for sid in started - finished:
        if sid is None:
            continue
        q_num = max_idx.get(sid, -1) + 2  # -1 (нет событий) → 1
        out[q_num] = out.get(q_num, 0) + 1
    return out
