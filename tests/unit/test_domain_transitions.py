"""Тесты domain/transitions.py — чистый модуль логики переходов фаз."""

from __future__ import annotations

import pytest

from data.models import ImageShape, P4Substep, SessionPhase
from domain.states import SessionStateData, UnblendingAttempt
from domain.transitions import (
    advance_p4_substep,
    advance_phase,
    apply_pivot,
    has_exhausted_pivots,
    record_unblending,
    should_pivot,
)


def _make_state(**overrides) -> SessionStateData:
    base = dict(
        target_part_name="X",
        target_part_shape=ImageShape.OBJECT,
        target_part_ifs_category="unclear",
    )
    base.update(overrides)
    return SessionStateData(**base)


# --- advance_phase: линейный путь P2_FOCUS → P3_FEEL → SELF_GATE → P4 → P5 → P6 → completed ---


@pytest.mark.parametrize(
    "current,expected",
    [
        (SessionPhase.P2_FOCUS, SessionPhase.P3_FEEL),
        (SessionPhase.P3_FEEL, SessionPhase.SELF_GATE),
        # SELF_GATE -> P4 при Self; -> UNBLENDING при Blended (отдельная функция)
        (SessionPhase.P4, SessionPhase.P5),
        (SessionPhase.P5, SessionPhase.P6),
        # P6 — конечная фаза, advance возвращает None (= completed)
        (SessionPhase.P6, None),
    ],
)
def test_advance_phase_linear(current, expected):
    assert advance_phase(current) == expected


def test_advance_phase_unblending_returns_none():
    """UNBLENDING не имеет одного 'next' — решает SELF_GATE classifier."""
    assert advance_phase(SessionPhase.UNBLENDING) is None


def test_advance_phase_self_gate_returns_none():
    """SELF_GATE решает classifier — не linear advance."""
    assert advance_phase(SessionPhase.SELF_GATE) is None


# --- advance_p4_substep: embodiment → origin → role → attitude → None ---


@pytest.mark.parametrize(
    "current,expected",
    [
        (P4Substep.EMBODIMENT, P4Substep.ORIGIN),
        (P4Substep.ORIGIN, P4Substep.ROLE),
        (P4Substep.ROLE, P4Substep.ATTITUDE),
        (P4Substep.ATTITUDE, None),  # P4 завершён, переходим в P5
    ],
)
def test_advance_p4_substep(current, expected):
    assert advance_p4_substep(current) == expected


# --- should_pivot: 3 подряд blended attempt'а ---


def test_should_pivot_false_when_no_history():
    state = _make_state()
    assert should_pivot(state) is False


def test_should_pivot_false_when_last_was_released():
    state = _make_state()
    state.unblending_history = [
        UnblendingAttempt(emotion="страх", outcome="blended"),
        UnblendingAttempt(emotion="злость", outcome="blended"),
        UnblendingAttempt(emotion="печаль", outcome="released"),
    ]
    assert should_pivot(state) is False


def test_should_pivot_true_after_three_consecutive_blended():
    state = _make_state()
    state.unblending_history = [
        UnblendingAttempt(emotion="страх", outcome="blended"),
        UnblendingAttempt(emotion="злость", outcome="blended"),
        UnblendingAttempt(emotion="безразличие", outcome="blended"),
    ]
    assert should_pivot(state) is True


def test_should_pivot_true_only_for_LAST_three():
    """Если ранее был released, считаем только хвост."""
    state = _make_state()
    state.unblending_history = [
        UnblendingAttempt(emotion="страх", outcome="blended"),
        UnblendingAttempt(emotion="злость", outcome="released"),  # сброс счётчика
        UnblendingAttempt(emotion="безразличие", outcome="blended"),
        UnblendingAttempt(emotion="печаль", outcome="blended"),
    ]
    # 2 подряд blended после released — пивота нет
    assert should_pivot(state) is False
    state.unblending_history.append(
        UnblendingAttempt(emotion="страх2", outcome="blended")
    )
    # теперь 3 подряд
    assert should_pivot(state) is True


# --- has_exhausted_pivots: 3-pivot лимит на сессию ---


@pytest.mark.parametrize(
    "count,expected",
    [(0, False), (1, False), (2, False), (3, True), (4, True)],
)
def test_has_exhausted_pivots(count, expected):
    assert has_exhausted_pivots(count) == expected


# --- record_unblending: добавляет attempt ---


def test_record_unblending_appends_attempt():
    state = _make_state()
    record_unblending(state, emotion="страх", outcome="released")
    assert len(state.unblending_history) == 1
    assert state.unblending_history[0].emotion == "страх"


# --- apply_pivot: меняет target_part, очищает promised поля, добавляет в pivot_history ---


def test_apply_pivot_swaps_target_part_and_resets():
    state = _make_state(
        target_part_name="Тёмная фигура",
        target_part_shape=ImageShape.HUMAN_LIKE,
        somatic_location="давит в груди",
        part_role="охраняет",
    )
    state.unblending_history = [
        UnblendingAttempt(emotion="злость", outcome="blended"),
        UnblendingAttempt(emotion="злость", outcome="blended"),
        UnblendingAttempt(emotion="злость", outcome="blended"),
    ]

    apply_pivot(state, new_part_name="Злость", new_shape=ImageShape.SENSATION)

    assert state.target_part_name == "Злость"
    assert state.target_part_shape == ImageShape.SENSATION
    assert state.target_part_ifs_category == "unclear"  # default для pivot'нутой части
    # промежуточные знания должны сброситься — новая часть, новые ответы
    assert state.somatic_location is None
    assert state.part_role is None
    assert state.pivot_history[-1] == "Тёмная фигура → Злость"
    # unblending_history очищается — иначе новая часть сразу триггернёт ещё pivot
    assert state.unblending_history == []
