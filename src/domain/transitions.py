"""Чистые функции переходов IFS state machine.

Не знают про БД и LLM. Принимают и возвращают immutable значения (SessionPhase,
P4Substep) или мутируют переданный SessionStateData in-place — это удобнее для
handler'а: построил state, прогнал через transition, записал обратно в БД.

См. spec §5 для схемы переходов.
"""

from __future__ import annotations

from data.models import ImageShape, P4Substep, SessionPhase
from domain.states import SessionStateData, UnblendingAttempt, UnblendingOutcome

# --- Линейные переходы фаз ---

_LINEAR_NEXT: dict[SessionPhase, SessionPhase | None] = {
    SessionPhase.P2_FOCUS: SessionPhase.P3_FEEL,
    SessionPhase.P3_FEEL: SessionPhase.SELF_GATE,
    # SELF_GATE → P4 или UNBLENDING — решает classifier, не linear
    SessionPhase.SELF_GATE: None,
    # UNBLENDING → SELF_GATE (повторный gate) или pivot — решает handler
    SessionPhase.UNBLENDING: None,
    SessionPhase.P4: SessionPhase.P5,
    SessionPhase.P5: SessionPhase.P6,
    SessionPhase.P6: None,  # completed
}


def advance_phase(current: SessionPhase) -> SessionPhase | None:
    """Следующая линейная фаза или None (если ветвление или completed).

    None означает: либо это conditional phase (SELF_GATE/UNBLENDING — handler решает),
    либо это терминальная P6 (completed).
    """
    return _LINEAR_NEXT[current]


# --- P4 sub-step линейный путь ---

_P4_NEXT: dict[P4Substep, P4Substep | None] = {
    P4Substep.EMBODIMENT: P4Substep.ORIGIN,
    P4Substep.ORIGIN: P4Substep.ROLE,
    P4Substep.ROLE: P4Substep.ATTITUDE,
    P4Substep.ATTITUDE: None,  # P4 завершён, handler переходит в P5
}


def advance_p4_substep(current: P4Substep) -> P4Substep | None:
    """Следующая P4-подфаза или None (P4 завершён)."""
    return _P4_NEXT[current]


# --- Pivot logic ---

_PIVOT_THRESHOLD = 3
_MAX_PIVOTS_PER_SESSION = 3


def should_pivot(state: SessionStateData) -> bool:
    """3 последних подряд unblending попытки — все 'blended' → нужен pivot.

    «Подряд» считаем от хвоста списка. Любой 'released' в последних трёх → нет.
    """
    tail = state.unblending_history[-_PIVOT_THRESHOLD:]
    if len(tail) < _PIVOT_THRESHOLD:
        return False
    return all(att.outcome == "blended" for att in tail)


def has_exhausted_pivots(pivot_count: int) -> bool:
    """Сессия исчерпала лимит pivot'ов и должна быть закрыта мягким сообщением."""
    return pivot_count >= _MAX_PIVOTS_PER_SESSION


def record_unblending(
    state: SessionStateData, *, emotion: str, outcome: UnblendingOutcome
) -> None:
    """Добавить одну попытку unblending'а в историю (мутирует state)."""
    state.unblending_history.append(UnblendingAttempt(emotion=emotion, outcome=outcome))


def apply_pivot(
    state: SessionStateData,
    *,
    new_part_name: str,
    new_shape: ImageShape,
    new_ifs_category: str = "unclear",
) -> None:
    """Перепривязать сессию к новой target_part.

    Старое имя сохраняется в pivot_history. Все промежуточные знания (somatic,
    part_role, part_origin_context, part_voice_form, part_fears) сбрасываются —
    новая часть, новые ответы. unblending_history тоже очищаем, иначе новая
    часть сразу триггернёт ещё один pivot.
    """
    state.pivot_history.append(f"{state.target_part_name} → {new_part_name}")
    state.target_part_name = new_part_name
    state.target_part_shape = new_shape
    state.target_part_ifs_category = new_ifs_category
    state.somatic_location = None
    state.part_voice_form = None
    state.part_origin_context = None
    state.part_role = None
    state.part_fears = None
    state.unblending_history = []
