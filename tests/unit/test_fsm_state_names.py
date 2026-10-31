"""Эталон строковых значений FSM-состояний (спека B7).

FSM хранится на проде СТРОКОЙ "Group:STATE" (fsm_storage.py). Разведка прода
2026-08-15: 15 юзеров в OnboardingStates:ASKING_DREAM, 8 в
QuickAnalysisStates:IN_QA, 1 в QuickAnalysisStates:ANSWERING_IMAGE_QUESTIONS.
Переименование любого из этих значений = непойманное сообщение = idle
catch-all = state.clear() = потеря прогресса живого юзера.

Тест падает при переименовании/удалении. ДОБАВЛЯТЬ новые состояния можно.
"""

from aiogram.fsm.state import State

from adapters.telegram import states as st

# name -> точная строка, как она лежит в prod fsm_state.state
_ETALON: dict[str, str] = {
    # OnboardingStates — 15 юзеров рекламной когорты стоят в ASKING_DREAM
    "OnboardingStates.ASKING_LIFE_CONTEXT": "OnboardingStates:ASKING_LIFE_CONTEXT",
    "OnboardingStates.LIFE_CONTEXT_LOOP": "OnboardingStates:LIFE_CONTEXT_LOOP",
    "OnboardingStates.ASKING_DREAM": "OnboardingStates:ASKING_DREAM",
    "OnboardingStates.DREAM_LOOP": "OnboardingStates:DREAM_LOOP",
    "OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB": "OnboardingStates:ASKING_PER_SESSION_CONTEXT_ONB",
    "OnboardingStates.GENERATING_INITIAL_ONB": "OnboardingStates:GENERATING_INITIAL_ONB",
    # QuickAnalysisStates — юзеры в IN_QA и в ANSWERING_IMAGE_QUESTIONS
    "QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT": "QuickAnalysisStates:ASKING_PER_SESSION_CONTEXT",
    "QuickAnalysisStates.GENERATING_INITIAL": "QuickAnalysisStates:GENERATING_INITIAL",
    "QuickAnalysisStates.IN_QA": "QuickAnalysisStates:IN_QA",
    "QuickAnalysisStates.GENERATING_FOLLOWUP": "QuickAnalysisStates:GENERATING_FOLLOWUP",
    "QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS": "QuickAnalysisStates:OFFERING_IMAGE_QUESTIONS",
    "QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS": "QuickAnalysisStates:ANSWERING_IMAGE_QUESTIONS",
    # NewDreamStates
    "NewDreamStates.ASKING_DREAM": "NewDreamStates:ASKING_DREAM",
    "NewDreamStates.DREAM_LOOP": "NewDreamStates:DREAM_LOOP",
    "NewDreamStates.CONFIRMING_ANALYSIS": "NewDreamStates:CONFIRMING_ANALYSIS",
    # DreamCardStates
    "DreamCardStates.DECIDING_ANALYZE": "DreamCardStates:DECIDING_ANALYZE",
    # ProfileEditStates
    "ProfileEditStates.AWAITING_LIFE_CONTEXT_MODE": "ProfileEditStates:AWAITING_LIFE_CONTEXT_MODE",
    "ProfileEditStates.EDITING_LIFE_CONTEXT": "ProfileEditStates:EDITING_LIFE_CONTEXT",
}


def _resolve(path: str) -> State:
    group_name, state_name = path.split(".")
    group = getattr(st, group_name)
    return getattr(group, state_name)


def test_all_prod_states_exist_with_exact_string_values():
    for path, expected in _ETALON.items():
        state = _resolve(path)  # AttributeError = состояние удалено/переименовано
        assert state.state == expected, (
            f"{path}: строковое значение изменилось ({state.state!r} != {expected!r}). "
            "Это потеряет прод-юзеров, висящих в этом состоянии (спека B7)."
        )


def test_etalon_covers_every_pre_entry_types_state():
    # Полнота эталона: если кто-то удалит состояние И строку из эталона разом,
    # первый тест это не поймает. Держим счётчик: до фичи было ровно 18 состояний.
    assert len(_ETALON) == 18
