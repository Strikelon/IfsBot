"""FSM-состояния для aiogram-хэндлеров.

Phase 1: OnboardingStates.
Phase 2: QuickAnalysisStates — диалог быстрого разбора.
"""

from aiogram.fsm.state import State, StatesGroup


class OnboardingStates(StatesGroup):
    """Онбординг: заполнение профиля → Quick разбор → завершение.

    LIFE_CONTEXT_LOOP / DREAM_LOOP — любой не-DONE текст/голос auto-append'ит
    к собранному и возвращает юзера в тот же state с клавиатурой [Это всё].
    """

    ASKING_LIFE_CONTEXT = State()
    LIFE_CONTEXT_LOOP = State()        # ждём «Это всё» или append-реплику
    ASKING_DREAM = State()             # ждём рассказ сна
    DREAM_LOOP = State()               # ждём «Это всё» или append-реплику
    ASKING_PER_SESSION_CONTEXT_ONB = State()  # spec §9 — C.5
    GENERATING_INITIAL_ONB = State()   # между per-session и handoff (ветка убрана)
    CHOOSING_ENTRY_TYPE = State()      # экран выбора типа обращения (только новые юзеры)


class QuickAnalysisStates(StatesGroup):
    """FSM быстрого разбора. session_id и dream_id хранятся в state.data."""

    ASKING_PER_SESSION_CONTEXT = State()
    GENERATING_INITIAL = State()       # между per-session и IN_QA (ветка убрана)
    IN_QA = State()
    GENERATING_FOLLOWUP = State()      # внутри IN_QA на время LLM
    OFFERING_IMAGE_QUESTIONS = State()   # приглашение к вопросам по образам (offer-first)
    ANSWERING_IMAGE_QUESTIONS = State()  # юзер отвечает на вопрос items[idx]


class NewDreamStates(StatesGroup):
    """Команда /newdream — опытный юзер хочет разобрать новый сон.

    DREAM_LOOP — любой не-DONE текст/голос auto-append'ит к transcription и
    возвращает юзера в тот же state с клавиатурой [Это всё].
    После DONE показывается карточка сна + reply-клава
    [🔍 Разобрать сон] [💤 Сохранить без разбора] (CONFIRMING_ANALYSIS).
    Тап «Разобрать» → QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.
    Тап «Сохранить» → FSM clear, dream остаётся RECORDED.
    """

    ASKING_DREAM = State()
    DREAM_LOOP = State()
    CONFIRMING_ANALYSIS = State()  # между DREAM_LOOP "Это всё" и QuickAnalysisStates


class DreamCardStates(StatesGroup):
    """Открыта карточка сна без разбора (кнопка «Разобрать сейчас»).

    Юзер может тапнуть кнопку — или ответить текстом/голосом: дешёвый классификатор
    решает analyze/save/unclear (как на gate /newdream). analyze → запуск разбора.
    """

    DECIDING_ANALYZE = State()


class ProfileEditStates(StatesGroup):
    """Edit мини-FSM на /profile. Имя НЕ редактируется (read-only из Telegram)."""

    AWAITING_LIFE_CONTEXT_MODE = State()  # L6: показана клава «заменить/дописать»
    EDITING_LIFE_CONTEXT = State()


class EntryStates(StatesGroup):
    """Рассказ о волнующей ситуации / внутреннем конфликте (спека B3).

    Общая группа для онбординга и повторных обращений; тип лежит в FSM-data
    (FSM_KEY_ENTRY_KIND), dream_id — под onboarding_dream_id (как у сна).
    ENTRY_LOOP — любой не-DONE текст/голос auto-append'ит к transcription.
    """

    ASKING_ENTRY = State()
    ENTRY_LOOP = State()
