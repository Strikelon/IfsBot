"""Тексты экрана выбора типа + универсальный «Сохранил.» + клавиатура (B3/B5/B6)."""

from adapters.telegram import texts
from adapters.telegram.keyboards import ENTRY_TYPE_CALLBACK_PREFIX, entry_type_keyboard
from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION


def test_entry_choice_intro_verbatim():
    assert texts.ENTRY_CHOICE_INTRO == (
        "Здравствуйте, {name}!\n\nВыберите, что вы хотите разобрать:"
    )
    assert texts.ENTRY_CHOICE_UNCLEAR == (
        "Не понял вас. Выберите, что вы хотите разобрать:"
    )


def test_ask_life_context_no_longer_lies_in_new_branches():
    # Спека B3 п.3: «нужна только для анализа ваших снов» → «нужна только для разбора».
    assert texts.ASK_LIFE_CONTEXT_ONB == (
        "Чтобы лучше понимать вас и то, что происходит в вашей жизни, расскажите "
        "немного о себе и своём жизненном контексте (возраст, семейная ситуация – "
        "дети, состав семьи и т.п., профессия, хобби, другие значимые сферы) – "
        "в нескольких фразах.\n\n"
        "Не волнуйтесь: всё, что вы рассказываете, строго конфиденциально. "
        "Эта информация нужна только для разбора.\n\n"
        "Отвечать можно текстом или голосом."
    )


# Правка 2026-08-24: «Чтобы вернуться к разобранным снам – /dreams» ×6 ушло в
# формат пункта меню («/dreams – сны»), новое встало перед прошлым — после
# «Сохранил.» вероятнее начать разбор, чем открыть архив.
_SAVED_BODY = (
    "Сохранил.\n\n"
    "Разобрать новое:\n"
    "/newdream – сон\n"
    "/new_external_situation – волнующая ситуация\n"
    "/new_internal_conflict – внутренний конфликт\n\n"
    "Вернуться к разобранному:\n"
    "/dreams – сны\n"
    "/external_situations – волнующие ситуации\n"
    "/internal_conflicts – внутренние конфликты"
)


def test_universal_saved_notice_verbatim_spec_b6():
    assert texts.SESSION_PAUSED_NOTICE == _SAVED_BODY
    # «Настройки» отбиты пустой строкой от блока команд разбора.
    assert texts.END_OF_ONBOARDING_TEXT == _SAVED_BODY + "\n\nНастройки – /profile"


def test_command_lists_dropped_the_word_chtoby():
    """В списках команд «Чтобы» не осталось."""
    from domain.entry_types import ENTRY_TYPES

    command_lists = [
        texts.SESSION_PAUSED_NOTICE,
        texts.END_OF_ONBOARDING_TEXT,
        texts.INTERVIEW_GRANTED_NOTICE,
        texts.IDLE_HINT,
        texts.WELCOME_BACK,
        texts.NO_DREAMS_YET,
        texts.DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE,
    ]
    for cfg in ENTRY_TYPES.values():
        command_lists += [cfg.list_empty, cfg.saved_without_analysis]
    for value in command_lists:
        assert "чтобы" not in value.lower(), value


def test_idle_hint_mentions_all_six_commands():
    for cmd in (
        "/newdream", "/dreams", "/new_external_situation", "/external_situations",
        "/new_internal_conflict", "/internal_conflicts", "/profile",
    ):
        assert cmd in texts.IDLE_HINT


def test_no_em_dash_in_new_texts():
    for value in (
        texts.ENTRY_CHOICE_INTRO, texts.ENTRY_CHOICE_UNCLEAR,
        texts.ASK_LIFE_CONTEXT_ONB, texts.SESSION_PAUSED_NOTICE,
        texts.END_OF_ONBOARDING_TEXT, texts.IDLE_HINT,
    ):
        assert "—" not in value and "―" not in value


def test_entry_type_keyboard_three_buttons_with_kind_callbacks():
    kb = entry_type_keyboard()
    rows = kb.inline_keyboard
    assert [b.text for row in rows for b in row] == [
        "Сон", "Волнующая ситуация", "Внутренний конфликт"
    ]
    assert [b.callback_data for row in rows for b in row] == [
        f"{ENTRY_TYPE_CALLBACK_PREFIX}{KIND_DREAM}",
        f"{ENTRY_TYPE_CALLBACK_PREFIX}{KIND_SITUATION}",
        f"{ENTRY_TYPE_CALLBACK_PREFIX}{KIND_CONFLICT}",
    ]
