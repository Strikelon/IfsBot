from aiogram.types import InlineKeyboardMarkup, ReplyKeyboardRemove

from adapters.telegram.keyboards import (
    QA_CALLBACK_CUSTOM,
    QA_CALLBACK_FINISH,
    QA_CALLBACK_SKIP,
    dream_actions_keyboard,
    loop_add_keyboard,
    newdream_confirm_analysis_keyboard,
    qa_options_keyboard,
    remove_keyboard,
)
from adapters.telegram.texts import (
    CONFIRM_ANALYZE_BTN,
    DREAM_ACTION_START_ANALYSIS,
    LOOP_ADD_BTN_DONE,
    QA_CUSTOM_QUESTION_BTN,
    QA_FINISH_BTN,
    QA_SKIP_BTN,
    SAVE_WITHOUT_ANALYSIS_BTN,
)


def test_remove_keyboard_returns_remove_marker() -> None:
    assert isinstance(remove_keyboard(), ReplyKeyboardRemove)


def test_newdream_confirm_analysis_keyboard_is_inline_two_buttons() -> None:
    """Гейт «Разобрать сон?» — inline на сообщении (не reply-клава), с callback_data."""
    kb = newdream_confirm_analysis_keyboard()
    assert isinstance(kb, InlineKeyboardMarkup)
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.text for b in flat] == [CONFIRM_ANALYZE_BTN, SAVE_WITHOUT_ANALYSIS_BTN]
    assert [b.callback_data for b in flat] == ["confirm_analyze", "confirm_save"]


def test_dream_actions_keyboard_has_one_inline_button() -> None:
    kb = dream_actions_keyboard(dream_id=42)
    assert isinstance(kb, InlineKeyboardMarkup)
    rows = kb.inline_keyboard
    flat = [btn for row in rows for btn in row]
    assert len(flat) == 1
    assert flat[0].text == DREAM_ACTION_START_ANALYSIS
    assert flat[0].callback_data == "start_analysis:42"


def test_dream_actions_keyboard_respects_custom_text() -> None:
    kb = dream_actions_keyboard(dream_id=42, text="Custom")
    flat = [b for row in kb.inline_keyboard for b in row]
    assert flat[0].text == "Custom"
    assert flat[0].callback_data == "start_analysis:42"


def test_image_pick_keyboard_renders_one_button_per_image():
    from adapters.telegram.keyboards import image_pick_keyboard

    kb = image_pick_keyboard(
        images=[(1, "Тёмная фигура"), (2, "Туман"), (3, "Удушье")]
    )
    rows = kb.inline_keyboard
    assert len(rows) == 3
    texts = [row[0].text for row in rows]
    assert texts == ["Тёмная фигура", "Туман", "Удушье"]
    callbacks = [row[0].callback_data for row in rows]
    assert callbacks == ["pick_image:1", "pick_image:2", "pick_image:3"]


def test_image_pick_keyboard_truncates_long_names():
    """Telegram кнопки ограничены 64 байтами текста — обрезаем имена."""
    from adapters.telegram.keyboards import image_pick_keyboard

    long_name = "ОЧЕНЬ " * 30 + "длинное имя"
    kb = image_pick_keyboard(images=[(1, long_name)])
    btn_text = kb.inline_keyboard[0][0].text
    assert len(btn_text.encode("utf-8")) <= 64


def test_deep_qa_keyboard_has_pause_button():
    """В IN_QA-эквиваленте Deep (любой phase, ждём ответ юзера) — кнопка «Пауза»."""
    from adapters.telegram.keyboards import deep_qa_keyboard

    kb = deep_qa_keyboard(session_id=42)
    callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert "deep_pause:42" in callbacks


def test_deep_qa_keyboard_shows_pause_for_regular_session():
    """is_onboarding_session=False (default) → Pause button visible."""
    from adapters.telegram.keyboards import deep_qa_keyboard

    kb = deep_qa_keyboard(session_id=42, is_onboarding_session=False)
    assert isinstance(kb, InlineKeyboardMarkup)
    callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
    assert "deep_pause:42" in callbacks


def test_deep_qa_keyboard_hides_pause_for_onboarding_session():
    """is_onboarding_session=True → no keyboard (returns None per §6.13)."""
    from adapters.telegram.keyboards import deep_qa_keyboard

    kb = deep_qa_keyboard(session_id=42, is_onboarding_session=True)
    assert kb is None


def test_loop_add_keyboard_is_inline_single_done_button() -> None:
    """«Это всё» — inline-кнопка на самом prompt-сообщении (не reply-клава)."""
    kb = loop_add_keyboard()
    assert isinstance(kb, InlineKeyboardMarkup)
    rows = kb.inline_keyboard
    assert len(rows) == 1 and len(rows[0]) == 1
    assert rows[0][0].text == LOOP_ADD_BTN_DONE
    assert rows[0][0].callback_data == "loop_done"


def test_qa_options_keyboard_with_two_chips() -> None:
    kb = qa_options_keyboard(["Линия А", "Линия Б"])
    assert isinstance(kb, InlineKeyboardMarkup)
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.text for b in flat] == [
        "Линия А", "Линия Б", QA_CUSTOM_QUESTION_BTN, QA_FINISH_BTN,
    ]
    assert [b.callback_data for b in flat] == [
        "qa:0", "qa:1", QA_CALLBACK_CUSTOM, QA_CALLBACK_FINISH,
    ]


def test_qa_options_keyboard_no_chips_has_only_controls() -> None:
    kb = qa_options_keyboard([])
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.callback_data for b in flat] == [QA_CALLBACK_CUSTOM, QA_CALLBACK_FINISH]


def test_qa_options_keyboard_truncates_long_label() -> None:
    long_label = "очень длинный лейбл " * 10  # > 60 байт
    kb = qa_options_keyboard([long_label])
    chip = kb.inline_keyboard[0][0]
    assert chip.callback_data == "qa:0"
    assert len(chip.text.encode("utf-8")) <= 64


def test_qa_options_keyboard_clarify_skip_and_finish_only() -> None:
    kb = qa_options_keyboard([], kind="clarify")
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.text for b in flat] == [QA_SKIP_BTN, QA_FINISH_BTN]
    assert [b.callback_data for b in flat] == [QA_CALLBACK_SKIP, QA_CALLBACK_FINISH]


def test_clarify_skip_button_says_dont_want_to_answer() -> None:
    kb = qa_options_keyboard([], kind="clarify")
    skip_btn = kb.inline_keyboard[0][0]
    assert skip_btn.text == "Не хочу отвечать"


def test_qa_options_keyboard_clarify_ignores_labels() -> None:
    # даже если лейблы переданы, clarify не показывает чипов
    kb = qa_options_keyboard(["X", "Y"], kind="clarify")
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.callback_data for b in flat] == [QA_CALLBACK_SKIP, QA_CALLBACK_FINISH]


def test_qa_options_keyboard_choice_default_kind_unchanged() -> None:
    kb = qa_options_keyboard(["Линия А"])  # kind defaults to choice
    flat = [b for row in kb.inline_keyboard for b in row]
    assert [b.callback_data for b in flat] == ["qa:0", "qa:custom", "qa:finish"]
