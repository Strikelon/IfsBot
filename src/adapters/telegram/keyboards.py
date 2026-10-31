"""Клавиатуры для UI. Не зависят от данных пользователя — чистые билдеры."""

from aiogram.types import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ReplyKeyboardRemove,
)

from adapters.telegram.texts import (
    CONFIRM_ANALYZE_BTN,
    DEEP_PAUSE_BTN,
    DREAM_ACTION_START_ANALYSIS,
    DREAM_DELETE_BTN_CANCEL,
    DREAM_DELETE_BTN_FINAL,
    DREAM_DELETE_BTN_YES,
    LOOP_ADD_BTN_DONE,
    PROFILE_EDIT_BTN_LIFE_CONTEXT,
    PROFILE_LIFE_BTN_APPEND,
    PROFILE_LIFE_BTN_REPLACE,
    QA_CUSTOM_QUESTION_BTN,
    QA_FINISH_BTN,
    QA_SKIP_BTN,
    SAVE_WITHOUT_ANALYSIS_BTN,
)

# callback_data inline-кнопки «Это всё». Хендлеры loop'ов фильтруют по нему.
LOOP_DONE_CALLBACK = "loop_done"


def loop_add_keyboard() -> InlineKeyboardMarkup:
    """Inline-кнопка «Это всё» на самом prompt-сообщении.

    Используется в онбординге и /newdream после каждой реплики юзера в LOOP-состоянии.
    Кнопка крепится к самому loop-prompt сообщению (а не к низу экрана) и гасится при
    тапе или при следующем сообщении юзера (см. handlers/_loop_prompt.py) — так
    устаревшую кнопку нельзя нажать повторно. Любой текст auto-append'ит контент.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=LOOP_ADD_BTN_DONE, callback_data=LOOP_DONE_CALLBACK)]
        ]
    )


# callback_data inline-кнопок гейта подтверждения разбора.
CONFIRM_ANALYZE_CALLBACK = "confirm_analyze"
CONFIRM_SAVE_CALLBACK = "confirm_save"


def newdream_confirm_analysis_keyboard() -> InlineKeyboardMarkup:
    """Inline-клавиатура гейта подтверждения (CONFIRMING_ANALYSIS).

    Крепится к сообщению «Разобрать этот сон сейчас?» и гасится при тапе или
    при ответе юзера текстом/голосом (см. handlers/newdream.py) — повторный тап
    устаревшей кнопки невозможен.
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=CONFIRM_ANALYZE_BTN, callback_data=CONFIRM_ANALYZE_CALLBACK)],
            [InlineKeyboardButton(
                text=SAVE_WITHOUT_ANALYSIS_BTN, callback_data=CONFIRM_SAVE_CALLBACK
            )],
        ]
    )


def remove_keyboard() -> ReplyKeyboardRemove:
    """Убрать reply-клавиатуру (после онбординга, перед обычным режимом)."""
    return ReplyKeyboardRemove()


def dream_list_item_inline_keyboard(dream_id: int) -> InlineKeyboardMarkup:
    """Кнопки под каждым сном в /dreams списке. Delete callbacks — Task 10."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text="↩ Вернуться", callback_data=f"dream_open:{dream_id}"),
            InlineKeyboardButton(text="🗑 Удалить", callback_data=f"dream_delete:{dream_id}"),
        ]]
    )


def dream_delete_confirm_1_keyboard(dream_id: int) -> InlineKeyboardMarkup:
    """Первое подтверждение удаления сна: [Да, удалить | Отмена]."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=DREAM_DELETE_BTN_YES,
                                 callback_data=f"dream_delete_confirm:{dream_id}"),
            InlineKeyboardButton(text=DREAM_DELETE_BTN_CANCEL,
                                 callback_data=f"dream_delete_cancel:{dream_id}"),
        ]]
    )


def dream_delete_confirm_2_keyboard(dream_id: int) -> InlineKeyboardMarkup:
    """Второе подтверждение удаления сна: [Да, удалить навсегда | Отмена]."""
    return InlineKeyboardMarkup(
        inline_keyboard=[[
            InlineKeyboardButton(text=DREAM_DELETE_BTN_FINAL,
                                 callback_data=f"dream_delete_final:{dream_id}"),
            InlineKeyboardButton(text=DREAM_DELETE_BTN_CANCEL,
                                 callback_data=f"dream_delete_cancel:{dream_id}"),
        ]]
    )


def dream_actions_keyboard(
    *, dream_id: int, text: str = DREAM_ACTION_START_ANALYSIS, callback_action: str = "start_analysis"
) -> InlineKeyboardMarkup:
    """Inline-клавиатура на карточке сна с одной кнопкой действия.

    Args:
        dream_id: ID сна.
        text: Текст кнопки.
        callback_action: Префикс callback_data — «start_analysis» или «continue_quick».
            RECORDED → «start_analysis» (создаёт новый Quick).
            PAUSED / COMPLETED → «continue_quick» (возобновляет существующий Quick).
    """
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=text, callback_data=f"{callback_action}:{dream_id}")]
        ]
    )


# --- Phase 3 ---

_MAX_BUTTON_BYTES = 60  # запас от Telegram-лимита 64


def _truncate_button(text: str) -> str:
    encoded = text.encode("utf-8")
    if len(encoded) <= _MAX_BUTTON_BYTES:
        return text
    # обрезаем до байтов с запасом, потом аккуратно к границе UTF-8
    truncated = encoded[: _MAX_BUTTON_BYTES - 3]
    while True:
        try:
            return truncated.decode("utf-8") + "..."
        except UnicodeDecodeError:
            truncated = truncated[:-1]


QA_CALLBACK_CUSTOM = "qa:custom"
QA_CALLBACK_FINISH = "qa:finish"
QA_CALLBACK_SKIP = "qa:skip"


def qa_options_keyboard(labels: list[str], *, kind: str = "choice") -> InlineKeyboardMarkup:
    """Inline-клава под репликой разбора, состав зависит от kind.

    - clarify → [⏭ Пропустить] (qa:skip) + [✅ Завершить] (qa:finish): открытый вопрос
      сам приглашает писать, чипов и «Свой вопрос» нет.
    - choice/confirm → чипы-варианты (callback_data=индекс qa:N) + «Свой вопрос» + «Завершить».
    labels игнорируются при kind="clarify".
    """
    if kind == "clarify":
        return InlineKeyboardMarkup(inline_keyboard=[
            [InlineKeyboardButton(text=QA_SKIP_BTN, callback_data=QA_CALLBACK_SKIP)],
            [InlineKeyboardButton(text=QA_FINISH_BTN, callback_data=QA_CALLBACK_FINISH)],
        ])
    rows: list[list[InlineKeyboardButton]] = [
        [InlineKeyboardButton(text=_truncate_button(label), callback_data=f"qa:{i}")]
        for i, label in enumerate(labels)
    ]
    rows.append(
        [InlineKeyboardButton(text=QA_CUSTOM_QUESTION_BTN, callback_data=QA_CALLBACK_CUSTOM)]
    )
    rows.append(
        [InlineKeyboardButton(text=QA_FINISH_BTN, callback_data=QA_CALLBACK_FINISH)]
    )
    return InlineKeyboardMarkup(inline_keyboard=rows)


# callback_data фичи «вопросы по образам сна».
IMGQ_CALLBACK_YES = "imgq:yes"
IMGQ_CALLBACK_NO = "imgq:no"
IMGQ_CALLBACK_SKIP = "imgq:skip"
IMGQ_CALLBACK_SKIP_ALL = "imgq:skip_all"


def image_offer_keyboard() -> InlineKeyboardMarkup:
    """Приглашение к вопросам по образам: [Я отвечу] / [Не хочу отвечать]."""
    from adapters.telegram.texts import IMG_Q_ANSWER_BTN

    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=IMG_Q_ANSWER_BTN, callback_data=IMGQ_CALLBACK_YES)],
        [InlineKeyboardButton(text=QA_SKIP_BTN, callback_data=IMGQ_CALLBACK_NO)],
    ])


def image_question_keyboard(*, total: int) -> InlineKeyboardMarkup:
    """Клава вопроса по образу: [Не хочу отвечать] всегда; [Пропустить все
    вопросы] — ТОЛЬКО при total >= 2 (при одном образе скип одного = скип всех,
    вторая кнопка избыточна — решение спеки §6.2)."""
    from adapters.telegram.texts import IMG_Q_SKIP_ALL_BTN

    rows = [[InlineKeyboardButton(text=QA_SKIP_BTN, callback_data=IMGQ_CALLBACK_SKIP)]]
    if total >= 2:
        rows.append(
            [InlineKeyboardButton(text=IMG_Q_SKIP_ALL_BTN, callback_data=IMGQ_CALLBACK_SKIP_ALL)]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def image_pick_keyboard(
    *,
    images: list[tuple[int, str]],
    include_none: bool = False,
) -> InlineKeyboardMarkup:
    """Список найденных P2_FIND образов как inline-кнопки.

    Args:
        images: список (dream_image_id, name).
        include_none: добавить кнопку «Никто из них» (callback_data='pick_none').
            Используется после SELF_GATE.no_response / p2_focus_router.no_response,
            когда юзер уже отказался от одной части и стоит дать выход из сессии.
    """
    from adapters.telegram.texts import DEEP_PICK_NONE_BTN

    rows: list[list[InlineKeyboardButton]] = [
        [
            InlineKeyboardButton(
                text=_truncate_button(name),
                callback_data=f"pick_image:{img_id}",
            )
        ]
        for img_id, name in images
    ]
    if include_none:
        rows.append(
            [InlineKeyboardButton(text=DEEP_PICK_NONE_BTN, callback_data="pick_none")]
        )
    return InlineKeyboardMarkup(inline_keyboard=rows)


def profile_edit_keyboard() -> InlineKeyboardMarkup:
    """Inline-кнопки на карточке /profile — БЕЗ кнопки "Изменить имя"
    (имя read-only, sync с Telegram first_name)."""
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text=PROFILE_EDIT_BTN_LIFE_CONTEXT, callback_data="profile:edit:life_context")],
    ])


# callback_data выбора режима правки life_context (L6): заменить целиком / дописать.
LIFE_CONTEXT_REPLACE_CALLBACK = "profile:life:replace"
LIFE_CONTEXT_APPEND_CALLBACK = "profile:life:append"


def life_context_edit_mode_keyboard() -> InlineKeyboardMarkup:
    """Выбор режима правки life_context: заменить целиком или дописать.

    Защита от L6: ручная правка не должна молча сносить факты,
    накопленные живой памятью.
    """
    return InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(
            text=PROFILE_LIFE_BTN_REPLACE, callback_data=LIFE_CONTEXT_REPLACE_CALLBACK
        )],
        [InlineKeyboardButton(
            text=PROFILE_LIFE_BTN_APPEND, callback_data=LIFE_CONTEXT_APPEND_CALLBACK
        )],
    ])


def deep_qa_keyboard(
    *, session_id: int, is_onboarding_session: bool = False
) -> InlineKeyboardMarkup | None:
    """Кнопка «Пауза» под каждой Deep-репликой, ждущей ответа юзера.

    Для онбординг-сессий клавиатура НЕ показывается (Pause скрыт по решению §6.13).
    Возвращает None — aiogram трактует это как «не прикреплять reply_markup».
    """
    if is_onboarding_session:
        return None
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text=DEEP_PAUSE_BTN, callback_data=f"deep_pause:{session_id}")]
        ]
    )


# callback_data экрана выбора типа обращения: "entry_type:<kind>".
ENTRY_TYPE_CALLBACK_PREFIX = "entry_type:"


def entry_type_keyboard() -> InlineKeyboardMarkup:
    """Три кнопки первого экрана: Сон / Волнующая ситуация / Внутренний конфликт.

    Лейблы берутся из реестра типов (domain.entry_types) — единственный источник.
    """
    from domain.entry_types import ALL_KINDS, ENTRY_TYPES

    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=ENTRY_TYPES[kind].label,
                    callback_data=f"{ENTRY_TYPE_CALLBACK_PREFIX}{kind}",
                )
            ]
            for kind in ALL_KINDS
        ]
    )
