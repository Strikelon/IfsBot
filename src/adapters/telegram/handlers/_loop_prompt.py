"""Loop-add prompt: показ inline-кнопки «Это всё» и её гашение.

Кнопка «Это всё» — inline на самом prompt-сообщении.
Чтобы юзер не мог тапнуть устаревшую кнопку (после того как уже прислал
ещё реплику или уже тапнул), мы:
- запоминаем `message_id` последнего prompt'а в FSM (`loop_prompt_msg_id`);
- гасим его markup при входящем сообщении в loop и при тапе кнопки.

UI-специфика (клавиатуры, тексты) намеренно живёт здесь, а не в `_common.py`,
который оторван от UI.
"""

from __future__ import annotations

import structlog
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from adapters.telegram.keyboards import loop_add_keyboard

log = structlog.get_logger(__name__)

_FSM_KEY_LOOP_PROMPT_MSG_ID = "loop_prompt_msg_id"


async def send_loop_prompt(*, message: Message, state: FSMContext, text: str) -> None:
    """Показать loop-prompt (`text`) с inline-кнопкой «Это всё» и запомнить его message_id.

    `text` различается по контексту (рассказ о себе / рассказ сна).
    message_id нужен, чтобы погасить кнопку, когда юзер пришлёт следующую
    реплику или тапнет кнопку (см. `extinguish_loop_button`).
    """
    sent = await message.answer(text, reply_markup=loop_add_keyboard())
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(**{_FSM_KEY_LOOP_PROMPT_MSG_ID: msg_id})


async def extinguish_inline_kb(
    *, state: FSMContext, bot: Bot | None, chat_id: int | None, fsm_key: str
) -> None:
    """Снять inline-клавиатуру с сообщения, чей message_id хранится в FSM под `fsm_key`.

    Универсальный хелпер для self-extinguishing inline-кнопок (loop-prompt,
    confirmation-gate и т.п.): гасим, чтобы устаревшую кнопку нельзя было нажать
    повторно. Ошибки редактирования глотаем (сообщение могло быть уже
    отредактировано/удалено).
    """
    data = await state.get_data()
    msg_id = data.get(fsm_key)
    if msg_id is None or bot is None or chat_id is None:
        return
    try:
        await bot.edit_message_reply_markup(
            chat_id=chat_id, message_id=msg_id, reply_markup=None
        )
    except Exception:  # noqa: BLE001
        log.debug("inline_kb.extinguish.failed", fsm_key=fsm_key, exc_info=True)
    await state.update_data(**{fsm_key: None})


async def extinguish_loop_button(
    *, state: FSMContext, bot: Bot | None, chat_id: int | None
) -> None:
    """Погасить inline-кнопку «Это всё» предыдущего loop-prompt'а (если известна)."""
    await extinguish_inline_kb(
        state=state, bot=bot, chat_id=chat_id, fsm_key=_FSM_KEY_LOOP_PROMPT_MSG_ID
    )
