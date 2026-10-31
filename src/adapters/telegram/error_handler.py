"""Глобальный обработчик необработанных исключений в хэндлерах.

Раньше любое исключение в хэндлере (например, Telegram 400 «message is too long»
при отправке слишком длинной карточки сна) всплывало и терялось — юзер видел
тишину. Теперь:
- логируем исключение с traceback (и tg_user_id через contextvars/из апдейта) —
  на проде по нику можно найти точную причину;
- шлём юзеру понятное сообщение вместо молчания.

Регистрируется на dispatcher: `dp.errors.register(handle_global_error)`.
"""

from __future__ import annotations

from typing import Any

import structlog

from adapters.telegram.texts import GENERIC_ERROR_NOTICE

log = structlog.get_logger(__name__)


def _extract_message_and_user(update: Any) -> tuple[Any, Any]:
    """Достать (message-для-ответа, tg_user) из message- или callback-апдейта."""
    message = getattr(update, "message", None)
    cq = getattr(update, "callback_query", None)
    if message is None and cq is not None:
        message = getattr(cq, "message", None)
    tg_user = getattr(getattr(update, "message", None), "from_user", None)
    if tg_user is None and cq is not None:
        tg_user = getattr(cq, "from_user", None)
    return message, tg_user


async def handle_global_error(event: Any) -> bool:
    """aiogram error-observer: логирует исключение + не оставляет юзера в тишине.

    Возвращает True — апдейт помечен обработанным (не падаем дальше).
    """
    update = getattr(event, "update", None)
    message, tg_user = _extract_message_and_user(update)
    log.error(
        "unhandled_handler_error",
        exc_info=getattr(event, "exception", None),
        update_id=getattr(update, "update_id", None),
        tg_user_id=getattr(tg_user, "id", None),
        tg_username=getattr(tg_user, "username", None),
    )
    if message is not None:
        try:
            await message.answer(GENERIC_ERROR_NOTICE)
        except Exception:  # noqa: BLE001 — уведомление не критично, лог уже есть
            log.info("unhandled_handler_error.notify_failed", exc_info=True)
    return True
