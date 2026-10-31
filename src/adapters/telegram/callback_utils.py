"""Утилиты для callback-запросов Telegram.

`safe_answer` — безопасный ack callback-кнопки. Telegram-запрос callback живёт
~15 секунд; если бот на миг терял связь (таймаут long-polling через прокси) или
просто был занят, тап доезжает поздно и `answerCallbackQuery` падает с
`query is too old / query ID is invalid`. Это НЕ должно ронять хэндлер — ack
кнопки косметический (убрать «часики»), а реальную работу (ответ юзеру) надо
довести до конца. Поэтому глотаем безобидные ошибки ack и логируем.
"""
from __future__ import annotations

import structlog
from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError
from aiogram.types import CallbackQuery

log = structlog.get_logger(__name__)


async def safe_answer(callback: CallbackQuery, *args: object, **kwargs: object) -> None:
    """Ack callback-кнопку, терпимо к протухшему запросу и сетевым сбоям.

    Принимает те же аргументы, что `CallbackQuery.answer` (text, show_alert, ...).
    Никогда не бросает: при сбое ack логирует и возвращает управление, чтобы
    хэндлер продолжил выполнять полезную работу.
    """
    try:
        await callback.answer(*args, **kwargs)  # type: ignore[arg-type]
    except TelegramBadRequest as exc:
        # «query is too old / query ID is invalid» — запрос протух (>15с). Норма.
        log.info("callback.answer.stale", error=str(exc))
    except TelegramNetworkError as exc:
        log.warning("callback.answer.network", error=str(exc))
