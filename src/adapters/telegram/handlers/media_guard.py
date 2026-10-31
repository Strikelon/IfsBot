"""Глобальный ловец неподдерживаемых типов сообщений.

Бот умеет только текст и голос. Если юзер прислал картинку, файл, стикер,
видео и т.п. (типичный кейс: скриншот сна вместо рассказа) — раньше бот молчал.
Теперь на любую медиа отвечаем мягкой подсказкой.

ВАЖНО: не трогаем FSM-state. Юзер мог случайно прислать скрин посреди
онбординга/рассказа сна — он должен иметь возможность сразу продолжить текстом
или голосом, не вылетев из текущего шага. Inline-кнопки (loop «Это всё», гейт)
тоже остаются на месте.

Регистрируется ПЕРЕД idle: фильтры медиа и idle (text/voice) не пересекаются.
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.types import Message

from adapters.telegram.texts import UNSUPPORTED_MEDIA_NOTICE

log = structlog.get_logger(__name__)

# Типы вложений, которые бот не обрабатывает (всё, кроме text и voice).
_MEDIA_FILTER = (
    F.photo
    | F.document
    | F.sticker
    | F.video
    | F.video_note
    | F.animation
    | F.audio
)


async def handle_unsupported_media(message: Message) -> None:
    """Любое медиа-вложение → подсказка «умею только текст/голос». State не трогаем."""
    if message.from_user is None:
        return
    await message.answer(UNSUPPORTED_MEDIA_NOTICE)
    log.info("media_guard.unsupported_media", telegram_user_id=message.from_user.id)


def create_router() -> Router:
    router = Router(name="media_guard")
    router.message.register(handle_unsupported_media, _MEDIA_FILTER)
    return router
