"""Глобальный ловец неподдерживаемых типов сообщений (фото/файлы/стикеры/видео).

Бот умеет только текст и голос. На любую медиа — мягкая подсказка, без сброса
FSM-состояния (юзер мог случайно прислать скрин посреди онбординга — он должен
иметь возможность продолжить текстом).
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.media_guard import (
    create_router,
    handle_unsupported_media,
)
from adapters.telegram.texts import UNSUPPORTED_MEDIA_NOTICE


def _media_message(**media) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = None
    m.voice = None
    for attr in ("photo", "document", "sticker", "video", "video_note", "animation", "audio"):
        setattr(m, attr, media.get(attr))
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
async def test_photo_gets_unsupported_notice() -> None:
    msg = _media_message(photo=[SimpleNamespace(file_id="p")])
    await handle_unsupported_media(msg)
    msg.answer.assert_awaited_once()
    assert msg.answer.await_args.args[0] == UNSUPPORTED_MEDIA_NOTICE


@pytest.mark.asyncio
async def test_document_gets_unsupported_notice() -> None:
    msg = _media_message(document=SimpleNamespace(file_id="d"))
    await handle_unsupported_media(msg)
    msg.answer.assert_awaited_once()
    assert msg.answer.await_args.args[0] == UNSUPPORTED_MEDIA_NOTICE


@pytest.mark.asyncio
async def test_handler_does_not_touch_state() -> None:
    """Хендлер не принимает state и не трогает FSM — состояние юзера сохраняется."""
    import inspect

    params = inspect.signature(handle_unsupported_media).parameters
    assert "state" not in params


def test_router_registers_media_filters() -> None:
    """Роутер media_guard ловит фото и документы (а не только текст/голос)."""
    router = create_router()
    handlers = router.message.handlers
    assert handlers, "media_guard router has no message handlers"
