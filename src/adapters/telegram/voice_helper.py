"""Единый помощник для приёма голоса с UX-плейсхолдером.

Везде, где принимается голос: показать «Слушаю и расшифровываю…» отдельным
сообщением, скачать аудио, прогнать через Whisper, и после ответа Whisper
ОТРЕДАКТИРОВАТЬ это же сообщение — заменить на «Расшифровал ваше голосовое»
(или «Не удалось расшифровать, попробуй ещё раз» при сбое).

Юзеру это даёт явный сигнал: «бот услышал и закончил расшифровывать».

Usage-учёт (Whisper billing) делается caller'ом — у каждого потока (Deep,
Quick, onboarding, voice) свои поля session_id и user_id для UsageRecorder.
Помощник возвращает только текст + длительность.
"""

from __future__ import annotations

from io import BytesIO

import structlog
from aiogram import Bot
from aiogram.types import Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.texts import (
    VOICE_TRANSCRIBE_FAILED,
    VOICE_TRANSCRIBED,
    VOICE_TRANSCRIBING,
)
from llm.stt import TranscriptionResult, transcribe_voice

log = structlog.get_logger(__name__)


async def transcribe_with_placeholder(
    *,
    message: Message,
    bot: Bot,
    openai_client: AsyncOpenAI,
    log_prefix: str,
    session: AsyncSession | None = None,
    settings=None,
) -> TranscriptionResult | None:
    """Прислать плейсхолдер «Слушаю…», расшифровать, отредактировать плейсхолдер.

    Возвращает `TranscriptionResult` (с .text и .duration_seconds) при успехе,
    `None` при любой ошибке скачивания/Whisper. Сообщения об ошибке юзер уже
    увидел (в виде отредактированного плейсхолдера).

    Если переданы `session`+`settings` и пользователь превысил spend-лимит —
    единая точка блокировки ВСЕГО голоса: шлём пейвол и возвращаем `None` ДО
    плейсхолдера и Whisper (caller трактует `None` как «уже показано сообщение»).
    """
    if message.voice is None:
        return None

    # Spend cap: gate ALL voice transcription in one place (before placeholder + Whisper).
    if session is not None and settings is not None and message.from_user is not None:
        from adapters.telegram.spend_guard import is_over_limit, show_paywall  # noqa: PLC0415
        from data.repositories import UserRepository  # noqa: PLC0415

        user = await UserRepository(session).get_by_tg_id(message.from_user.id)
        if user is not None and await is_over_limit(session, user, settings):
            await show_paywall(message, session, user, settings)
            return None

    placeholder = await message.answer(VOICE_TRANSCRIBING)

    buf = BytesIO()
    try:
        await bot.download(message.voice, destination=buf)
    except Exception:  # noqa: BLE001
        log.exception(f"{log_prefix}.voice.download.failed")
        await _safe_edit(placeholder, VOICE_TRANSCRIBE_FAILED)
        return None

    try:
        result = await transcribe_voice(
            client=openai_client,
            audio_bytes=buf.getvalue(),
            filename=f"{message.voice.file_id}.ogg",
            language="ru",
        )
    except Exception:  # noqa: BLE001
        log.exception(f"{log_prefix}.voice.transcribe.failed")
        await _safe_edit(placeholder, VOICE_TRANSCRIBE_FAILED)
        return None

    await _safe_edit(placeholder, VOICE_TRANSCRIBED)
    return result


async def _safe_edit(msg: Message, text: str) -> None:
    """edit_text может упасть (сообщение удалили, слишком старое и т.п.) —
    это не должно ломать поток: расшифровка либо уже отдана, либо упала, и
    плейсхолдер — чисто UX-сигнал. Логируем на debug и идём дальше.
    """
    try:
        await msg.edit_text(text)
    except Exception:  # noqa: BLE001
        log.debug("voice.placeholder.edit_failed", exc_info=True)
