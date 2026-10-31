"""Обёртка над OpenAI Whisper API.

Тонкий слой: получаем audio_bytes, отправляем в OpenAI, возвращаем
`TranscriptionResult`. Cost-учёт делает вызывающий через `UsageRecorder`.
Слой `llm` не знает про БД — это ответственность handlers.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI

log = structlog.get_logger(__name__)


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    duration_seconds: float


async def transcribe_voice(
    *,
    client: AsyncOpenAI,
    audio_bytes: bytes,
    filename: str = "voice.ogg",
    language: str = "ru",
) -> TranscriptionResult:
    """Распознать голос через Whisper-1.

    Args:
        client: настроенный AsyncOpenAI (с прокси или без).
        audio_bytes: содержимое голосового файла (Telegram отдаёт OGG/Opus).
        filename: имя файла (нужно OpenAI для определения формата).
        language: ISO-код языка ('ru' для русского, 'en' для английского).

    Returns:
        TranscriptionResult с распознанным текстом и длительностью в секундах.
        Если API не вернул duration — возвращаем 0.0 (защита от падения учёта).
    """
    log.info("stt.transcribe.start", bytes_len=len(audio_bytes), language=language)

    response = await client.audio.transcriptions.create(
        model="whisper-1",
        file=(filename, audio_bytes, "audio/ogg"),
        language=language,
        response_format="verbose_json",
    )

    text = response.text
    duration = getattr(response, "duration", None)
    duration_f = float(duration) if duration is not None else 0.0

    log.info(
        "stt.transcribe.done",
        text_len=len(text),
        duration_seconds=duration_f,
    )
    return TranscriptionResult(text=text, duration_seconds=duration_f)
