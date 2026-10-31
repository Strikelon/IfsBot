"""Безопасная отправка длинных сообщений в Telegram.

Telegram режет сообщения длиннее 4096 символов. LLM-ответы (особенно Quick.v2 с
IFS-интерпретацией) регулярно вылезают за этот лимит. Эти утилиты разбивают
текст на безопасные чанки и отправляют их последовательно, прикрепляя
reply_markup только к последнему чанку.

HTML-safety: бот создан с `parse_mode=ParseMode.HTML` (см. app/bot.py), поэтому
любой `<` или `&` в LLM-ответе Telegram попытается распарсить как HTML и
вернёт 400 "can't parse entities". `answer_long` по умолчанию прогоняет текст
через `html.escape`, чтобы такие фразы (например, «связь <X> с <Y>») отправились
корректно. Если вызывающий точно знает, что текст — валидный HTML с
намеренными тегами, можно передать `escape_html=False`.
"""

from __future__ import annotations

from html import escape as html_escape
from typing import Any

from aiogram.types import Message

TELEGRAM_MESSAGE_LIMIT = 4096
# Запас под форматирование/перевод строк, добавляемый кодом отправки.
_SAFE_CHUNK_LIMIT = 4000


def split_for_telegram(text: str, *, limit: int = _SAFE_CHUNK_LIMIT) -> list[str]:
    """Разбить text на чанки длиной ≤ limit.

    Приоритет границ: \\n\\n (абзац) → \\n (строка) → пробел → hard cut.
    Граница ищется в правой половине окна — чтобы не выпускать слишком короткие
    чанки. Сохраняем разделители абзацев/строк только при сплите по ним.

    Args:
        text: исходный текст. Может быть пустой строкой.
        limit: целевой лимит чанка. Default 4000 — запас от Telegram-лимита 4096.

    Returns:
        Непустой список строк, каждая ≤ limit, склейка == text (плюс/минус
        обрезанные ведущие пробелы у второго+ чанков).
    """
    if not text:
        return [""]
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        window = remaining[:limit]
        cut_at: int | None = None
        for sep in ("\n\n", "\n", " "):
            idx = window.rfind(sep)
            if idx >= limit // 2:
                cut_at = idx + len(sep)
                break
        if cut_at is None:
            cut_at = limit  # hard cut

        chunks.append(remaining[:cut_at].rstrip())
        remaining = remaining[cut_at:].lstrip()

    return chunks


async def answer_long(
    message: Message,
    text: str,
    *,
    reply_markup: Any | None = None,
    parse_mode: Any | None = None,
    escape_html: bool = True,
) -> Message | None:
    """Отправить text как одно или несколько Telegram-сообщений.

    Возвращает последний отправленный Message (тот, к которому прикреплён
    reply_markup), либо None если текст пуст и ничего не ушло. Остальное
    поведение без изменений (reply_markup/parse_mode — только на последнем чанке).

    Reply_markup и parse_mode прикрепляются только к ПОСЛЕДНЕМУ чанку — у
    промежуточных нет ни клавиатуры, ни форматирования. Это стандартное поведение
    для длинных LLM-ответов, чтобы кнопки появлялись под финальной частью.

    Args:
        message: aiogram Message для ответа.
        text: текст ответа (произвольной длины).
        reply_markup: клавиатура — крепится к последнему чанку.
        parse_mode: явный override parse_mode для последнего чанка.
        escape_html: если True (default) — пройти через `html.escape` перед
            сплитом. Бот работает в режиме `parse_mode=HTML`, поэтому любой
            `<` или `&` в LLM-ответе иначе валит send'а 400 Bad Request.
            Установите False только если text — валидный HTML с намеренными
            тегами форматирования.
    """
    if escape_html:
        text = html_escape(text)
    chunks = split_for_telegram(text)
    last_idx = len(chunks) - 1
    sent: Message | None = None
    for i, chunk in enumerate(chunks):
        kwargs: dict[str, Any] = {}
        if i == last_idx:
            if reply_markup is not None:
                kwargs["reply_markup"] = reply_markup
            if parse_mode is not None:
                kwargs["parse_mode"] = parse_mode
        sent = await message.answer(chunk, **kwargs)
    return sent
