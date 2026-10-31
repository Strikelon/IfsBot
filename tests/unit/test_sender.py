"""Тесты для adapters/telegram/sender.py — сплит длинных LLM-ответов."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aiogram.types import Message

from adapters.telegram.sender import (
    TELEGRAM_MESSAGE_LIMIT,
    answer_long,
    split_for_telegram,
)


def test_short_text_returns_single_chunk() -> None:
    text = "Привет, это короткое сообщение."
    assert split_for_telegram(text) == [text]


def test_empty_text_returns_single_empty_chunk() -> None:
    # Контракт: возвращаем непустой список даже для "" (чтобы answer_long не падал).
    assert split_for_telegram("") == [""]


def test_splits_on_paragraph_boundary_when_possible() -> None:
    para = "x" * 3000
    text = f"{para}\n\n{para}"
    chunks = split_for_telegram(text)
    assert len(chunks) == 2
    # Каждый чанк целый абзац, без «\n\n» в середине
    assert chunks[0] == para
    assert chunks[1] == para


def test_falls_back_to_line_boundary_when_no_paragraph() -> None:
    line = "y" * 3000
    text = f"{line}\n{line}"
    chunks = split_for_telegram(text)
    assert len(chunks) == 2
    assert chunks[0] == line
    assert chunks[1] == line


def test_falls_back_to_space_when_no_line_break() -> None:
    # Один длинный «параграф» с пробелами, без переводов строк
    word = "слово" * 200  # 1000 chars
    text = " ".join([word] * 5)  # ≈ 5000 chars без \n
    chunks = split_for_telegram(text)
    assert len(chunks) >= 2
    # Никакой чанк не превышает SAFE limit
    for chunk in chunks:
        assert len(chunk) <= 4000


def test_hard_cut_when_no_separator_found() -> None:
    # Полностью без разделителей и пробелов — единственный путь это hard cut
    text = "a" * 8500
    chunks = split_for_telegram(text)
    assert len(chunks) >= 2
    # Все чанки ≤ Telegram-лимит
    for chunk in chunks:
        assert len(chunk) <= TELEGRAM_MESSAGE_LIMIT
    # Склейка == исходник (для hard cut разделители не съедаются)
    assert "".join(chunks) == text


def test_all_chunks_under_telegram_limit() -> None:
    """Главный контракт: ни один чанк не должен превышать 4096."""
    # Эмулируем реальный LLM-ответ Quick.v2.1 на ~10000 chars
    para = ("«цитата из сна» " + "разбор разбор разбор. " * 50) + "\n\n"
    text = para * 10
    chunks = split_for_telegram(text)
    for i, chunk in enumerate(chunks):
        assert len(chunk) <= TELEGRAM_MESSAGE_LIMIT, f"chunk #{i} = {len(chunk)} chars"


@pytest.mark.asyncio
async def test_answer_long_short_text_sends_single_message() -> None:
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock()
    await answer_long(msg, "короткое сообщение", reply_markup="MARKUP")
    msg.answer.assert_awaited_once()
    args = msg.answer.await_args
    assert args.args[0] == "короткое сообщение"
    assert args.kwargs.get("reply_markup") == "MARKUP"


@pytest.mark.asyncio
async def test_answer_long_attaches_markup_only_to_last_chunk() -> None:
    """Многосообщенный ответ: reply_markup ставится только на последний chunk."""
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock()
    long_text = "x" * 9000  # точно ≥ 2 чанков
    await answer_long(msg, long_text, reply_markup="KEYBOARD")
    assert msg.answer.await_count >= 2
    calls = msg.answer.await_args_list
    # Все, кроме последнего, без reply_markup
    for call in calls[:-1]:
        assert "reply_markup" not in call.kwargs or call.kwargs.get("reply_markup") is None
    # Последний — с markup
    assert calls[-1].kwargs.get("reply_markup") == "KEYBOARD"


@pytest.mark.asyncio
async def test_answer_long_without_markup_omits_kwarg() -> None:
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock()
    await answer_long(msg, "обычный ответ")
    msg.answer.assert_awaited_once_with("обычный ответ")


@pytest.mark.asyncio
async def test_answer_long_escapes_html_by_default() -> None:
    """Бот стартует с parse_mode=HTML — спецсимволы `<`, `>`, `&` в LLM-ответе
    обязаны escape'иться, иначе Telegram вернёт 400 can't parse entities."""
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock()
    await answer_long(msg, "Защитник <X> уберегает от <Y> & боли")
    sent = msg.answer.await_args.args[0]
    # `<`, `>`, `&` экранированы
    assert "<X>" not in sent
    assert "&" in sent and "&amp;" in sent
    # Содержание сохранено в читаемом виде
    assert "&lt;X&gt;" in sent
    assert "&lt;Y&gt;" in sent


@pytest.mark.asyncio
async def test_answer_long_can_skip_escape_when_explicit() -> None:
    """Для текстов с намеренным HTML-форматированием можно явно отключить escape."""
    msg = AsyncMock(spec=Message)
    msg.answer = AsyncMock()
    await answer_long(msg, "Привет, <b>Александр</b>!", escape_html=False)
    sent = msg.answer.await_args.args[0]
    assert sent == "Привет, <b>Александр</b>!"


@pytest.mark.asyncio
async def test_answer_long_returns_last_sent_message() -> None:
    msg = AsyncMock()
    sentinel = SimpleNamespace(message_id=555)
    msg.answer = AsyncMock(return_value=sentinel)

    result = await answer_long(msg, "короткий текст", reply_markup={"kb": 1})

    assert result is sentinel
    msg.answer.assert_awaited_once()


@pytest.mark.asyncio
async def test_answer_long_returns_last_chunk_message_when_split() -> None:
    msg = AsyncMock()
    returns = [SimpleNamespace(message_id=1), SimpleNamespace(message_id=2)]
    msg.answer = AsyncMock(side_effect=returns)

    long_text = "a" * 4500  # > _SAFE_CHUNK_LIMIT → ≥2 чанка
    result = await answer_long(msg, long_text)

    assert result is returns[-1]
    assert msg.answer.await_count == 2
