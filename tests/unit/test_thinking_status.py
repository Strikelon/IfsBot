"""Unit test for thinking_status rotation.

Регрессионный тест на BUG #1 (2026-05-21): убеждаемся, что фоновая задача
действительно тикает (edit_text вызывается) при долгом ожидании.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from adapters.telegram.thinking_status import thinking_status


@pytest.mark.asyncio
async def test_thinking_status_rotates_at_least_once() -> None:
    placeholder = MagicMock()
    placeholder.edit_text = AsyncMock()
    placeholder.delete = AsyncMock()
    placeholder.chat.id = 42
    placeholder.message_id = 100

    message = MagicMock()
    message.answer = AsyncMock(return_value=placeholder)

    async with thinking_status(message):
        # Ждём чуть больше одного периода (5с) — должна быть как минимум 1 ротация
        await asyncio.sleep(5.5)

    assert placeholder.edit_text.call_count >= 1, (
        f"thinking_status не сделал ни одной ротации за 5.5 секунд "
        f"(edit_text.call_count={placeholder.edit_text.call_count})"
    )
    # При выходе плейсхолдер удаляется
    placeholder.delete.assert_called_once()


@pytest.mark.asyncio
async def test_thinking_status_initial_send_failure_does_not_propagate() -> None:
    message = MagicMock()
    message.answer = AsyncMock(side_effect=Exception("network blip"))

    # Должно тихо проглотить ошибку и пройти контекст
    async with thinking_status(message):
        pass


@pytest.mark.asyncio
async def test_thinking_status_cancels_rotation_on_exit() -> None:
    placeholder = MagicMock()
    placeholder.edit_text = AsyncMock()
    placeholder.delete = AsyncMock()
    placeholder.chat.id = 42
    placeholder.message_id = 100

    message = MagicMock()
    message.answer = AsyncMock(return_value=placeholder)

    async with thinking_status(message):
        await asyncio.sleep(0.1)  # быстрый выход — ротации быть не должно

    assert placeholder.edit_text.call_count == 0
    placeholder.delete.assert_called_once()
