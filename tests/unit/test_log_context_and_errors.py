"""LogContextMiddleware (привязка tg-юзера к логам) + глобальный error-handler."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import structlog


@pytest.mark.asyncio
async def test_log_context_middleware_binds_and_clears_user() -> None:
    from adapters.telegram.middlewares import LogContextMiddleware

    mw = LogContextMiddleware()
    seen: dict = {}

    async def handler(event, data):
        seen.update(structlog.contextvars.get_contextvars())
        return "ok"

    event = SimpleNamespace(
        from_user=SimpleNamespace(id=777, username="vasya", is_bot=False)
    )
    result = await mw(handler, event, {})

    assert result == "ok"
    # внутри обработки апдейта tg-поля привязаны к логам
    assert seen.get("tg_user_id") == 777
    assert seen.get("tg_username") == "vasya"
    # после обработки — очищены (не текут в следующий апдейт)
    after = structlog.contextvars.get_contextvars()
    assert "tg_user_id" not in after


@pytest.mark.asyncio
async def test_log_context_middleware_handles_missing_user() -> None:
    from adapters.telegram.middlewares import LogContextMiddleware

    mw = LogContextMiddleware()

    async def handler(event, data):
        return "ok"

    event = SimpleNamespace(from_user=None)
    assert await mw(handler, event, {}) == "ok"


@pytest.mark.asyncio
async def test_global_error_handler_logs_and_notifies_user() -> None:
    from adapters.telegram.error_handler import handle_global_error
    from adapters.telegram.texts import GENERIC_ERROR_NOTICE

    message = AsyncMock()
    message.from_user = SimpleNamespace(id=42, username="kate", is_bot=False)
    message.answer = AsyncMock()
    update = SimpleNamespace(update_id=5, message=message, callback_query=None)
    event = SimpleNamespace(exception=RuntimeError("boom"), update=update)

    handled = await handle_global_error(event)

    assert handled is True
    message.answer.assert_awaited_once_with(GENERIC_ERROR_NOTICE)


@pytest.mark.asyncio
async def test_global_error_handler_notify_failure_is_swallowed() -> None:
    """Если даже уведомление юзера упало — handler не падает (возвращает True)."""
    from adapters.telegram.error_handler import handle_global_error

    message = AsyncMock()
    message.from_user = SimpleNamespace(id=42, username=None, is_bot=False)
    message.answer = AsyncMock(side_effect=RuntimeError("send failed"))
    update = SimpleNamespace(update_id=5, message=message, callback_query=None)
    event = SimpleNamespace(exception=RuntimeError("boom"), update=update)

    assert await handle_global_error(event) is True
