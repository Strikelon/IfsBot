from typing import Any
from unittest.mock import MagicMock

from aiogram import Bot, Dispatcher
from pydantic import SecretStr

from app.bot import create_bot, create_dispatcher


def test_create_bot_returns_bot_with_token() -> None:
    bot = create_bot(token=SecretStr("123:ABC"))
    assert isinstance(bot, Bot)
    assert bot.token == "123:ABC"


def test_create_dispatcher_no_args_works() -> None:
    """Обратная совместимость: пустой вызов всё ещё валиден (для Phase 0 тестов)."""
    dp = create_dispatcher()
    assert isinstance(dp, Dispatcher)
    assert len(dp.sub_routers) >= 1


def test_create_dispatcher_registers_session_middleware() -> None:
    session_factory = MagicMock()
    dp = create_dispatcher(session_factory=session_factory)

    # Middleware попадает в outer_middlewares? aiogram .middleware() пишет в .outer_middleware
    # вернее, в .middleware (внутренние). Проверим, что у message router он есть.
    assert len(dp.message.middleware) >= 1


def test_create_dispatcher_stashes_workflow_data() -> None:
    sentinel: Any = object()
    dp = create_dispatcher(workflow_data={"openai_client": sentinel, "answer_to_life": 42})

    assert dp["openai_client"] is sentinel
    assert dp["answer_to_life"] == 42
