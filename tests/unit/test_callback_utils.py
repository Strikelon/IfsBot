"""Тесты safe_answer: ack callback-кнопки не должен ронять хэндлер."""
from __future__ import annotations

from aiogram.exceptions import TelegramBadRequest, TelegramNetworkError

from adapters.telegram.callback_utils import safe_answer


class _FakeCallback:
    def __init__(self, exc: Exception | None) -> None:
        self._exc = exc
        self.calls = 0

    async def answer(self, *args: object, **kwargs: object) -> None:
        self.calls += 1
        if self._exc is not None:
            raise self._exc


async def test_safe_answer_swallows_stale_query() -> None:
    # «query is too old» — протухший callback не должен бросать наружу
    cb = _FakeCallback(
        TelegramBadRequest(
            method="answerCallbackQuery",
            message="query is too old and response timeout expired or query ID is invalid",
        )
    )
    await safe_answer(cb)  # не бросает
    assert cb.calls == 1


async def test_safe_answer_swallows_network_error() -> None:
    cb = _FakeCallback(TelegramNetworkError(method="answerCallbackQuery", message="timeout"))
    await safe_answer(cb)  # не бросает
    assert cb.calls == 1


async def test_safe_answer_forwards_args_on_success() -> None:
    cb = _FakeCallback(None)
    await safe_answer(cb, "текст", show_alert=True)
    assert cb.calls == 1
