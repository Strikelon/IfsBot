"""Глобальный вентиль конкуренции для LLM/STT-вызовов.

Семафор живёт на уровне httpx-транспорта: ВСЕ запросы через клиент
(generator, classifier, intent, context — все делят один OpenRouter-ключ)
проходят через один общий семафор. Лишние ждут в очереди, а не пробивают
лимит провайдера в потолок (что давало бы 429). Это очередь, не user-лимит:
никому не отказ, ожидание укрыто индикатором «бот думает…».
"""
from __future__ import annotations

import asyncio

import httpx


def make_semaphore(limit: int) -> asyncio.Semaphore:
    """Семафор на N одновременных in-flight запросов.

    Создавать ДО старта event loop — семафор привязывается к loop лениво при первом acquire
    (безопасно на Python ≥3.10).
    """
    return asyncio.Semaphore(limit)


class GatedTransport(httpx.AsyncHTTPTransport):
    """httpx-транспорт, ограничивающий число одновременных запросов семафором.

    `_inner_mock` — точка подмены в тестах (MockTransport). В проде None —
    запрос идёт через настоящий AsyncHTTPTransport (super()).
    """

    def __init__(self, *, semaphore: asyncio.Semaphore, **transport_kwargs: object) -> None:
        super().__init__(**transport_kwargs)  # type: ignore[arg-type]
        self._semaphore = semaphore
        self._inner_mock: httpx.MockTransport | None = None

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        async with self._semaphore:
            if self._inner_mock is not None:
                return await self._inner_mock.handle_async_request(request)
            return await super().handle_async_request(request)
