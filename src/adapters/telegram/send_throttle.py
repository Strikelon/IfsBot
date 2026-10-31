"""Глобальный throttle исходящих вызовов Telegram Bot API.

Telegram режет на ~30 msg/сек глобально на бота. При спайке (бурст
welcome/онбординга) это даёт 429. Мидлварь: token-bucket ≤rate/сек +
авто-retry на TelegramRetryAfter. Это сглаживание (очередь), а не отказ.

Замечание по контракту aiogram: BaseRequestMiddleware.__call__ принимает
(make_request, bot, method) без timeout — timeout уже привязан через
functools.partial к innermost make_request в session.middleware.wrap_middlewares.
Поэтому make_request здесь вызывается как make_request(bot, method).
"""
from __future__ import annotations

import asyncio

import structlog
from aiogram import Bot
from aiogram.client.session.middlewares.base import BaseRequestMiddleware, NextRequestMiddlewareType
from aiogram.exceptions import TelegramRetryAfter
from aiogram.methods import TelegramMethod
from aiogram.methods.base import Response, TelegramType

log = structlog.get_logger(__name__)

_MAX_RETRIES = 3


class SendThrottleMiddleware(BaseRequestMiddleware):
    """Token-bucket throttle + flood-retry для исходящих Telegram Bot API вызовов.

    rate_per_sec: максимум запросов в секунду (глобально на бота, не на юзера).
    По умолчанию 25 — оставляет запас до лимита Telegram ~30/сек.
    """

    def __init__(self, rate_per_sec: int = 25) -> None:
        self._min_interval = 1.0 / max(1, rate_per_sec)
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def _acquire(self) -> None:
        """Дождаться своей «ячейки» в token-bucket и зарезервировать следующую."""
        async with self._lock:
            loop = asyncio.get_running_loop()
            now = loop.time()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = loop.time()
            self._next_allowed = max(now, self._next_allowed) + self._min_interval

    async def __call__(
        self,
        make_request: NextRequestMiddlewareType[TelegramType],
        bot: Bot,
        method: TelegramMethod[TelegramType],
    ) -> Response[TelegramType]:
        """Вызов middleware: throttle + до 3 ретраев при TelegramRetryAfter."""
        for attempt in range(_MAX_RETRIES + 1):
            await self._acquire()
            try:
                return await make_request(bot, method)
            except TelegramRetryAfter as exc:
                if attempt == _MAX_RETRIES:
                    raise
                log.warning("tg.flood.retry", retry_after=exc.retry_after, attempt=attempt)
                await asyncio.sleep(exc.retry_after)
        raise RuntimeError("unreachable")  # pragma: no cover
