"""Unit tests for SendThrottleMiddleware.

Adapted from plan brief: aiogram's BaseRequestMiddleware.__call__ contract does NOT include
a `timeout` parameter — timeout is bound via functools.partial to the innermost make_request
before middlewares are chained. Tests therefore omit `timeout` to match the actual contract.
"""

import asyncio

import pytest
from aiogram.exceptions import TelegramRetryAfter

from adapters.telegram.send_throttle import SendThrottleMiddleware


@pytest.mark.asyncio
async def test_retries_after_flood_then_succeeds() -> None:
    calls = {"n": 0}

    async def make_request(bot, method):
        calls["n"] += 1
        if calls["n"] == 1:
            raise TelegramRetryAfter(method=method, message="flood", retry_after=0)
        return "ok"

    mw = SendThrottleMiddleware(rate_per_sec=1000)
    result = await mw(make_request, bot=None, method=object())
    assert result == "ok"
    assert calls["n"] == 2  # один ретрай после flood


@pytest.mark.asyncio
async def test_token_bucket_caps_rate() -> None:
    mw = SendThrottleMiddleware(rate_per_sec=5)

    async def make_request(bot, method):
        return "ok"

    loop = asyncio.get_running_loop()
    start = loop.time()
    for _ in range(10):
        await mw(make_request, bot=None, method=object())
    elapsed = loop.time() - start
    # 10 вызовов при 5/сек не могут пройти мгновенно
    assert elapsed >= 0.8
