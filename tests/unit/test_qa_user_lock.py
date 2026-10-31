"""Per-user lock: первый caller получает True, второй concurrent — False."""

import asyncio

import pytest

from adapters.telegram.handlers._qa_lock import qa_user_lock


@pytest.mark.asyncio
async def test_first_caller_acquires_lock():
    async with qa_user_lock(telegram_user_id=1) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_concurrent_caller_does_not_acquire():
    user_id = 2
    first_held = asyncio.Event()
    second_result: dict = {}

    async def first_caller():
        async with qa_user_lock(user_id) as acquired:
            assert acquired is True
            first_held.set()
            await asyncio.sleep(0.1)

    async def second_caller():
        await first_held.wait()
        async with qa_user_lock(user_id) as acquired:
            second_result["acquired"] = acquired

    await asyncio.gather(first_caller(), second_caller())
    assert second_result["acquired"] is False


@pytest.mark.asyncio
async def test_sequential_acquires_succeed():
    user_id = 3
    async with qa_user_lock(user_id) as acquired:
        assert acquired is True
    # После выхода — lock освобождён, новый caller получает True
    async with qa_user_lock(user_id) as acquired:
        assert acquired is True


@pytest.mark.asyncio
async def test_different_users_independent():
    async def caller(uid, results):
        async with qa_user_lock(uid) as acquired:
            results[uid] = acquired
            await asyncio.sleep(0.05)
    results = {}
    await asyncio.gather(caller(10, results), caller(20, results))
    assert results == {10: True, 20: True}
