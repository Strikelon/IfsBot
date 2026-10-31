"""Per-user lock для предотвращения concurrent Quick LLM-calls.

Когда юзер шлёт followup или тапает branch — handler берёт user-lock.
Если второе сообщение прилетает пока lock держится — handler сразу шлёт
THINKING_GUARD_NOTICE без LLM-вызова. Это закрывает race window между
aiogram middleware FSM-read и нашим thinking_status.set_state.

Ключ — `telegram_user_id` (НЕ user.id из БД), чтобы idle и другие handler'ы
могли проверить is_locked без DB-запроса.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

# Per-user lock registries. Keys: telegram_user_id.
_qa_locks: dict[int, asyncio.Lock] = {}
_stt_locks: dict[int, asyncio.Lock] = {}


def _get_qa_lock(telegram_user_id: int) -> asyncio.Lock:
    lock = _qa_locks.get(telegram_user_id)
    if lock is None:
        lock = asyncio.Lock()
        _qa_locks[telegram_user_id] = lock
    return lock


def _get_stt_lock(telegram_user_id: int) -> asyncio.Lock:
    lock = _stt_locks.get(telegram_user_id)
    if lock is None:
        lock = asyncio.Lock()
        _stt_locks[telegram_user_id] = lock
    return lock


def is_locked(telegram_user_id: int) -> bool:
    """True если у юзера активный LLM-вызов в Quick handler'е."""
    lock = _qa_locks.get(telegram_user_id)
    return lock is not None and lock.locked()


def is_stt_locked(telegram_user_id: int) -> bool:
    """True если у юзера идёт расшифровка голосового (Whisper STT)."""
    lock = _stt_locks.get(telegram_user_id)
    return lock is not None and lock.locked()


@asynccontextmanager
async def qa_user_lock(telegram_user_id: int) -> AsyncIterator[bool]:
    """Yields True если lock взят, False если уже занят (другой handler в работе).

    Использование в handler:
        async with qa_user_lock(message.from_user.id) as acquired:
            if not acquired:
                await message.answer(THINKING_GUARD_NOTICE)
                return
            # ... handler body, включая thinking_status и LLM-вызов ...
    """
    lock = _get_qa_lock(telegram_user_id)
    if lock.locked():
        yield False
        return
    async with lock:
        yield True


@asynccontextmanager
async def stt_user_lock(telegram_user_id: int) -> AsyncIterator[bool]:
    """Yields True если stt-lock взят, False если уже занят.

    Использование вокруг любой Whisper-транскрипции:
        async with stt_user_lock(message.from_user.id) as acquired:
            if not acquired:
                await message.answer(STT_GUARD_NOTICE)
                return None
            stt_result = await transcribe_with_placeholder(...)
    """
    lock = _get_stt_lock(telegram_user_id)
    if lock.locked():
        yield False
        return
    async with lock:
        yield True
