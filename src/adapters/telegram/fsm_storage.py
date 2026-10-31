"""Persistent FSM-storage для aiogram3 поверх SQLAlchemy.

Зачем: `MemoryStorage` (по умолчанию у aiogram) держит state в RAM, поэтому
рестарт uvicorn стирает FSM-state у ВСЕХ активных юзеров. Юзер, остановленный
посреди Quick QA или онбординга, после деплоя теряет контекст и не может
ответить (aiogram не находит handler для его сообщения).

Эта реализация хранит state и data в таблице `fsm_state` (миграция 0006).
Ключ — стабильное строковое представление aiogram `StorageKey`. Соответствует
интерфейсу `BaseStorage` aiogram3.

Sharing-session: в рамках одного update FSM storage переиспользует ту же
AsyncSession что и handler (set через `current_session` ContextVar в
`DbSessionMiddleware`). Без шеринга две async-сессии конкурируют за SQLite
write-lock (handler держит BEGIN всё время пока крутятся Telegram API-вызовы),
что в реальном времени даёт `database is locked` даже под WAL+busy_timeout.
Когда контекст не выставлен (фон/тесты), открываем свою короткоживущую сессию.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

from aiogram.fsm.state import State
from aiogram.fsm.storage.base import BaseStorage, StorageKey
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from data.models import FsmState

current_session: ContextVar[AsyncSession | None] = ContextVar(
    "fsm_current_session", default=None
)
"""Set by DbSessionMiddleware to the request-scoped AsyncSession. Read by
SqlAlchemyFSMStorage to avoid opening a second concurrent SQLite writer."""


def _stringify(key: StorageKey) -> str:
    """Сериализовать composite StorageKey в строку для PK."""
    return (
        f"{key.bot_id}:{key.chat_id}:{key.user_id}:"
        f"{key.thread_id or 0}:{key.business_connection_id or ''}:{key.destiny}"
    )


class SqlAlchemyFSMStorage(BaseStorage):
    """FSM-storage поверх SQLAlchemy async session.

    Per call:
      - Если выставлен `current_session` ContextVar (handler-flow через
        DbSessionMiddleware) — используем эту сессию, БЕЗ commit. Middleware
        закоммитит на успешном handler, откатит на исключении. FSM-апдейт
        становится атомарным с DB-апдейтами handler'а.
      - Иначе (фон/тесты/cleanup) — открываем свою короткоживущую сессию
        через `session_factory` и коммитим сами.
    """

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def set_state(self, key: StorageKey, state: State | str | None = None) -> None:
        state_value: str | None
        if state is None:
            state_value = None
        elif isinstance(state, State):
            state_value = state.state
        else:
            state_value = state
        await self._write(
            key=_stringify(key),
            state=state_value,
            data=None,
            preserve_data=True,
        )

    async def get_state(self, key: StorageKey) -> str | None:
        return await self._read_state(_stringify(key))

    async def set_data(self, key: StorageKey, data: dict[str, Any]) -> None:
        await self._write(
            key=_stringify(key),
            state=None,
            data=dict(data) if data else {},
            preserve_state=True,
        )

    async def get_data(self, key: StorageKey) -> dict[str, Any]:
        return await self._read_data(_stringify(key))

    async def close(self) -> None:
        # session_factory управляется владельцем (app.deps) — здесь ничего не делаем.
        return None

    async def _write(
        self,
        *,
        key: str,
        state: str | None,
        data: dict[str, Any] | None,
        preserve_state: bool = False,
        preserve_data: bool = False,
    ) -> None:
        shared = current_session.get()
        if shared is not None:
            await self._upsert(
                shared,
                key=key,
                state=state,
                data=data,
                preserve_state=preserve_state,
                preserve_data=preserve_data,
            )
            await shared.flush()
            return
        async with self._session_factory() as session:
            await self._upsert(
                session,
                key=key,
                state=state,
                data=data,
                preserve_state=preserve_state,
                preserve_data=preserve_data,
            )
            await session.commit()

    async def _read_state(self, key: str) -> str | None:
        shared = current_session.get()
        if shared is not None:
            row = await shared.get(FsmState, key)
            return row.state if row else None
        async with self._session_factory() as session:
            row = await session.get(FsmState, key)
            return row.state if row else None

    async def _read_data(self, key: str) -> dict[str, Any]:
        shared = current_session.get()
        if shared is not None:
            row = await shared.get(FsmState, key)
        else:
            async with self._session_factory() as session:
                row = await session.get(FsmState, key)
        if row is None or row.data is None:
            return {}
        return dict(row.data)

    @staticmethod
    async def _upsert(
        session: AsyncSession,
        *,
        key: str,
        state: str | None,
        data: dict[str, Any] | None,
        preserve_state: bool = False,
        preserve_data: bool = False,
    ) -> None:
        """Insert-or-update строки в fsm_state.

        preserve_state=True — при апдейте поле `state` НЕ перезаписывается
        (используется когда вызывается set_data). preserve_data — симметрично
        для set_state.
        """
        existing = await session.get(FsmState, key)
        if existing is None:
            session.add(FsmState(key=key, state=state, data=data))
            return
        if not preserve_state:
            existing.state = state
        if not preserve_data:
            existing.data = data
