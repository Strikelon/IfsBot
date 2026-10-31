"""Aiogram-middlewares для инжекта зависимостей в хэндлеры.

DbSessionMiddleware открывает AsyncSession per-update, передаёт через
`data['session']`, коммитит на успешный handler, откатывает на исключение.
Также выставляет `current_session` ContextVar, чтобы `SqlAlchemyFSMStorage`
писал в эту же сессию, а не открывал свою (иначе на SQLite получаем
`database is locked` — две сессии конкурируют за write-lock).

BusyGuardMiddleware перехватывает Message-update'ы, если для этого юзера
сейчас активен stt_lock или qa_lock — отвечает guard notice и НЕ диспатчит
дальше. Для voice-сообщений сам захватывает stt_lock на всю длительность
handler'а — так гарантирует что вся обработка voice (STT + DB write +
reply) проходит атомарно с точки зрения других сообщений юзера.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any

import structlog
from aiogram import BaseMiddleware
from aiogram.types import Message, TelegramObject
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from adapters.telegram.fsm_storage import current_session
from adapters.telegram.menu import ensure_menu_fresh
from data.analytics_events import AnalyticsEventRepository
from data.repositories import UserRepository

log = structlog.get_logger(__name__)


class LogContextMiddleware(BaseMiddleware):
    """Привязывает tg_user_id/username/тип-апдейта к structlog contextvars на апдейт.

    Регистрируется ПЕРВЫМ (внешним). `merge_contextvars` уже в цепочке процессоров
    (см. app/logging.py), поэтому ВСЕ лог-строки в рамках обработки апдейта несут
    эти поля. На проде по нику/id можно вытащить всю цепочку событий юзера:
    `journalctl -u dream-bot | grep <username|id>`. Поля очищаются в finally,
    чтобы не «протекли» в следующий апдейт.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        tg_user = getattr(event, "from_user", None)
        structlog.contextvars.bind_contextvars(
            tg_user_id=getattr(tg_user, "id", None),
            tg_username=getattr(tg_user, "username", None),
            update=type(event).__name__,
        )
        try:
            return await handler(event, data)
        finally:
            structlog.contextvars.unbind_contextvars(
                "tg_user_id", "tg_username", "update"
            )


class DbSessionMiddleware(BaseMiddleware):
    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        async with self._session_factory() as session:
            data["session"] = session
            token = current_session.set(session)
            try:
                result = await handler(event, data)
            except Exception:
                await session.rollback()
                raise
            else:
                await session.commit()
                return result
            finally:
                current_session.reset(token)


class DailyActiveMiddleware(BaseMiddleware):
    """Пишет один daily_active на юзера в день. Никогда не роняет обработчик."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        try:
            session = data.get("session")
            from_user = getattr(event, "from_user", None)
            if session is not None and from_user is not None:
                async with session.begin_nested():
                    user = await UserRepository(session).get_by_tg_id(from_user.id)
                    if user is not None:
                        await AnalyticsEventRepository(session).emit_daily_active(
                            user_id=user.id, telegram_user_id=from_user.id
                        )
        except Exception:  # noqa: BLE001 — аналитика не должна ломать апдейт
            log.warning("daily_active_emit_failed", exc_info=True)
        return await handler(event, data)


class BusyGuardMiddleware(BaseMiddleware):
    """Глобальный guard на сообщения юзера во время STT/LLM-обработки.

    Если у юзера активен stt_lock или qa_lock — отвечаем соответствующим
    guard notice и НЕ диспатчим update дальше. Это закрывает race window
    между «бот начал обрабатывать voice» и «бот закончил отправку reply».
    Для voice-сообщений сам захватывает stt_lock на всю длительность
    handler'а — гарантирует что весь pipeline (STT + DB + reply) проходит
    атомарно.
    """

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        if not isinstance(event, Message) or event.from_user is None:
            return await handler(event, data)

        # Lazy-import чтобы не таскать handler-зависимости в middleware-импортах.
        from adapters.telegram.handlers._qa_lock import (
            is_locked as is_qa_locked,
        )
        from adapters.telegram.handlers._qa_lock import (
            is_stt_locked,
            stt_user_lock,
        )
        from adapters.telegram.texts import STT_GUARD_NOTICE, THINKING_GUARD_NOTICE

        tg_id = event.from_user.id

        # Если другая обработка уже в полёте — отвечаем guard и не пропускаем.
        if is_stt_locked(tg_id):
            log.info("busy_guard.intercepted", lock="stt", tg_id=tg_id)
            await event.answer(STT_GUARD_NOTICE)
            return None
        if is_qa_locked(tg_id):
            log.info("busy_guard.intercepted", lock="qa", tg_id=tg_id)
            await event.answer(THINKING_GUARD_NOTICE)
            return None

        # Для voice-сообщений захватываем stt_lock на ВЕСЬ handler — STT,
        # DB-запись и финальный reply «Записал» проходят атомарно.
        if event.voice is not None:
            async with stt_user_lock(tg_id) as acquired:
                if not acquired:
                    log.info("busy_guard.intercepted", lock="stt_race", tg_id=tg_id)
                    await event.answer(STT_GUARD_NOTICE)
                    return None
                return await handler(event, data)

        return await handler(event, data)


class UserActivityMiddleware(BaseMiddleware):
    """Bump last_seen_at + sync username on every Message/CallbackQuery from a user.

    MUST be registered AFTER DbSessionMiddleware so data['session'] is present.
    Tracking failures are swallowed — they must never break message handling.
    """

    async def __call__(self, handler, event, data):
        session = data.get("session")
        tg_user = getattr(event, "from_user", None)
        if session is not None and tg_user is not None and not tg_user.is_bot:
            try:
                repo = UserRepository(session)
                user = await repo.get_or_create_by_tg_id(
                    telegram_user_id=tg_user.id, display_name=tg_user.first_name
                )
                await repo.touch_activity(
                    user_id=user.id, username=tg_user.username, now=datetime.now(UTC)
                )
                # Меню «☰»: догоняем набор команд, если он отстал от текущей
                # версии. Telegram хранит per-chat список вечно, а раньше он
                # выставлялся только на /start и в конце онбординга.
                bot = data.get("bot")
                if bot is not None:
                    await ensure_menu_fresh(
                        bot=bot, session=session, user=user,
                        settings=data.get("settings"),
                    )
            except Exception:  # tracking must never break message handling
                pass
        return await handler(event, data)
