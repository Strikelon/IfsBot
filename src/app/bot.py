from typing import Any

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.fsm.storage.base import BaseStorage
from aiogram.fsm.storage.memory import MemoryStorage
from pydantic import SecretStr
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from adapters.telegram.error_handler import handle_global_error
from adapters.telegram.fsm_storage import SqlAlchemyFSMStorage
from adapters.telegram.middlewares import (
    BusyGuardMiddleware,
    DailyActiveMiddleware,
    DbSessionMiddleware,
    LogContextMiddleware,
    UserActivityMiddleware,
)
from adapters.telegram.router import get_router
from adapters.telegram.send_throttle import SendThrottleMiddleware

# Таймаут одного вызова Telegram API. Дефолт aiogram — 60 секунд: при подвисшей
# сети человек минуту смотрит в мёртвый экран и жмёт кнопку второй раз (второй
# платный вызов). Long-polling не задет — для getUpdates aiogram сам просит
# request_timeout = session.timeout + polling_timeout. Скачивание голосовых тоже:
# у stream_content свой таймаут.
TELEGRAM_REQUEST_TIMEOUT_SEC = 15.0


def create_bot(
    token: SecretStr,
    proxy: SecretStr | None = None,
    send_rate_per_sec: int = 25,
) -> Bot:
    """Создать `Bot` с дефолтным parse_mode=HTML.

    HTML, а не Markdown — спецсимволы в пользовательском тексте (звёздочки в снах,
    подчёркивания) не сломают форматирование.

    proxy: если задан — Telegram-сессия идёт через HTTP-прокси. Нужно там, где
    подсети api.telegram.org заблокированы на сетевом уровне (напр. прод-VPS в РФ),
    и прямой доступ к Telegram таймаутит. Прокси-аутентификация в URL поддерживается
    aiohttp (`http://user:pass@host:port`).

    send_rate_per_sec: максимум исходящих Telegram API вызовов в секунду (token-bucket
    throttle). Telegram позволяет ~30/сек глобально; дефолт 25 — оставляет запас.

    Таймаут вызова — TELEGRAM_REQUEST_TIMEOUT_SEC (см. комментарий у константы).
    """
    session = (
        AiohttpSession(
            proxy=proxy.get_secret_value(), timeout=TELEGRAM_REQUEST_TIMEOUT_SEC
        )
        if proxy
        else AiohttpSession(timeout=TELEGRAM_REQUEST_TIMEOUT_SEC)
    )
    session.middleware(SendThrottleMiddleware(rate_per_sec=send_rate_per_sec))
    return Bot(
        token=token.get_secret_value(),
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


def create_dispatcher(
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    workflow_data: dict[str, Any] | None = None,
    storage: BaseStorage | None = None,
) -> Dispatcher:
    """Создать `Dispatcher` с middlewares и инжектированными зависимостями.

    Args:
        session_factory: если передан, регистрируется DbSessionMiddleware на
            message и callback_query. Также используется как backend для
            persistent FSM-storage (`SqlAlchemyFSMStorage`), если `storage` не
            задан явно. В Phase 0 не было — передаётся как None.
        workflow_data: словарь с долгоживущими зависимостями (openai_client,
            settings). Доступен в handlers как kwargs по имени ключа.
        storage: явный FSM-storage. Если None — берётся persistent SQL-storage
            (когда есть session_factory) или MemoryStorage (для тестов без БД).
    """
    if storage is None:
        storage = (
            SqlAlchemyFSMStorage(session_factory) if session_factory else MemoryStorage()
        )
    dp = Dispatcher(storage=storage)
    dp.include_router(get_router())
    dp.errors.register(handle_global_error)

    # LogContextMiddleware — ПЕРВЫМ (внешним): привязывает tg_user_id/username ко
    # всем логам в рамках апдейта. Регистрируем до DbSession/прочих.
    log_ctx = LogContextMiddleware()
    dp.message.middleware(log_ctx)
    dp.callback_query.middleware(log_ctx)
    dp.my_chat_member.middleware(log_ctx)

    if session_factory is not None:
        middleware = DbSessionMiddleware(session_factory)
        dp.message.middleware(middleware)
        dp.callback_query.middleware(middleware)
        dp.my_chat_member.middleware(middleware)
        activity = UserActivityMiddleware()
        dp.message.middleware(activity)
        dp.callback_query.middleware(activity)
        daily_active = DailyActiveMiddleware()
        dp.message.middleware(daily_active)
        dp.callback_query.middleware(daily_active)

    # Busy-guard перехватывает сообщения юзера во время STT/LLM-обработки.
    # Регистрируется на message (callback'и сами через qa_user_lock).
    dp.message.middleware(BusyGuardMiddleware())

    if workflow_data:
        for key, value in workflow_data.items():
            dp[key] = value

    return dp
