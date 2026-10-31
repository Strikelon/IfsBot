"""Контейнер всех долгоживущих зависимостей приложения.

Создаётся один раз в lifespan, передаётся в фабрики бота/диспетчера/FastAPI.
Очищается в lifespan на shutdown.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker

from app.config import Settings
from data.database import create_engine, create_session_factory
from llm.gate import make_semaphore
from llm.http_client import create_openai_client, create_openrouter_client

log = structlog.get_logger(__name__)


@dataclass(frozen=True, slots=True)
class AppDependencies:
    settings: Settings
    engine: AsyncEngine
    session_factory: async_sessionmaker[AsyncSession]
    openai_client: AsyncOpenAI
    openrouter_client: AsyncOpenAI
    openrouter_classifier_client: AsyncOpenAI


def create_dependencies(settings: Settings) -> AppDependencies:
    engine = create_engine(settings.database_url)
    session_factory = create_session_factory(engine)

    # Семафоры-вентили: один общий для обоих OpenRouter-клиентов (один ключ → один провайдерный лимит),
    # отдельный для STT (другой ключ и тип нагрузки). 0 в конфиге = без лимита.
    llm_sem = make_semaphore(settings.llm_max_concurrency) if settings.llm_max_concurrency else None
    stt_sem = make_semaphore(settings.stt_max_concurrency) if settings.stt_max_concurrency else None

    openai_client = create_openai_client(
        api_key=settings.openai_api_key, proxy=settings.llm_proxy, semaphore=stt_sem
    )
    # ОБА openrouter-клиента делят ОДИН семафор (один ключ → один лимит провайдера):
    openrouter_client = create_openrouter_client(
        api_key=settings.openrouter_api_key,
        proxy=settings.llm_proxy,
        base_url=settings.openrouter_base_url,
        semaphore=llm_sem,
    )
    openrouter_classifier_client = create_openrouter_client(
        api_key=settings.openrouter_api_key,
        proxy=settings.llm_proxy,
        base_url=settings.openrouter_base_url,
        semaphore=llm_sem,
    )
    return AppDependencies(
        settings=settings,
        engine=engine,
        session_factory=session_factory,
        openai_client=openai_client,
        openrouter_client=openrouter_client,
        openrouter_classifier_client=openrouter_classifier_client,
    )


async def dispose_dependencies(deps: AppDependencies) -> None:
    """Чистый shutdown: закрыть оба OpenAI httpx-клиента и engine pool.

    Каждый шаг изолирован: если один ресурс не закрывается (network hang,
    уже-закрытый httpx и т.п.), остальные всё равно будут освобождены —
    иначе утечёт socket/pool.
    """
    try:
        await deps.openai_client.close()
    except Exception:
        log.warning("deps.dispose.openai_client.failed", exc_info=True)

    try:
        await deps.openrouter_client.close()
    except Exception:
        log.warning("deps.dispose.openrouter_client.failed", exc_info=True)

    try:
        await deps.openrouter_classifier_client.close()
    except Exception:
        log.warning("deps.dispose.openrouter_classifier_client.failed", exc_info=True)

    try:
        await deps.engine.dispose()
    except Exception:
        log.warning("deps.dispose.engine.failed", exc_info=True)
