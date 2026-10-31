"""Фабрика AsyncOpenAI-клиента с опциональным прокси.

Используется и для Whisper STT (Phase 1), и для OpenRouter Gemini (Phase 2+).
Прокси нужен, потому что бот деплоится на VPS в РФ, а OpenAI/OpenRouter
блокируют исходящий трафик с РФ-IP.

Использует `httpx.AsyncClient` как transport, чтобы можно было прозрачно
подменить outbound endpoint через `proxy=`.
"""

from __future__ import annotations

import asyncio

import httpx
import structlog
from openai import AsyncOpenAI
from pydantic import SecretStr

from llm.gate import GatedTransport

log = structlog.get_logger(__name__)

# Таймаут для OpenRouter (generator может думать десятки секунд): быстрый connect, длинный read.
_OPENAI_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=600.0, pool=600.0)

# Таймаут для Whisper STT: голос распознаётся за секунды (даже длинный — десятки),
# поэтому read=120с — щедрый потолок. Если соединение зависло, запрос упадёт за ~2 мин
# (а не 10), и юзер получит VOICE_TRANSCRIBE_FAILED вместо долгого фриза под stt_lock.
_STT_TIMEOUT = httpx.Timeout(connect=5.0, read=120.0, write=120.0, pool=120.0)


def _build_httpx_client(
    *,
    proxy: SecretStr | None,
    timeout: httpx.Timeout,
    semaphore: asyncio.Semaphore | None,
) -> httpx.AsyncClient:
    """Собрать httpx.AsyncClient с опциональным семафором-транспортом и прокси.

    Если semaphore задан — использует GatedTransport (ограничивает конкуренцию).
    Прокси прозрачно пробрасывается в обоих ветках.
    """
    proxy_url = proxy.get_secret_value() if proxy else None
    if semaphore is not None:
        transport = GatedTransport(semaphore=semaphore, proxy=proxy_url)
        return httpx.AsyncClient(transport=transport, timeout=timeout)
    return httpx.AsyncClient(proxy=proxy_url, timeout=timeout)


# Публичный псевдоним для тестов и внешнего кода.
make_gated_httpx_client = _build_httpx_client


def create_openai_client(
    *,
    api_key: SecretStr,
    proxy: SecretStr | None,
    semaphore: asyncio.Semaphore | None = None,
) -> AsyncOpenAI:
    """Создать AsyncOpenAI-клиент для Whisper STT.

    Таймаут read=120с применяется ВСЕГДА (и с прокси, и без), вместо SDK-дефолта
    600с. `max_retries=0` — чтобы зависший запрос падал за один таймаут, а не
    множился ретраями (для голоса проще попросить юзера переслать).

    Args:
        api_key: OpenAI API ключ.
        proxy: Опциональный HTTP/HTTPS proxy URL ('http://user:pass@host:port').
               Если None — клиент создаётся без прокси (для разработки вне РФ).
        semaphore: Опциональный семафор конкуренции (GatedTransport). Если None —
                   обычный AsyncClient без ограничения.

    Returns:
        AsyncOpenAI: настроенный клиент.

    Важно: чтобы закрыть клиент чисто на shutdown, в lifespan вызвать
    `await client.close()` (закроет вложенный httpx-клиент).
    """
    if proxy is not None:
        log.info("openai_client.proxy.configured")
    else:
        log.info("openai_client.proxy.absent")
    http_client = _build_httpx_client(proxy=proxy, timeout=_STT_TIMEOUT, semaphore=semaphore)
    return AsyncOpenAI(
        api_key=api_key.get_secret_value(), http_client=http_client, max_retries=0
    )


def create_openrouter_client(
    *,
    api_key: SecretStr,
    proxy: SecretStr | None,
    base_url: str,
    semaphore: asyncio.Semaphore | None = None,
) -> AsyncOpenAI:
    """Создать AsyncOpenAI-клиент, нацеленный на OpenRouter.

    Те же таймауты и опциональный прокси, что и у Whisper-клиента. Отдельная
    фабрика, чтобы в Phase 3+ можно было дать Flash-клиент с другим таймаутом
    (Flash отвечает за миллисекунды, ставить read=600 — оверкилл).

    Args:
        api_key: OpenRouter API ключ (sk-or-v1-...).
        proxy: Опциональный HTTP/HTTPS-прокси.
        base_url: OpenRouter API base URL ('https://openrouter.ai/api/v1').
        semaphore: Опциональный семафор конкуренции (GatedTransport). Если None —
                   обычный AsyncClient без ограничения.

    Returns:
        AsyncOpenAI: настроенный клиент. Закрывать через `await client.close()` в lifespan.
    """
    if proxy is not None:
        log.info("openrouter_client.proxy.configured", base_url=base_url)
    else:
        log.info("openrouter_client.proxy.absent", base_url=base_url)
    http_client = _build_httpx_client(proxy=proxy, timeout=_OPENAI_TIMEOUT, semaphore=semaphore)
    return AsyncOpenAI(
        api_key=api_key.get_secret_value(),
        base_url=base_url,
        http_client=http_client,
        max_retries=2,
    )
