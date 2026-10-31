from unittest.mock import MagicMock, patch

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr

import llm.http_client as sut
from llm.http_client import create_openai_client, create_openrouter_client


def test_creates_client_without_proxy() -> None:
    client = create_openai_client(api_key=SecretStr("sk-test"), proxy=None)
    try:
        assert isinstance(client, AsyncOpenAI)
        assert client.api_key == "sk-test"
    finally:
        # client.aclose() — async, нельзя из sync-теста; httpx-клиент GC соберёт
        pass


def test_creates_client_with_proxy() -> None:
    proxy_url = "http://u:p@proxy.example:8080"

    # patch.object заменяет ссылку на httpx внутри модуля sut, не трогая
    # глобальный httpx.  Это нужно, потому что AsyncOpenAI.__init__ делает
    # isinstance(http_client, httpx.AsyncClient) — передача MagicMock() напрямую
    # провалит эту проверку.  side_effect делегирует реальному классу, так что
    # возвращаемый объект является настоящим httpx.AsyncClient.
    fake_httpx = MagicMock(spec=httpx)
    fake_httpx.AsyncClient.side_effect = lambda **kwargs: httpx.AsyncClient(**kwargs)

    with patch.object(sut, "httpx", fake_httpx):
        client = create_openai_client(
            api_key=SecretStr("sk-test"),
            proxy=SecretStr(proxy_url),
        )

        # httpx.AsyncClient должен быть создан с proxy=<url>
        fake_httpx.AsyncClient.assert_called_once()
        call_kwargs = fake_httpx.AsyncClient.call_args.kwargs
        assert call_kwargs.get("proxy") == proxy_url

        assert isinstance(client, AsyncOpenAI)


def test_openai_client_stt_read_timeout_is_120s() -> None:
    """STT-клиент: read-таймаут 120с (а не 600), чтобы зависший Whisper падал быстро."""
    fake_httpx = MagicMock(spec=httpx)
    fake_httpx.AsyncClient.side_effect = lambda **kwargs: httpx.AsyncClient(**kwargs)
    with patch.object(sut, "httpx", fake_httpx):
        create_openai_client(
            api_key=SecretStr("sk-test"), proxy=SecretStr("http://u:p@px:8080")
        )
        timeout = fake_httpx.AsyncClient.call_args.kwargs.get("timeout")
        assert timeout.read == 120.0
        assert timeout.connect == 5.0


def test_openai_client_no_proxy_still_sets_timeout() -> None:
    """Даже без прокси (dev) STT-клиент получает 120с-таймаут, не SDK-дефолт 600с."""
    fake_httpx = MagicMock(spec=httpx)
    fake_httpx.AsyncClient.side_effect = lambda **kwargs: httpx.AsyncClient(**kwargs)
    with patch.object(sut, "httpx", fake_httpx):
        create_openai_client(api_key=SecretStr("sk-test"), proxy=None)
        fake_httpx.AsyncClient.assert_called_once()
        assert fake_httpx.AsyncClient.call_args.kwargs.get("timeout").read == 120.0


def test_openai_client_fails_fast_no_retries() -> None:
    """STT-клиент без ретраев: зависший запрос падает за ~один таймаут, не множится."""
    client = create_openai_client(api_key=SecretStr("sk-test"), proxy=None)
    assert client.max_retries == 0


def test_openai_client_repr_does_not_expose_api_key() -> None:
    """AsyncOpenAI repr должно не показывать api_key (проверка контракта SDK)."""
    proxy_url = "http://u:p@proxy.example:8080"
    client = create_openai_client(
        api_key=SecretStr("sk-test"),
        proxy=SecretStr(proxy_url),
    )
    rep = repr(client)
    assert "sk-test" not in rep


def test_openrouter_client_uses_base_url() -> None:
    client = create_openrouter_client(
        api_key=SecretStr("sk-or-test"),
        proxy=None,
        base_url="https://openrouter.ai/api/v1",
    )
    assert isinstance(client, AsyncOpenAI)
    assert client.api_key == "sk-or-test"
    assert str(client.base_url).rstrip("/") == "https://openrouter.ai/api/v1"


def test_openrouter_client_with_proxy() -> None:
    proxy_url = "http://u:p@proxy.example:8080"
    fake_httpx = MagicMock(spec=httpx)
    fake_httpx.AsyncClient.side_effect = lambda **kwargs: httpx.AsyncClient(**kwargs)

    with patch.object(sut, "httpx", fake_httpx):
        client = create_openrouter_client(
            api_key=SecretStr("sk-or-test"),
            proxy=SecretStr(proxy_url),
            base_url="https://openrouter.ai/api/v1",
        )
        fake_httpx.AsyncClient.assert_called_once()
        assert fake_httpx.AsyncClient.call_args.kwargs.get("proxy") == proxy_url
        assert isinstance(client, AsyncOpenAI)


def test_openrouter_client_repr_does_not_expose_api_key() -> None:
    client = create_openrouter_client(
        api_key=SecretStr("sk-or-secret"),
        proxy=None,
        base_url="https://openrouter.ai/api/v1",
    )
    assert "sk-or-secret" not in repr(client)
