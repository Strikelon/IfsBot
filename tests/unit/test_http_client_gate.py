from pydantic import SecretStr

from llm.gate import GatedTransport
from llm.http_client import create_openrouter_client, make_gated_httpx_client  # noqa: F401


def test_openrouter_client_uses_gated_transport_when_semaphore_given() -> None:
    import asyncio

    loop = asyncio.new_event_loop()
    try:
        sem = loop.run_until_complete(_mk_sem())
    finally:
        loop.close()
    client = create_openrouter_client(
        api_key=SecretStr("k"),
        proxy=None,
        base_url="https://openrouter.ai/api/v1",
        semaphore=sem,
    )
    transport = client._client._transport  # httpx внутри AsyncOpenAI
    assert isinstance(transport, GatedTransport)


async def _mk_sem():
    import asyncio

    return asyncio.Semaphore(3)


def test_openrouter_client_has_retries() -> None:
    from pydantic import SecretStr

    from llm.http_client import create_openrouter_client

    client = create_openrouter_client(
        api_key=SecretStr("k"), proxy=None, base_url="https://openrouter.ai/api/v1",
    )
    assert client.max_retries == 2
