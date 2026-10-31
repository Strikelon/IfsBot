import asyncio

import httpx
import pytest

from llm.gate import GatedTransport, make_semaphore


@pytest.mark.asyncio
async def test_gate_caps_concurrency() -> None:
    sem = make_semaphore(2)
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.05)
        in_flight -= 1
        return httpx.Response(200, json={"ok": True})

    # GatedTransport поверх MockTransport: семафор должен держать peak<=2
    transport = GatedTransport(semaphore=sem)
    transport._inner_mock = httpx.MockTransport(handler)  # см. реализацию: делегирует в mock

    async with httpx.AsyncClient(transport=transport) as client:
        await asyncio.gather(*[client.get("http://x/") for _ in range(10)])

    assert peak <= 2
