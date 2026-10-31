"""Тесты llm/classifier.py — generic strict-JSON wrapper для Flash."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import BaseModel


class FakeResult(BaseModel):
    flag: bool
    reason: str


def _or_response(content: str, p_tokens: int = 100, c_tokens: int = 20):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=p_tokens, completion_tokens=c_tokens),
    )


@pytest.mark.asyncio
async def test_classify_returns_parsed_pydantic_model():
    from llm.classifier import classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('{"flag": true, "reason": "X"}')
    )

    result = await classify(
        client=client,
        model="google/gemini-2.5-flash",
        messages=[{"role": "user", "content": "x"}],
        schema=FakeResult,
    )
    assert result.parsed.flag is True
    assert result.parsed.reason == "X"
    assert result.input_tokens == 100
    assert result.output_tokens == 20


@pytest.mark.asyncio
async def test_classify_passes_response_format_json_object():
    from llm.classifier import classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('{"flag": false, "reason": ""}')
    )

    await classify(
        client=client,
        model="m",
        messages=[{"role": "user", "content": "x"}],
        schema=FakeResult,
    )
    call_kwargs = client.chat.completions.create.await_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_classify_uses_default_max_tokens():
    from llm.classifier import _CLASSIFIER_MAX_TOKENS, classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('{"flag": false, "reason": ""}')
    )
    await classify(client=client, model="m", messages=[], schema=FakeResult)
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["max_tokens"] == _CLASSIFIER_MAX_TOKENS == 600


@pytest.mark.asyncio
async def test_classify_honours_max_tokens_override():
    """Вызовы с длинным JSON (экстрактор памяти) поднимают потолок вывода."""
    from llm.classifier import classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('{"flag": false, "reason": ""}')
    )
    await classify(
        client=client, model="m", messages=[], schema=FakeResult, max_tokens=1200
    )
    assert client.chat.completions.create.await_args.kwargs["max_tokens"] == 1200


@pytest.mark.asyncio
async def test_classify_raises_on_invalid_json():
    from llm.classifier import ClassifierParseError, classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response("это не json")
    )

    with pytest.raises(ClassifierParseError):
        await classify(
            client=client,
            model="m",
            messages=[],
            schema=FakeResult,
        )


@pytest.mark.asyncio
async def test_classify_raises_on_schema_validation_fail():
    from llm.classifier import ClassifierParseError, classify

    client = AsyncMock()
    # валидный JSON, но не подходит под FakeResult (отсутствует поле reason)
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('{"flag": true}')
    )

    with pytest.raises(ClassifierParseError):
        await classify(
            client=client,
            model="m",
            messages=[],
            schema=FakeResult,
        )


@pytest.mark.asyncio
async def test_classify_handles_empty_choices():
    """OpenRouter может вернуть choices=[] при content-filter — бросаем ClassifierParseError."""
    from llm.classifier import ClassifierParseError, classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=0),
        )
    )
    with pytest.raises(ClassifierParseError):
        await classify(
            client=client,
            model="m",
            messages=[],
            schema=FakeResult,
        )


@pytest.mark.asyncio
async def test_classify_raises_on_empty_content():
    """choices present, but message.content is whitespace — distinct from empty choices."""
    from llm.classifier import ClassifierParseError, classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response("   \n  ")
    )
    with pytest.raises(ClassifierParseError):
        await classify(
            client=client,
            model="m",
            messages=[],
            schema=FakeResult,
        )


@pytest.mark.asyncio
async def test_classify_strips_markdown_fences():
    """Gemini иногда оборачивает JSON в ```json ... ``` — wrapper должен снять."""
    from llm.classifier import classify

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_or_response('```json\n{"flag": true, "reason": "ok"}\n```')
    )
    result = await classify(
        client=client,
        model="m",
        messages=[],
        schema=FakeResult,
    )
    assert result.parsed.flag is True
