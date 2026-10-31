"""Unit-тесты generator-обёртки. OpenRouter-клиент мокается."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm.generator import (
    ErrorCategory,
    GeneratorResult,
    _classify_error,
    generate_deep_response,
    generate_quick_response,
)


def _mock_openrouter_response(
    text: str | None,
    prompt_tokens: int,
    completion_tokens: int,
    finish_reason: str = "stop",
):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=text),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        model_dump=lambda: {"provider": "test-provider"},
    )


def _mock_error_response(code: int, message: str):
    """Симулирует response где choices=[] и есть error-поле (OpenRouter error format)."""
    return SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0),
        error=SimpleNamespace(code=code, message=message),
        model_dump=lambda: {
            "provider": "test-provider",
            "error": {"code": code, "message": message},
        },
    )


# ============================================================
# Старые тесты (kept + adapted for new signature)
# ============================================================


@pytest.mark.asyncio
async def test_generate_quick_response_returns_text_and_tokens() -> None:
    long_text = "Это достаточно длинный ответ IFS-психолога для прохождения min-length check." * 2
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_mock_openrouter_response(long_text, prompt_tokens=1000, completion_tokens=200)
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "system", "content": "x"}, {"role": "user", "content": "y"}],
    )

    assert isinstance(result, GeneratorResult)
    assert result.text == long_text
    assert result.input_tokens == 1000
    assert result.output_tokens == 200
    assert result.error_category is None


@pytest.mark.asyncio
async def test_generate_quick_response_passes_correct_args_to_client() -> None:
    long_text = "Это достаточно длинный ответ IFS-психолога для прохождения min-length check." * 2
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=_mock_openrouter_response(long_text, 5, 5)
    )
    messages = [{"role": "user", "content": "hi"}]

    await generate_quick_response(client=client, messages=messages)

    client.chat.completions.create.assert_awaited()
    kwargs = client.chat.completions.create.await_args.kwargs
    # Should use first model in chain (flash)
    assert kwargs["model"] == "google/gemini-3.5-flash"
    assert kwargs["messages"] == messages
    assert kwargs["temperature"] == 0.6


@pytest.mark.asyncio
async def test_generate_quick_response_handles_missing_usage() -> None:
    """Если OpenRouter не вернул usage — возвращаем нули (защита учёта от падения)."""
    long_text = "ok, это достаточно длинный ответ IFS-психолога чтобы пройти min length check. " * 2
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=long_text),
                    finish_reason="stop",
                )
            ],
            usage=None,
            model_dump=lambda: {},
        )
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )
    assert result.text.startswith("ok")
    assert result.input_tokens == 0
    assert result.output_tokens == 0


@pytest.mark.asyncio
async def test_generate_deep_response_passes_correct_temperature():
    client = AsyncMock()
    long_content = "а" * 200  # > 100 chars
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=long_content),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            model_dump=lambda: {},
        )
    )
    result = await generate_deep_response(
        client=client,
        model="m",
        messages=[{"role": "user", "content": "x"}],
        temperature=0.4,
    )
    assert result.text == long_content
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["temperature"] == 0.4


@pytest.mark.asyncio
async def test_generate_deep_response_empty_choices_returns_empty():
    empty_choices_resp = SimpleNamespace(
        choices=[],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=0),
        model_dump=lambda: {},
    )
    client = AsyncMock()
    # deep still uses legacy signature with single model — all 3 retries fail
    client.chat.completions.create = AsyncMock(
        side_effect=[empty_choices_resp, empty_choices_resp, empty_choices_resp]
    )

    result = await generate_deep_response(
        client=client, model="m", messages=[], temperature=0.5,
    )
    assert result.text == ""


# ============================================================
# Новые TDD тесты: ErrorCategory + fallback chain
# ============================================================


@pytest.mark.parametrize(
    "error_payload,expected",
    [
        ({"code": 502, "message": "Upstream idle timeout"}, ErrorCategory.TRANSIENT),
        ({"code": 503, "message": "Service unavailable"}, ErrorCategory.TRANSIENT),
        ({"code": 429, "message": "Rate limit exceeded"}, ErrorCategory.TRANSIENT),
        ({"code": 403, "message": "content_policy_violation"}, ErrorCategory.CONTENT_FILTER),
        ({"code": 402, "message": "Insufficient credits"}, ErrorCategory.QUOTA_EXCEEDED),
        ({"code": 403, "message": "quota exceeded"}, ErrorCategory.QUOTA_EXCEEDED),
        ({"code": 400, "message": "Invalid request"}, ErrorCategory.PERMANENT),
        # 400 от Gemini при срабатывании фильтра безопасности: это НЕ наш баг,
        # у другого вендора запрос может пройти → CONTENT_FILTER, цепочка едет дальше.
        (
            {"code": 400, "message": "Gemini blocked the request: PROHIBITED_CONTENT"},
            ErrorCategory.CONTENT_FILTER,
        ),
        (
            {"code": 400, "message": "Response was blocked due to SAFETY"},
            ErrorCategory.CONTENT_FILTER,
        ),
        (
            {"code": 400, "message": "flagged by content moderation"},
            ErrorCategory.CONTENT_FILTER,
        ),
        # Настоящий 400 (наш баг в запросе) обязан остаться PERMANENT.
        ({"code": 400, "message": "messages: field required"}, ErrorCategory.PERMANENT),
        ({"code": 401, "message": "Unauthorized"}, ErrorCategory.PERMANENT),
        # 401 не трогаем.
        ({"code": 401, "message": "No auth credentials found"}, ErrorCategory.PERMANENT),
    ],
)
def test_classify_error_from_response(error_payload, expected):
    assert _classify_error(error_payload) == expected


@pytest.mark.asyncio
async def test_fallback_chain_first_model_fails_second_succeeds():
    """First model returns 502, second model returns valid content."""
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    error_resp = _mock_error_response(502, "Upstream idle timeout")
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[error_resp, success_resp]
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert result.error_category is None
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_fallback_chain_all_models_fail():
    """All 3 models return 502. Result should have error_category=TRANSIENT."""
    error_resp = _mock_error_response(502, "Upstream idle timeout")
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[error_resp, error_resp, error_resp]
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == ""
    assert result.error_category == ErrorCategory.TRANSIENT
    assert client.chat.completions.create.await_count == 3


@pytest.mark.asyncio
async def test_fallback_chain_timeout_on_attempt():
    """asyncio.TimeoutError on first attempt → classified as transient → next model tried."""
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    call_count = 0

    async def mock_create(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise TimeoutError()
        return success_resp

    client = AsyncMock()
    client.chat.completions.create = mock_create

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert result.error_category is None
    assert call_count == 2


@pytest.mark.asyncio
async def test_fallback_chain_content_filter_tries_next():
    """finish_reason='content_filter' on first model → next model tried.

    С 2026-07-27 «следующая» — это haiku, а не pro: при content_filter
    оставшиеся модели того же вендора пропускаются. Вызовов по-прежнему 2.
    """
    content_filter_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=None),
                finish_reason="content_filter",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=0),
        model_dump=lambda: {},
    )
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[content_filter_resp, success_resp]
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert result.error_category is None
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_fallback_chain_finish_reason_error_tries_next():
    """finish_reason='error' с непустым контентом (обрыв апстрима на полуслове) →
    НЕ успех, пробуем следующую модель. Иначе юзеру уходит обрезанный разбор без
    хвоста ===QA_OPTIONS=== (нет вопроса и чипов)."""
    truncated = "Разбор оборвался на середине, хвоста опций ещё нет. " * 3
    error_resp = _mock_openrouter_response(truncated, 3000, 0, finish_reason="error")
    long_text = "Полный корректный IFS-разбор с хвостом опций в конце. " * 3
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=[error_resp, success_resp])

    result = await generate_quick_response(
        client=client, messages=[{"role": "user", "content": "x"}]
    )
    assert result.text == long_text
    assert result.error_category is None
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_fallback_chain_finish_reason_length_tries_next():
    """finish_reason='length' (обрезка по max_tokens) → тоже не успех, следующая модель."""
    truncated = "Слишком длинный разбор, обрезан по лимиту токенов на середине. " * 3
    length_resp = _mock_openrouter_response(truncated, 3000, 4000, finish_reason="length")
    long_text = "Полный корректный разбор, уложился в лимит, с хвостом опций. " * 3
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=[length_resp, success_resp])

    result = await generate_quick_response(
        client=client, messages=[{"role": "user", "content": "x"}]
    )
    assert result.text == long_text
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_fallback_chain_quota_exceeded_stops_immediately():
    """402 Insufficient credits → chain stops immediately (no retry)."""
    error_resp = _mock_error_response(402, "Insufficient credits")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=error_resp)

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == ""
    assert result.error_category == ErrorCategory.QUOTA_EXCEEDED
    # Should stop after first attempt
    assert client.chat.completions.create.await_count == 1


@pytest.mark.asyncio
async def test_fallback_chain_short_content_tries_next():
    """Content shorter than 100 chars → classified as failed → next model tried."""
    short_resp = _mock_openrouter_response("Окей", 100, 5)
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[short_resp, success_resp]
    )

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert result.error_category is None
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_per_model_max_completion_tokens_haiku():
    """Claude haiku должен получать max_completion_tokens=4096, не 16384."""
    long_text = "а" * 200
    success_resp = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=long_text),
                finish_reason="stop",
            )
        ],
        usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        model_dump=lambda: {},
    )
    # Make first two models fail so haiku is tried
    error_resp_1 = _mock_error_response(502, "Upstream")
    error_resp_2 = _mock_error_response(502, "Upstream")

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        side_effect=[error_resp_1, error_resp_2, success_resp]
    )

    await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    # Third call should be to haiku
    assert client.chat.completions.create.await_count == 3
    third_call_kwargs = client.chat.completions.create.await_args_list[2].kwargs
    assert third_call_kwargs["model"] == "anthropic/claude-haiku-4.5"
    assert third_call_kwargs["max_completion_tokens"] == 4096


@pytest.mark.asyncio
async def test_generator_result_has_error_category_field():
    """GeneratorResult должен иметь поле error_category."""
    r = GeneratorResult(text="ok", input_tokens=1, output_tokens=1)
    assert r.error_category is None

    r2 = GeneratorResult(
        text="", input_tokens=0, output_tokens=0, error_category=ErrorCategory.TRANSIENT
    )
    assert r2.error_category == ErrorCategory.TRANSIENT


@pytest.mark.asyncio
async def test_fallback_chain_records_actual_model_used():
    """When flash fails 502 and pro succeeds, result.model_used == pro slug."""
    from llm.generator import _QUICK_FALLBACK_CHAIN

    flash_502 = _mock_error_response(502, "Upstream idle timeout")
    long_text = "а" * 200  # > 100 chars — passes min-length check
    pro_ok = _mock_openrouter_response(long_text, prompt_tokens=1000, completion_tokens=200)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=[flash_502, pro_ok])

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.error_category is None
    assert result.text == long_text
    # Second model in chain (pro) should be recorded
    assert result.model_used == _QUICK_FALLBACK_CHAIN[1]
    assert client.chat.completions.create.await_count == 2


@pytest.mark.asyncio
async def test_content_block_400_skips_same_vendor_and_reaches_anthropic():
    """Gemini 400 PROHIBITED_CONTENT → gemini-2.5-pro пропускается, зовётся haiku.

    Регрессия прод-инцидента 2026-07-20: цепочка обрывалась на первой модели,
    юзер получал «Что-то сломалось в коде бота» вместо разбора.
    """
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    blocked_resp = _mock_error_response(400, "Gemini blocked the request: PROHIBITED_CONTENT")
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    seen_models: list[str] = []

    async def mock_create(**kwargs):
        seen_models.append(kwargs["model"])
        return blocked_resp if len(seen_models) == 1 else success_resp

    client = AsyncMock()
    client.chat.completions.create = mock_create

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert result.error_category is None
    assert seen_models == ["google/gemini-3.5-flash", "anthropic/claude-haiku-4.5"]


@pytest.mark.asyncio
async def test_content_block_on_all_vendors_returns_content_filter():
    """Заблокировали и Google, и Anthropic → CONTENT_FILTER (не PERMANENT)."""
    blocked_resp = _mock_error_response(400, "Gemini blocked the request: PROHIBITED_CONTENT")

    seen_models: list[str] = []

    async def mock_create(**kwargs):
        seen_models.append(kwargs["model"])
        return blocked_resp

    client = AsyncMock()
    client.chat.completions.create = mock_create

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == ""
    assert result.error_category == ErrorCategory.CONTENT_FILTER
    assert seen_models == ["google/gemini-3.5-flash", "anthropic/claude-haiku-4.5"]


@pytest.mark.asyncio
async def test_transient_error_does_not_skip_same_vendor():
    """502 у первой модели НЕ должен пропускать вторую google-модель."""
    long_text = "Это достаточно длинный ответ для прохождения min-length check в IFS-разборе." * 2
    error_resp = _mock_error_response(502, "Upstream idle timeout")
    success_resp = _mock_openrouter_response(long_text, 1000, 200)

    seen_models: list[str] = []

    async def mock_create(**kwargs):
        seen_models.append(kwargs["model"])
        return error_resp if len(seen_models) == 1 else success_resp

    client = AsyncMock()
    client.chat.completions.create = mock_create

    result = await generate_quick_response(
        client=client,
        messages=[{"role": "user", "content": "x"}],
    )

    assert result.text == long_text
    assert seen_models == ["google/gemini-3.5-flash", "google/gemini-2.5-pro"]
