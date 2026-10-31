from decimal import Decimal
from unittest.mock import AsyncMock

import pytest
from sqlalchemy import select

from data.models import LlmService, LlmUsageEvent, UsageMetric
from data.repositories import UsageEventRepository, UserRepository
from llm.usage import UsageRecorder


@pytest.mark.asyncio
async def test_usage_recorder_writes_whisper_event() -> None:
    repo = AsyncMock()
    recorder = UsageRecorder(repo)

    await recorder.record_whisper(user_id=7, audio_seconds=87)

    repo.record.assert_awaited_once_with(
        user_id=7,
        service=LlmService.WHISPER,
        model="whisper-1",
        usage_metric=UsageMetric.AUDIO_SECONDS,
        usage_amount=87,
        cost_usd=Decimal("0.008700"),
        session_id=None,
        prompt_version=None,
    )


@pytest.mark.asyncio
async def test_usage_recorder_passes_session_id() -> None:
    repo = AsyncMock()
    recorder = UsageRecorder(repo)

    await recorder.record_whisper(user_id=7, audio_seconds=10, session_id=42)

    repo.record.assert_awaited_once()
    assert repo.record.await_args.kwargs["session_id"] == 42


@pytest.mark.asyncio
async def test_usage_recorder_rounds_audio_seconds_up() -> None:
    """7.4 секунды → 8 (округляем ВВЕРХ — мы платим, не получаем)"""
    repo = AsyncMock()
    recorder = UsageRecorder(repo)

    await recorder.record_whisper(user_id=7, audio_seconds=7.4)

    assert repo.record.await_args.kwargs["usage_amount"] == 8


@pytest.mark.asyncio
async def test_usage_recorder_writes_openrouter_generator_two_events() -> None:
    """OpenRouter generator пишет два события: input_tokens + output_tokens."""
    repo = AsyncMock()
    recorder = UsageRecorder(repo)

    await recorder.record_openrouter_generator(
        user_id=7,
        session_id=42,
        model="google/gemini-3-pro",
        input_tokens=1000,
        output_tokens=500,
        prompt_version="quick.v1",
    )

    assert repo.record.await_count == 2
    call_input = repo.record.await_args_list[0].kwargs
    call_output = repo.record.await_args_list[1].kwargs

    assert call_input["service"] == LlmService.OPENROUTER
    assert call_input["model"] == "google/gemini-3-pro"
    assert call_input["usage_metric"] == UsageMetric.INPUT_TOKENS
    assert call_input["usage_amount"] == 1000
    assert call_input["session_id"] == 42
    assert call_input["prompt_version"] == "quick.v1"
    # Вся стоимость на input-событии. По актуальному прайсу gemini-2.5-pro
    # ($1.25 input / $10.00 output per 1M, см. src/llm/pricing.py 2026-05-19):
    # 1000 * 1.25 / 1M + 500 * 10 / 1M = 0.00125 + 0.005 = 0.006250.
    assert call_input["cost_usd"] == Decimal("0.006250")

    assert call_output["service"] == LlmService.OPENROUTER
    assert call_output["usage_metric"] == UsageMetric.OUTPUT_TOKENS
    assert call_output["usage_amount"] == 500
    assert call_output["session_id"] == 42
    assert call_output["prompt_version"] == "quick.v1"
    assert call_output["cost_usd"] == Decimal("0")


@pytest.mark.asyncio
async def test_usage_recorder_openrouter_handles_zero_output() -> None:
    """Если по какой-то причине output=0 — всё равно пишем два события."""
    repo = AsyncMock()
    recorder = UsageRecorder(repo)

    await recorder.record_openrouter_generator(
        user_id=7,
        session_id=None,
        model="google/gemini-3-pro",
        input_tokens=2000,
        output_tokens=0,
        prompt_version="quick.v1",
    )

    assert repo.record.await_count == 2


@pytest.mark.asyncio
async def test_record_openrouter_classifier_writes_two_events(test_session, schema):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=1001, display_name="x")
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_openrouter_classifier(
        user_id=user.id,
        session_id=None,
        model="google/gemini-2.5-flash",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        prompt_version="self_gate.v1",
    )

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert len(events) == 2
    assert all(e.service == LlmService.OPENROUTER for e in events)
    assert all(e.model == "google/gemini-2.5-flash" for e in events)
    assert all(e.prompt_version == "self_gate.v1" for e in events)

    by_metric = {e.usage_metric: e for e in events}
    assert by_metric[UsageMetric.INPUT_TOKENS].usage_amount == 1_000_000
    assert by_metric[UsageMetric.INPUT_TOKENS].cost_usd == Decimal("2.800000")  # full cost
    assert by_metric[UsageMetric.OUTPUT_TOKENS].usage_amount == 1_000_000
    assert by_metric[UsageMetric.OUTPUT_TOKENS].cost_usd == Decimal("0")


@pytest.mark.asyncio
async def test_record_openrouter_intent_writes_two_events(test_session, schema):
    """Intent-классификатор (gpt-4o-mini) пишет два события с ценой gpt-4o-mini."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=1003, display_name="x")
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_openrouter_intent(
        user_id=user.id,
        session_id=None,
        model="openai/gpt-4o-mini",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        prompt_version="intent.loop.v1",
    )

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert len(events) == 2
    assert all(e.service == LlmService.OPENROUTER for e in events)
    assert all(e.model == "openai/gpt-4o-mini" for e in events)
    by_metric = {e.usage_metric: e for e in events}
    # 1M input × $0.15 + 1M output × $0.60 = $0.75, вся стоимость на input-событии.
    assert by_metric[UsageMetric.INPUT_TOKENS].cost_usd == Decimal("0.750000")
    assert by_metric[UsageMetric.OUTPUT_TOKENS].cost_usd == Decimal("0")


@pytest.mark.asyncio
async def test_record_openrouter_context_priced_by_flash_lite(test_session, schema):
    """Память (memory.extract) и компоновка едут на google/gemini-3.1-flash-lite.

    Ставка $0.25/M input + $1.50/M output; вся стоимость на input-событии.
    Раньше здесь считалось по gpt-4o-mini — после смены модели лимит трат врал.
    """
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=1004, display_name="x")
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_openrouter_context(
        user_id=user.id,
        session_id=None,
        model="google/gemini-3.1-flash-lite",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        prompt_version="memory.extract.v4",
    )

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert len(events) == 2
    assert all(e.model == "google/gemini-3.1-flash-lite" for e in events)
    by_metric = {e.usage_metric: e for e in events}
    assert by_metric[UsageMetric.INPUT_TOKENS].cost_usd == Decimal("1.750000")
    assert by_metric[UsageMetric.OUTPUT_TOKENS].cost_usd == Decimal("0")


@pytest.mark.asyncio
async def test_record_openrouter_classifier_zero_tokens(test_session, schema):
    """OpenRouter может вернуть пустой usage — пишем 0 и cost=0, чтобы не падать."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=1002, display_name="x")
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_openrouter_classifier(
        user_id=user.id,
        session_id=None,
        model="google/gemini-2.5-flash",
        input_tokens=0,
        output_tokens=0,
        prompt_version="p2_find.v1",
    )

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert len(events) == 2
    assert all(e.cost_usd == Decimal("0") for e in events)
