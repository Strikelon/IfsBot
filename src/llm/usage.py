"""UsageRecorder — высокоуровневая обёртка над UsageEventRepository.

Знает, какую цену по какой формуле считать для каждого сервиса. Хэндлеры
не должны импортировать pricing.py напрямую — они вызывают
`recorder.record_whisper(...)` и не думают про деньги.
"""

from __future__ import annotations

import math
from decimal import Decimal

from data.models import LlmService, UsageMetric
from data.repositories import UsageEventRepository
from llm.pricing import (
    gemini_3_pro_cost_usd,
    gemini_31_flash_lite_cost_usd,
    gemini_flash_cost_usd,
    gpt_4o_mini_cost_usd,
    whisper_cost_usd,
)


class UsageRecorder:
    def __init__(self, repo: UsageEventRepository) -> None:
        self._repo = repo

    async def record_whisper(
        self,
        *,
        user_id: int,
        audio_seconds: float,
        session_id: int | None = None,
    ) -> None:
        # Округляем вверх (мы платим OpenAI за полные секунды биллинга)
        seconds_int = max(0, math.ceil(audio_seconds))
        await self._repo.record(
            user_id=user_id,
            service=LlmService.WHISPER,
            model="whisper-1",
            usage_metric=UsageMetric.AUDIO_SECONDS,
            usage_amount=seconds_int,
            cost_usd=whisper_cost_usd(seconds_int),
            session_id=session_id,
            prompt_version=None,
        )

    async def record_openrouter_generator(
        self,
        *,
        user_id: int,
        session_id: int | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt_version: str | None,
    ) -> None:
        """Записать один OpenRouter generator-вызов как ДВА события.

        Первое (INPUT_TOKENS) несёт всю стоимость; второе (OUTPUT_TOKENS) — с cost_usd=0.
        Так СУММА по cost_usd за период == фактическим расходам, и можно
        отдельно агрегировать input/output для оптимизации промптов.
        """
        total_cost = gemini_3_pro_cost_usd(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.INPUT_TOKENS,
            usage_amount=input_tokens,
            cost_usd=total_cost,
            prompt_version=prompt_version,
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.OUTPUT_TOKENS,
            usage_amount=output_tokens,
            cost_usd=Decimal("0"),
            prompt_version=prompt_version,
        )

    async def record_openrouter_classifier(
        self,
        *,
        user_id: int,
        session_id: int | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt_version: str | None,
    ) -> None:
        """Записать один OpenRouter classifier-вызов (Gemini Flash) как ДВА события.

        Та же модель учёта, что и в record_openrouter_generator: input-событие
        несёт всю стоимость, output-событие — с cost_usd=0. Стоимость считается
        через gemini_flash_cost_usd (дешевле в ~10× от Pro).
        """
        total_cost = gemini_flash_cost_usd(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.INPUT_TOKENS,
            usage_amount=input_tokens,
            cost_usd=total_cost,
            prompt_version=prompt_version,
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.OUTPUT_TOKENS,
            usage_amount=output_tokens,
            cost_usd=Decimal("0"),
            prompt_version=prompt_version,
        )

    async def record_openrouter_intent(
        self,
        *,
        user_id: int,
        session_id: int | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt_version: str | None,
    ) -> None:
        """Записать один OpenRouter intent-вызов (gpt-4o-mini) как ДВА события.

        Та же модель учёта, что и classifier, но стоимость считается по тарифу
        gpt-4o-mini (дешевле flash). Используется для intent-классификаторов
        (loop continue/done, Q&A finish, confirm analyze/save) и сжатия кастомного
        запроса ветки.
        """
        total_cost = gpt_4o_mini_cost_usd(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.INPUT_TOKENS,
            usage_amount=input_tokens,
            cost_usd=total_cost,
            prompt_version=prompt_version,
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.OUTPUT_TOKENS,
            usage_amount=output_tokens,
            cost_usd=Decimal("0"),
            prompt_version=prompt_version,
        )

    async def record_openrouter_context(
        self,
        *,
        user_id: int,
        session_id: int | None,
        model: str,
        input_tokens: int,
        output_tokens: int,
        prompt_version: str | None,
    ) -> None:
        """Записать один OpenRouter-вызов конвейеров памяти (`openrouter_context_model`).

        Потребители: экстрактор единиц памяти (memory.extract на паузе сессии) и
        компоновка самоописания в онбординге (context.compact). Обе едут на
        google/gemini-3.1-flash-lite (смена модели 2026-08-15) — поэтому и стоимость
        считается по её тарифу, а не по gpt-4o-mini, на котором остались intent'ы.
        Иначе лимит трат юзера занижал бы расход памяти в ~2.3 раза.
        Модель учёта та же: две записи, вся стоимость на input-событии.
        """
        total_cost = gemini_31_flash_lite_cost_usd(
            input_tokens=input_tokens, output_tokens=output_tokens
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.INPUT_TOKENS,
            usage_amount=input_tokens,
            cost_usd=total_cost,
            prompt_version=prompt_version,
        )
        await self._repo.record(
            user_id=user_id,
            session_id=session_id,
            service=LlmService.OPENROUTER,
            model=model,
            usage_metric=UsageMetric.OUTPUT_TOKENS,
            usage_amount=output_tokens,
            cost_usd=Decimal("0"),
            prompt_version=prompt_version,
        )
