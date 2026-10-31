"""Strict-JSON classifier wrapper для OpenRouter Gemini Flash.

Принимает Pydantic-схему ожидаемого результата. Парсит ответ модели (с
вырезанием ```json-фенсов на всякий случай), валидирует, возвращает инстанс
схемы + usage. Любая ошибка парсинга/валидации — ClassifierParseError.

Не знает про БД и Telegram. Не пишет UsageRecorder — это делает вызывающий.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

log = structlog.get_logger(__name__)

# Дефолтные параметры classifier'а: низкая температура (нам нужна детерминированность),
# короткий max_tokens (JSON-ответы не длинные). Потолок переопределяется на вызов
# (max_tokens=...): у экстрактора памяти ответ длиннее — два списка вместо одного,
# а обрыв JSON по потолку теряет ВСЮ экстракцию сессии целиком.
_CLASSIFIER_MAX_TOKENS = 600
_CLASSIFIER_TEMPERATURE = 0.0


class ClassifierParseError(RuntimeError):
    """Поднимается, когда Flash вернул не-JSON, пустые choices или JSON не подходит под схему."""


@dataclass(frozen=True)
class ClassifierResult[T: BaseModel]:
    parsed: T
    input_tokens: int
    output_tokens: int


# Полная-строка anchor (^...$) намеренная: OpenRouter с response_format=json_object
# не вернёт прозу вокруг JSON, так что fence-обёртка возможна только как bare-fence.
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?(.+?)\n?```\s*$", re.DOTALL)


def _strip_fences(text: str) -> str:
    """Снять ```json ... ``` обёртку, если есть. Иначе вернуть как есть."""
    text = text.strip()
    m = _FENCE_RE.match(text)
    return m.group(1).strip() if m else text


async def classify[T: BaseModel](
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    schema: type[T],
    max_tokens: int | None = None,
) -> ClassifierResult[T]:
    """Вызов OpenRouter Flash + строгий парс JSON в схему.

    Args:
        client: AsyncOpenAI, нацеленный на OpenRouter (base_url=...).
        model: slug Flash-модели.
        messages: уже собранный messages-список (system + few-shot + user).
        schema: Pydantic-класс ожидаемого результата.
        max_tokens: потолок вывода; None — дефолтный _CLASSIFIER_MAX_TOKENS.
            Поднимать там, где ожидаемый JSON длиннее короткого вердикта.

    Returns:
        ClassifierResult с инстансом схемы и usage.

    Raises:
        ClassifierParseError: модель вернула не-JSON / невалидную структуру / пустой ответ.
    """
    cap = max_tokens if max_tokens is not None else _CLASSIFIER_MAX_TOKENS
    log.info("classifier.start", model=model, n_messages=len(messages), max_tokens=cap)

    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=cap,
        temperature=_CLASSIFIER_TEMPERATURE,
        response_format={"type": "json_object"},
    )

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0

    if not response.choices:
        log.warning("classifier.no_choices", model=model)
        raise ClassifierParseError(f"{model}: empty choices")

    raw_text = response.choices[0].message.content or ""
    if not raw_text.strip():
        raise ClassifierParseError(f"{model}: empty content")

    cleaned = _strip_fences(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        log.warning("classifier.json_decode_failed", model=model, raw_preview=cleaned[:200])
        raise ClassifierParseError(f"{model}: JSONDecodeError {exc}") from exc

    try:
        parsed = schema.model_validate(payload)
    except ValidationError as exc:
        log.warning("classifier.schema_validation_failed", model=model, errors=str(exc))
        raise ClassifierParseError(f"{model}: schema mismatch: {exc}") from exc

    log.info(
        "classifier.done",
        model=model,
        schema=schema.__name__,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return ClassifierResult(parsed=parsed, input_tokens=input_tokens, output_tokens=output_tokens)
