"""LLM-generator обёртки (Gemini через OpenRouter).

Не знает про БД, не знает про Telegram. Принимает уже собранный messages-список
(см. llm.prompts.quick) и AsyncOpenAI-клиент, нацеленный на OpenRouter.
Возвращает текст + распакованный usage. Cost-учёт делает вызывающий.

Fallback chain: flash → pro → haiku. При transient-ошибке, content_filter или
слишком коротком ответе переходим к следующей модели. При quota_exceeded/permanent
— немедленно финализируем. Per-attempt timeout = 120 s.
"""

from __future__ import annotations

import asyncio
import enum
import time
from dataclasses import dataclass
from typing import Any

import structlog
from openai import AsyncOpenAI

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

# Температура для Quick-генератора
_QUICK_TEMPERATURE = 0.6

# Минимальная длина content, чтобы считаться успешным ответом.
_MIN_CONTENT_LEN = 100

# Таймаут одной попытки (секунды).
_ATTEMPT_TIMEOUT_S = 120.0

# Fallback chain для Quick-генератора.
_QUICK_FALLBACK_CHAIN: list[str] = [
    "google/gemini-3.5-flash",
    "google/gemini-2.5-pro",
    "anthropic/claude-haiku-4.5",
]

# Параметры, специфичные для каждой модели.
# Если модель не в словаре — берём параметры flash.
_MODEL_PARAMS: dict[str, dict[str, Any]] = {
    "google/gemini-3.5-flash": {"max_completion_tokens": 16384},
    "google/gemini-2.5-pro": {"max_completion_tokens": 16384},
    "anthropic/claude-haiku-4.5": {"max_completion_tokens": 4096},
}
_DEFAULT_MODEL_PARAMS = _MODEL_PARAMS["google/gemini-3.5-flash"]


# ---------------------------------------------------------------------------
# Классификация ошибок
# ---------------------------------------------------------------------------


class ErrorCategory(str, enum.Enum):
    """Категория ошибки LLM-вызова.

    TRANSIENT       — попробуй следующую модель в chain.
    CONTENT_FILTER  — у этой модели сработал фильтр; следующая может пропустить.
    QUOTA_EXCEEDED  — бюджет исчерпан, не retry.
    PERMANENT       — наш баг (400 без маркеров фильтра / 401), не retry, critical log.
    """

    TRANSIENT = "transient"
    CONTENT_FILTER = "content_filter"
    QUOTA_EXCEEDED = "quota_exceeded"
    PERMANENT = "permanent"


# Маркеры «провайдер зарубил контент» в тексте ошибки. Gemini отдаёт такую
# блокировку как HTTP 400 (не 403!), поэтому по коду её не отличить от нашего
# кривого запроса – различаем по сообщению. Голого "blocked" в списке нет
# намеренно: под него попадают блокировки аккаунта/биллинга, а это не наш кейс.
_CONTENT_BLOCK_MARKERS: tuple[str, ...] = (
    "prohibited_content",
    "blocked the request",
    "safety",
    "content_policy",
    "policy_violation",
    "content policy",
    "content filter",
    "moderation",
)


def _is_content_block(message: str) -> bool:
    """True, если текст ошибки говорит о срабатывании фильтра контента."""
    return any(kw in message for kw in _CONTENT_BLOCK_MARKERS)


def _classify_error(error_payload: dict[str, Any]) -> ErrorCategory:
    """Классифицировать ошибку по HTTP-коду и сообщению.

    Args:
        error_payload: словарь с полями ``code`` (int) и ``message`` (str).
    """
    code = int(error_payload.get("code") or 0)
    message = str(error_payload.get("message") or "").lower()

    if code == 402:
        return ErrorCategory.QUOTA_EXCEEDED
    if code == 429:
        return ErrorCategory.TRANSIENT
    if code in (502, 503):
        return ErrorCategory.TRANSIENT
    if code == 403:
        # quota/credits → QUOTA_EXCEEDED; content policy → CONTENT_FILTER
        if any(kw in message for kw in ("quota", "credit", "insufficient")):
            return ErrorCategory.QUOTA_EXCEEDED
        if _is_content_block(message):
            return ErrorCategory.CONTENT_FILTER
        # Generic 403 → content filter (gives user chance to rephrase)
        return ErrorCategory.CONTENT_FILTER
    if code == 400:
        # Gemini блокирует запрос фильтром безопасности с кодом 400. Это не наш
        # баг: у модели другого вендора запрос может пройти → пусть цепочка едет.
        if _is_content_block(message):
            return ErrorCategory.CONTENT_FILTER
        return ErrorCategory.PERMANENT
    if code == 401:
        return ErrorCategory.PERMANENT
    # Unknown code: treat as transient (may recover)
    return ErrorCategory.TRANSIENT


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeneratorResult:
    text: str
    input_tokens: int
    output_tokens: int
    error_category: ErrorCategory | None = None  # None = success
    model_used: str | None = None  # which model in chain actually answered; None on chain exhaustion


# ---------------------------------------------------------------------------
# Core fallback chain logic
# ---------------------------------------------------------------------------


async def _call_with_fallback_chain(
    *,
    client: AsyncOpenAI,
    models: list[str],
    messages: list[dict[str, Any]],
    temperature: float,
    log_prefix: str,
) -> tuple[Any | None, ErrorCategory | None, str | None, list[dict[str, Any]]]:
    """Попытать каждую модель из chain последовательно.

    Возвращает ``(response | None, error_category | None, model_used | None, attempts_log)``.
    - response — первый успешный ChatCompletion с достаточно длинным content.
    - error_category — None если успех, иначе категория последней ошибки.
    - model_used — slug модели, которая ответила успешно; None если chain исчерпан.
    - attempts_log — детали каждой попытки для финального лога.

    Останавливается немедленно при QUOTA_EXCEEDED или PERMANENT.
    """
    attempts_log: list[dict[str, Any]] = []
    last_error_category: ErrorCategory | None = None
    # Вендоры, чьи модели зарубили контент фильтром. Внутри одного вендора
    # политика одна – вторая google-модель зарубит ровно так же, только время
    # и деньги. Прыгаем сразу на другого вендора (в цепочке это Anthropic).
    blocked_vendors: set[str] = set()

    total_start = time.monotonic()

    for attempt_n, model in enumerate(models, start=1):
        vendor = model.split("/")[0]
        if vendor in blocked_vendors:
            log.info(
                "generator.attempt.skipped",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                reason="content_filter_same_vendor",
            )
            continue

        model_params = _MODEL_PARAMS.get(model, _DEFAULT_MODEL_PARAMS)
        prompt_len = sum(len(str(m.get("content") or "")) for m in messages)

        log.debug(
            "generator.attempt.started",
            log_prefix=log_prefix,
            model=model,
            attempt_n=attempt_n,
            prompt_len=prompt_len,
        )
        attempt_start = time.monotonic()

        # --- Вызов с per-attempt timeout ---
        response = None
        error_category: ErrorCategory | None = None
        error_code: int | None = None
        error_msg: str | None = None

        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    **model_params,
                ),
                timeout=_ATTEMPT_TIMEOUT_S,
            )
        except TimeoutError:
            error_category = ErrorCategory.TRANSIENT
            error_msg = f"asyncio.TimeoutError after {_ATTEMPT_TIMEOUT_S}s"
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg=error_msg,
                duration=time.monotonic() - attempt_start,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": error_msg,
            })
            last_error_category = error_category
            continue
        except (ConnectionError, OSError) as exc:
            error_category = ErrorCategory.TRANSIENT
            error_msg = str(exc)
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg=error_msg,
                duration=time.monotonic() - attempt_start,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": error_msg,
            })
            last_error_category = error_category
            continue
        except Exception as exc:  # noqa: BLE001
            # Unexpected exception — classify as transient, log with traceback
            error_category = ErrorCategory.TRANSIENT
            error_msg = str(exc)
            log.exception(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg=error_msg,
                duration=time.monotonic() - attempt_start,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": error_msg,
            })
            last_error_category = error_category
            continue

        # --- Разбор response ---
        response_dump = (
            response.model_dump() if hasattr(response, "model_dump") else {}
        )
        provider = response_dump.get("provider")
        duration = time.monotonic() - attempt_start

        # Проверить error-поле в response (OpenRouter error format)
        raw_error = response_dump.get("error") or getattr(response, "error", None)
        if raw_error is not None:
            if isinstance(raw_error, dict):
                error_payload = raw_error
            else:
                error_payload = {
                    "code": getattr(raw_error, "code", 0),
                    "message": getattr(raw_error, "message", ""),
                }
            error_category = _classify_error(error_payload)
            error_code = error_payload.get("code")
            error_msg = error_payload.get("message", "")
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=error_code,
                error_msg=error_msg,
                duration=duration,
                provider=provider,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": error_code,
                "error_msg": error_msg,
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            if error_category == ErrorCategory.CONTENT_FILTER:
                blocked_vendors.add(vendor)
            # Немедленно остановиться при quota/permanent
            if error_category in (ErrorCategory.QUOTA_EXCEEDED, ErrorCategory.PERMANENT):
                log.critical(
                    "generator.fatal_error",
                    log_prefix=log_prefix,
                    error_category=error_category,
                    error_code=error_code,
                    error_msg=error_msg,
                    model=model,
                )
                break
            continue

        # Нет choices
        if not response.choices:
            error_category = ErrorCategory.TRANSIENT
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg="no_choices",
                duration=duration,
                provider=provider,
                response_dump=response_dump,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": "no_choices",
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            continue

        choice = response.choices[0]
        content = choice.message.content
        finish_reason = getattr(choice, "finish_reason", None)

        # finish_reason='content_filter' → content_filter category
        if finish_reason == "content_filter":
            error_category = ErrorCategory.CONTENT_FILTER
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg="content_filter",
                finish_reason=finish_reason,
                duration=duration,
                provider=provider,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": "content_filter",
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            blocked_vendors.add(vendor)
            continue

        # finish_reason='error'/'length' → генерация оборвалась (апстрим-ошибка на
        # полуслове или обрезка по max_tokens). Контент неполный — у quick-разбора
        # не будет хвоста ===QA_OPTIONS=== (вопрос+чипы), юзер застрянет. НЕ принимаем
        # как успех, пробуем следующую модель цепочки (flash→pro→haiku).
        if finish_reason in ("error", "length"):
            error_category = ErrorCategory.TRANSIENT
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg=f"bad_finish_{finish_reason}",
                finish_reason=finish_reason,
                duration=duration,
                provider=provider,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": f"bad_finish_{finish_reason}",
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            continue

        # Пустой content
        if not content:
            error_category = ErrorCategory.TRANSIENT
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg="empty_content",
                finish_reason=finish_reason,
                duration=duration,
                provider=provider,
                response_dump=response_dump,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": error_category,
                "error_code": None,
                "error_msg": "empty_content",
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            continue

        # Min-content-length check
        if len(content) < _MIN_CONTENT_LEN:
            error_category = ErrorCategory.TRANSIENT
            log.warning(
                "generator.attempt.failed",
                log_prefix=log_prefix,
                model=model,
                attempt_n=attempt_n,
                error_category=error_category,
                error_code=None,
                error_msg="content_too_short",
                content_len=len(content),
                duration=duration,
                provider=provider,
            )
            attempts_log.append({
                "model": model,
                "attempt_n": attempt_n,
                "error_category": "content_too_short",
                "error_code": None,
                "error_msg": f"content len {len(content)} < {_MIN_CONTENT_LEN}",
                "duration": duration,
                "provider": provider,
            })
            last_error_category = error_category
            continue

        # ---- Успех! ----
        usage = getattr(response, "usage", None)
        usage_dump = (
            usage.model_dump() if usage and hasattr(usage, "model_dump") else None
        )
        cost_usd = response_dump.get("cost_usd") or response_dump.get("usage", {})
        reasoning_tokens = (
            getattr(usage, "completion_tokens_details", None) and
            getattr(usage.completion_tokens_details, "reasoning_tokens", None)
        ) if usage else None

        log.info(
            "generator.attempt.succeeded",
            log_prefix=log_prefix,
            model=model,
            attempt_n=attempt_n,
            content_len=len(content),
            duration=duration,
            provider=provider,
            cost_usd=cost_usd,
            reasoning_tokens=reasoning_tokens,
            finish_reason=finish_reason,
            usage=usage_dump,
        )
        return response, None, model, attempts_log

    # Цепочка исчерпана
    total_duration = time.monotonic() - total_start
    tried_models = [a.get("model") for a in attempts_log]
    final_category = last_error_category or ErrorCategory.TRANSIENT
    log.error(
        "generator.fallback_chain.exhausted",
        log_prefix=log_prefix,
        tried_models=tried_models,
        total_duration=total_duration,
        final_error_category=final_category,
    )
    return None, final_category, None, attempts_log


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def generate_quick_response(
    *,
    client: AsyncOpenAI,
    messages: list[dict[str, Any]],
    # model param is DEPRECATED — generator now owns the chain internally.
    # Kept for backward compatibility only; ignored if provided.
    model: str | None = None,
) -> GeneratorResult:
    """Вызов OpenRouter generator для Quick analysis.

    Args:
        client: AsyncOpenAI, нацеленный на OpenRouter (base_url=...).
        messages: уже собранный список chat-messages из llm.prompts.quick.
        model: **deprecated**, игнорируется. Цепочка задана в _QUICK_FALLBACK_CHAIN.

    Returns:
        GeneratorResult с text, распакованным usage и error_category.
        error_category=None при успехе; иначе — категория финальной ошибки.
    """
    log.info(
        "generator.quick.start",
        n_messages=len(messages),
        chain=_QUICK_FALLBACK_CHAIN,
    )

    response, error_category, model_used, attempts_log = await _call_with_fallback_chain(
        client=client,
        models=_QUICK_FALLBACK_CHAIN,
        messages=messages,
        temperature=_QUICK_TEMPERATURE,
        log_prefix="quick",
    )

    if response is None:
        return GeneratorResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            error_category=error_category,
        )

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    text = response.choices[0].message.content or ""

    log.info(
        "generator.quick.done",
        text_len=len(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_used=model_used,
    )
    return GeneratorResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_category=None,
        model_used=model_used,
    )


async def generate_deep_response(
    *,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
) -> GeneratorResult:
    """Generator-вызов для Deep-фаз. Параметризован температурой под фазу.

    P2_FOCUS / P3_FEEL / UNBLENDING / P4-подфазы / P5 — temp=0.5.
    P6 — синтез, temp=0.4 (нужна структура).

    Deep использует single-model chain (заданную caller'ом), но с той же
    защитой: fallback chain из [model] (один элемент), те же timeout/logging.
    """
    log.info("generator.deep.start", model=model, n_messages=len(messages))

    response, error_category, model_used, _attempts_log = await _call_with_fallback_chain(
        client=client,
        models=[model],
        messages=messages,
        temperature=temperature,
        log_prefix="deep",
    )

    if response is None:
        return GeneratorResult(
            text="",
            input_tokens=0,
            output_tokens=0,
            error_category=error_category,
        )

    usage = getattr(response, "usage", None)
    input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0) if usage else 0
    output_tokens = int(getattr(usage, "completion_tokens", 0) or 0) if usage else 0
    text = response.choices[0].message.content or ""

    log.info(
        "generator.deep.done",
        model=model,
        text_len=len(text),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_used=model_used,
    )
    return GeneratorResult(
        text=text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        error_category=None,
        model_used=model_used,
    )
