"""Прайс-лист внешних AI-сервисов.

Цены сверять с https://openai.com/api/pricing и https://openrouter.ai/models.
Менять при изменении тарифов. В Phase 1 только Whisper; в Phase 2+ добавим
Gemini Pro/Flash через OpenRouter.
"""

from decimal import ROUND_HALF_UP, Decimal

# OpenAI Whisper: $0.006 / минута, биллится поминутно (на счёт идёт посекундно)
WHISPER_USD_PER_SECOND: Decimal = Decimal("0.0001")


def whisper_cost_usd(audio_seconds: int) -> Decimal:
    """Стоимость распознавания аудио в долларах. Округление до 6 знаков (как cost_usd в БД)."""
    if audio_seconds < 0:
        raise ValueError(f"audio_seconds must be >= 0, got {audio_seconds}")
    cost = Decimal(audio_seconds) * WHISPER_USD_PER_SECOND
    return cost.quantize(
        Decimal("0.000001"), rounding=ROUND_HALF_UP
    )  # matches Numeric(10,6) in models.py


# OpenRouter google/gemini-2.5-pro pricing (текущий generator).
# Сверено с https://openrouter.ai/api/v1/models 2026-05-19: $1.25 / $10.00 per 1M.
# Имя константы GEMINI_3_PRO_* осталось от Phase 2 (раньше был gemini-3.1-pro-preview);
# фактическая модель задаётся через .env (OPENROUTER_GENERATOR_MODEL). При смене модели —
# обновлять числа здесь под актуальный тариф OpenRouter.
GEMINI_3_PRO_INPUT_USD_PER_M_TOKENS: Decimal = Decimal("1.25")
GEMINI_3_PRO_OUTPUT_USD_PER_M_TOKENS: Decimal = Decimal("10.00")


def gemini_3_pro_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_m: Decimal = GEMINI_3_PRO_INPUT_USD_PER_M_TOKENS,
    output_per_m: Decimal = GEMINI_3_PRO_OUTPUT_USD_PER_M_TOKENS,
) -> Decimal:
    """Полная стоимость одного OpenRouter-вызова к Gemini 3 Pro.

    Считаем суммарную стоимость, потому что в БД у нас одна цифра на одно
    «событие». UsageRecorder создаст два события (INPUT_TOKENS, OUTPUT_TOKENS),
    но cost_usd делит между ними не симметрично — пишет ВСЮ стоимость на input-
    событие, output-событие пишет с cost_usd=0. См. UsageRecorder в следующей задаче.

    Args:
        input_tokens: usage.prompt_tokens от OpenRouter (>= 0).
        output_tokens: usage.completion_tokens от OpenRouter (>= 0).
        input_per_m: цена за миллион входных токенов (Decimal). Для тестов перетираем.
        output_per_m: цена за миллион выходных токенов.

    Returns:
        Decimal с 6 знаками после запятой, соответствует Numeric(10,6).
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"tokens must be >= 0, got input={input_tokens}, output={output_tokens}"
        )
    cost = (
        Decimal(input_tokens) * input_per_m + Decimal(output_tokens) * output_per_m
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# OpenRouter openai/gpt-4o-mini pricing — дешёвый intent-классификатор.
# Сверено с https://openai.com/api/pricing: $0.15 / $0.60 per 1M.
GPT_4O_MINI_INPUT_USD_PER_M_TOKENS: Decimal = Decimal("0.15")
GPT_4O_MINI_OUTPUT_USD_PER_M_TOKENS: Decimal = Decimal("0.60")


def gpt_4o_mini_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_m: Decimal = GPT_4O_MINI_INPUT_USD_PER_M_TOKENS,
    output_per_m: Decimal = GPT_4O_MINI_OUTPUT_USD_PER_M_TOKENS,
) -> Decimal:
    """Полная стоимость одного OpenRouter-вызова к gpt-4o-mini (intent-классификатор).

    Та же модель учёта, что и у gemini_flash_cost_usd: суммарная стоимость идёт
    на input-событие, output-событие пишется с cost_usd=0.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"tokens must be >= 0, got input={input_tokens}, output={output_tokens}"
        )
    cost = (
        Decimal(input_tokens) * input_per_m + Decimal(output_tokens) * output_per_m
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# OpenRouter google/gemini-3.1-flash-lite pricing — конвейеры памяти
# (экстрактор memory.extract + компоновка самоописания context.compact).
# Прайс OpenRouter на 2026-08-15: $0.25 / $1.50 per 1M.
GEMINI_31_FLASH_LITE_INPUT_USD_PER_M_TOKENS: Decimal = Decimal("0.25")
GEMINI_31_FLASH_LITE_OUTPUT_USD_PER_M_TOKENS: Decimal = Decimal("1.50")


def gemini_31_flash_lite_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_m: Decimal = GEMINI_31_FLASH_LITE_INPUT_USD_PER_M_TOKENS,
    output_per_m: Decimal = GEMINI_31_FLASH_LITE_OUTPUT_USD_PER_M_TOKENS,
) -> Decimal:
    """Полная стоимость одного OpenRouter-вызова к gemini-3.1-flash-lite (память).

    Та же модель учёта, что и у остальных: суммарная стоимость идёт на
    input-событие, output-событие пишется с cost_usd=0.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"tokens must be >= 0, got input={input_tokens}, output={output_tokens}"
        )
    cost = (
        Decimal(input_tokens) * input_per_m + Decimal(output_tokens) * output_per_m
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)


# OpenRouter google/gemini-2.5-flash pricing.
# Сверено с https://openrouter.ai/api/v1/models 2026-05-19: $0.30 / $2.50 per 1M.
# Used for: P2_FIND classifier, SELF_GATE classifier, soft Self-check classifier (Phase 3+).
GEMINI_FLASH_INPUT_USD_PER_M_TOKENS: Decimal = Decimal("0.30")
GEMINI_FLASH_OUTPUT_USD_PER_M_TOKENS: Decimal = Decimal("2.50")


def gemini_flash_cost_usd(
    *,
    input_tokens: int,
    output_tokens: int,
    input_per_m: Decimal = GEMINI_FLASH_INPUT_USD_PER_M_TOKENS,
    output_per_m: Decimal = GEMINI_FLASH_OUTPUT_USD_PER_M_TOKENS,
) -> Decimal:
    """Полная стоимость одного OpenRouter-вызова к Gemini Flash classifier'у.

    Та же модель учёта, что и для Pro generator'а (см. gemini_3_pro_cost_usd):
    UsageRecorder пишет ДВА события на один LLM-вызов (INPUT_TOKENS + OUTPUT_TOKENS).
    Эта функция возвращает СУММАРНУЮ стоимость; вся стоимость идёт на input-событие,
    output-событие пишется с cost_usd=0. Так сумма по cost_usd за период == факт.

    Args:
        input_tokens: usage.prompt_tokens (>= 0).
        output_tokens: usage.completion_tokens (>= 0).
        input_per_m: цена за миллион входных токенов (override для тестов).
        output_per_m: цена за миллион выходных токенов.

    Returns:
        Decimal с 6 знаками после запятой.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            f"tokens must be >= 0, got input={input_tokens}, output={output_tokens}"
        )
    cost = (
        Decimal(input_tokens) * input_per_m + Decimal(output_tokens) * output_per_m
    ) / Decimal("1000000")
    return cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
