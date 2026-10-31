from decimal import Decimal

import pytest

from llm.pricing import (
    GEMINI_3_PRO_INPUT_USD_PER_M_TOKENS,
    GEMINI_3_PRO_OUTPUT_USD_PER_M_TOKENS,
    GEMINI_31_FLASH_LITE_INPUT_USD_PER_M_TOKENS,
    GEMINI_31_FLASH_LITE_OUTPUT_USD_PER_M_TOKENS,
    GPT_4O_MINI_INPUT_USD_PER_M_TOKENS,
    GPT_4O_MINI_OUTPUT_USD_PER_M_TOKENS,
    WHISPER_USD_PER_SECOND,
    gemini_3_pro_cost_usd,
    gemini_31_flash_lite_cost_usd,
    gemini_flash_cost_usd,
    gpt_4o_mini_cost_usd,
    whisper_cost_usd,
)


def test_whisper_per_second_constant() -> None:
    # OpenAI Whisper: $0.006 / минута = $0.0001 / секунда
    assert WHISPER_USD_PER_SECOND == Decimal("0.0001")


def test_whisper_cost_usd_basic() -> None:
    # 60 секунд = $0.006
    assert whisper_cost_usd(60) == Decimal("0.006000")


def test_whisper_cost_usd_short() -> None:
    # 7 секунд = $0.0007
    assert whisper_cost_usd(7) == Decimal("0.000700")


def test_whisper_cost_usd_zero() -> None:
    # Граничное значение: 0 секунд допустимо
    assert whisper_cost_usd(0) == Decimal("0.000000")


def test_whisper_cost_usd_rejects_negative() -> None:
    with pytest.raises(ValueError):
        whisper_cost_usd(-1)


def test_gemini_3_pro_pricing_constants_positive() -> None:
    """Sanity: цены положительные и output дороже input (всегда так у Gemini)."""
    assert GEMINI_3_PRO_INPUT_USD_PER_M_TOKENS > Decimal("0")
    assert GEMINI_3_PRO_OUTPUT_USD_PER_M_TOKENS > GEMINI_3_PRO_INPUT_USD_PER_M_TOKENS


def test_gemini_3_pro_cost_usd_basic() -> None:
    # 1000 input + 500 output по дефолтным placeholder-ценам ($2.50/M input, $10/M output):
    # 1000 * 2.50/1_000_000 + 500 * 10/1_000_000 = 0.0025 + 0.005 = 0.0075
    cost = gemini_3_pro_cost_usd(
        input_tokens=1000,
        output_tokens=500,
        input_per_m=Decimal("2.50"),
        output_per_m=Decimal("10.00"),
    )
    assert cost == Decimal("0.007500")


def test_gemini_3_pro_cost_usd_zero_output() -> None:
    cost = gemini_3_pro_cost_usd(
        input_tokens=2000,
        output_tokens=0,
        input_per_m=Decimal("2.50"),
        output_per_m=Decimal("10.00"),
    )
    assert cost == Decimal("0.005000")


def test_gemini_3_pro_cost_usd_rejects_negative() -> None:
    with pytest.raises(ValueError):
        gemini_3_pro_cost_usd(
            input_tokens=-1,
            output_tokens=0,
            input_per_m=Decimal("2.50"),
            output_per_m=Decimal("10.00"),
        )
    with pytest.raises(ValueError):
        gemini_3_pro_cost_usd(
            input_tokens=0,
            output_tokens=-1,
            input_per_m=Decimal("2.50"),
            output_per_m=Decimal("10.00"),
        )


def test_gemini_flash_cost_zero_tokens_is_zero() -> None:
    assert gemini_flash_cost_usd(input_tokens=0, output_tokens=0) == Decimal("0.000000")


def test_gemini_flash_cost_for_default_pricing() -> None:
    """1M input + 1M output = $0.30 + $2.50 = $2.80."""
    cost = gemini_flash_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == Decimal("2.800000")


def test_gemini_flash_cost_partial_tokens() -> None:
    """500 input × $0.30/M + 200 output × $2.50/M = 0.00015 + 0.0005 = 0.00065."""
    cost = gemini_flash_cost_usd(input_tokens=500, output_tokens=200)
    assert cost == Decimal("0.000650")


def test_gemini_flash_cost_negative_raises() -> None:
    with pytest.raises(ValueError):
        gemini_flash_cost_usd(input_tokens=-1, output_tokens=0)
    with pytest.raises(ValueError):
        gemini_flash_cost_usd(input_tokens=0, output_tokens=-5)


def test_gemini_flash_cost_override_pricing() -> None:
    """Можно перезаписать цены через kwargs (для будущих тарифных изменений)."""
    cost = gemini_flash_cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_per_m=Decimal("0.50"),
        output_per_m=Decimal("3.00"),
    )
    assert cost == Decimal("3.500000")


def test_gpt_4o_mini_pricing_constants() -> None:
    """gpt-4o-mini: $0.15/M input, $0.60/M output (сверено openai.com/api/pricing)."""
    assert GPT_4O_MINI_INPUT_USD_PER_M_TOKENS == Decimal("0.15")
    assert GPT_4O_MINI_OUTPUT_USD_PER_M_TOKENS == Decimal("0.60")


def test_gpt_4o_mini_cost_for_default_pricing() -> None:
    """1M input + 1M output = $0.15 + $0.60 = $0.75."""
    cost = gpt_4o_mini_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == Decimal("0.750000")


def test_gpt_4o_mini_cost_partial_tokens() -> None:
    """500 input × $0.15/M + 20 output × $0.60/M = 0.000075 + 0.000012 = 0.000087."""
    cost = gpt_4o_mini_cost_usd(input_tokens=500, output_tokens=20)
    assert cost == Decimal("0.000087")


def test_gpt_4o_mini_cost_zero_tokens_is_zero() -> None:
    assert gpt_4o_mini_cost_usd(input_tokens=0, output_tokens=0) == Decimal("0.000000")


def test_gpt_4o_mini_cost_negative_raises() -> None:
    with pytest.raises(ValueError):
        gpt_4o_mini_cost_usd(input_tokens=-1, output_tokens=0)
    with pytest.raises(ValueError):
        gpt_4o_mini_cost_usd(input_tokens=0, output_tokens=-5)


def test_gemini_31_flash_lite_pricing_constants() -> None:
    """google/gemini-3.1-flash-lite: $0.25/M input, $1.50/M output (прайс OpenRouter)."""
    assert GEMINI_31_FLASH_LITE_INPUT_USD_PER_M_TOKENS == Decimal("0.25")
    assert GEMINI_31_FLASH_LITE_OUTPUT_USD_PER_M_TOKENS == Decimal("1.50")


def test_gemini_31_flash_lite_cost_for_default_pricing() -> None:
    """1M input + 1M output = $0.25 + $1.50 = $1.75."""
    cost = gemini_31_flash_lite_cost_usd(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == Decimal("1.750000")


def test_gemini_31_flash_lite_cost_partial_tokens() -> None:
    """5000 input × $0.25/M + 350 output × $1.50/M = 0.00125 + 0.000525 = 0.001775."""
    cost = gemini_31_flash_lite_cost_usd(input_tokens=5000, output_tokens=350)
    assert cost == Decimal("0.001775")


def test_gemini_31_flash_lite_cost_zero_tokens_is_zero() -> None:
    assert gemini_31_flash_lite_cost_usd(input_tokens=0, output_tokens=0) == Decimal(
        "0.000000"
    )


def test_gemini_31_flash_lite_cost_negative_raises() -> None:
    with pytest.raises(ValueError):
        gemini_31_flash_lite_cost_usd(input_tokens=-1, output_tokens=0)
    with pytest.raises(ValueError):
        gemini_31_flash_lite_cost_usd(input_tokens=0, output_tokens=-5)


def test_gemini_31_flash_lite_cost_override_pricing() -> None:
    """Ставки перетираются через kwargs — как у остальных расчётов."""
    cost = gemini_31_flash_lite_cost_usd(
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        input_per_m=Decimal("0.50"),
        output_per_m=Decimal("3.00"),
    )
    assert cost == Decimal("3.500000")


def test_gemini_31_flash_lite_is_dearer_than_gpt_4o_mini() -> None:
    """Смена модели памяти дороже старой ставки — лимит трат должен это видеть."""
    same_tokens = {"input_tokens": 10_000, "output_tokens": 700}
    assert gemini_31_flash_lite_cost_usd(**same_tokens) > gpt_4o_mini_cost_usd(
        **same_tokens
    )
