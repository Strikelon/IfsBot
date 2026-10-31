"""Классификаторы фичи image questions: дефолты, гарды, составной вход."""
from unittest.mock import AsyncMock

import pytest

from llm.classifier import ClassifierParseError
from llm.intent import (
    IMAGE_ANSWER_MAX_INTENT_CHARS,
    INTENT_IMAGE_ANSWER_VERSION,
    INTENT_IMAGE_OFFER_VERSION,
    classify_image_answer_intent,
    classify_image_offer_intent,
)


def test_versions():
    assert INTENT_IMAGE_OFFER_VERSION == "intent.image_offer.v1"
    assert INTENT_IMAGE_ANSWER_VERSION == "intent.image_answer.v1"


@pytest.mark.asyncio
async def test_offer_default_is_unclear_on_parse_error(monkeypatch):
    monkeypatch.setattr(
        "llm.intent.classify", AsyncMock(side_effect=ClassifierParseError("boom"))
    )
    res = await classify_image_offer_intent(client=AsyncMock(), model="m", user_text="ну эээ")
    assert res.outcome == "unclear"
    assert res.input_tokens == 0 and res.output_tokens == 0


@pytest.mark.asyncio
async def test_offer_default_is_unclear_on_network_error(monkeypatch):
    monkeypatch.setattr("llm.intent.classify", AsyncMock(side_effect=RuntimeError("net")))
    res = await classify_image_offer_intent(client=AsyncMock(), model="m", user_text="давай")
    assert res.outcome == "unclear"


@pytest.mark.asyncio
async def test_offer_passes_reply_to_model(monkeypatch):
    spy = AsyncMock(side_effect=ClassifierParseError("x"))
    monkeypatch.setattr("llm.intent.classify", spy)
    await classify_image_offer_intent(client=AsyncMock(), model="m", user_text="а зачем это?")
    sent = spy.call_args.kwargs["messages"][-1]["content"]
    assert "а зачем это?" in sent


@pytest.mark.asyncio
async def test_answer_default_is_answer_on_failure(monkeypatch):
    monkeypatch.setattr(
        "llm.intent.classify", AsyncMock(side_effect=ClassifierParseError("boom"))
    )
    res = await classify_image_answer_intent(
        client=AsyncMock(), model="m",
        user_text="не знаю даже", bot_question="Как вы относитесь к тиграм?",
    )
    assert res.outcome == "answer"  # контент юзера не теряем


@pytest.mark.asyncio
async def test_answer_length_guard_no_llm_call(monkeypatch):
    """Реплика длиннее 240 симв. → answer БЕЗ вызова LLM (урок _is_real_finish)."""
    spy = AsyncMock()
    monkeypatch.setattr("llm.intent.classify", spy)
    long_text = "я " * (IMAGE_ANSWER_MAX_INTENT_CHARS)  # 2*240 = 480 символов
    res = await classify_image_answer_intent(
        client=AsyncMock(), model="m", user_text=long_text, bot_question="Вопрос?",
    )
    spy.assert_not_awaited()
    assert res.outcome == "answer"
    assert res.input_tokens == 0 and res.output_tokens == 0


@pytest.mark.asyncio
async def test_answer_composed_input_contains_bot_question(monkeypatch):
    spy = AsyncMock(side_effect=ClassifierParseError("x"))
    monkeypatch.setattr("llm.intent.classify", spy)
    await classify_image_answer_intent(
        client=AsyncMock(), model="m",
        user_text="в каком смысле?", bot_question="Как вы относитесь к тиграм?",
    )
    sent = spy.call_args.kwargs["messages"][-1]["content"]
    assert "[вопрос бота] Как вы относитесь к тиграм?" in sent
    assert "[ответ юзера] в каком смысле?" in sent


@pytest.mark.asyncio
async def test_answer_241_chars_exactly_guarded(monkeypatch):
    spy = AsyncMock()
    monkeypatch.setattr("llm.intent.classify", spy)
    res = await classify_image_answer_intent(
        client=AsyncMock(), model="m",
        user_text="х" * (IMAGE_ANSWER_MAX_INTENT_CHARS + 1), bot_question="В?",
    )
    spy.assert_not_awaited()
    assert res.outcome == "answer"
