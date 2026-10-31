"""intent.entry_type.v1: кнопочные значения, question с answer, дефолт unclear."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm.intent import (
    INTENT_ENTRY_ANSWER_VERSION,
    INTENT_ENTRY_OFFER_VERSION,
    INTENT_ENTRY_TYPE_VERSION,
    classify_entry_answer_intent,
    classify_entry_offer_intent,
    classify_entry_type_intent,
)

_MODEL = "openai/gpt-4o-mini"


def _client(payload: str) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=6),
        )
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome", ["dream", "external_situation", "internal_conflict", "unclear"]
)
async def test_outcomes_pass_through(outcome):
    res = await classify_entry_type_intent(
        client=_client(f'{{"outcome": "{outcome}", "answer": "", "reason": "x"}}'),
        model=_MODEL,
        user_text="мне приснился тигр",
    )
    assert res.outcome == outcome
    assert res.input_tokens == 40 and res.output_tokens == 6


@pytest.mark.asyncio
async def test_question_carries_answer():
    res = await classify_entry_type_intent(
        client=_client(
            '{"outcome": "question", "answer": "Разберу любой из трёх вариантов. '
            'Что выберете?", "reason": "встречный вопрос"}'
        ),
        model=_MODEL,
        user_text="а в чём разница?",
    )
    assert res.outcome == "question"
    assert "Что выберете?" in res.answer


@pytest.mark.asyncio
async def test_network_failure_defaults_to_unclear_zero_tokens():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    res = await classify_entry_type_intent(
        client=client, model=_MODEL, user_text="ну это"
    )
    assert res.outcome == "unclear"
    assert res.input_tokens == 0 and res.output_tokens == 0


def test_version_constant():
    assert INTENT_ENTRY_TYPE_VERSION == "intent.entry_type.v1"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["answer", "decline", "unclear"])
async def test_entry_offer_outcomes_pass_through(outcome):
    res = await classify_entry_offer_intent(
        client=_client(f'{{"outcome": "{outcome}", "answer": "", "reason": "x"}}'),
        model=_MODEL,
        user_text="давай",
    )
    assert res.outcome == outcome


@pytest.mark.asyncio
async def test_entry_offer_question_answer_has_no_dream_words():
    res = await classify_entry_offer_intent(
        client=_client(
            '{"outcome": "question", "answer": "Это займёт пару минут и сделает '
            'разбор точнее. Ответите?", "reason": "встречный вопрос"}'
        ),
        model=_MODEL,
        user_text="а зачем это?",
    )
    assert res.outcome == "question"
    assert "сон" not in res.answer.lower()


@pytest.mark.asyncio
async def test_entry_offer_defaults_to_unclear():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    res = await classify_entry_offer_intent(client=client, model=_MODEL, user_text="эээ")
    assert res.outcome == "unclear"
    assert res.input_tokens == 0 and res.output_tokens == 0


def test_entry_offer_system_prompt_is_dream_free():
    """В промпте не должно быть ни одной формы слова «сон» — иначе ответ
    классификатора (он показывается юзеру дословно) уедет не в ту степь.
    Проверяем по словам, а не по подстроке «сн»: в тексте легально живут
    «снять непонимание» и «сначала»."""
    import re

    from llm.intent import _ENTRY_OFFER_SYSTEM

    dream_words = re.findall(
        r"\b(?:сон|сна|сне|сну|сном|сны|снов|снам|сновидени\w*|приснил\w*)\b",
        _ENTRY_OFFER_SYSTEM.lower(),
    )
    assert dream_words == []
    assert INTENT_ENTRY_OFFER_VERSION == "intent.entry_offer.v1"


@pytest.mark.asyncio
async def test_told_true_when_user_starts_telling():
    res = await classify_entry_type_intent(
        client=_client(
            '{"outcome": "external_situation", "told": true, "answer": "", '
            '"reason": "сразу описывает ссору"}'
        ),
        model=_MODEL,
        user_text="вчера поругалась с мамой из-за денег и до сих пор трясёт",
    )
    assert res.outcome == "external_situation"
    assert res.told is True


@pytest.mark.asyncio
async def test_told_false_for_bare_choice():
    res = await classify_entry_type_intent(
        client=_client('{"outcome": "dream", "told": false, "answer": "", "reason": "выбор"}'),
        model=_MODEL,
        user_text="хочу разобрать сон",
    )
    assert res.told is False


@pytest.mark.asyncio
async def test_told_defaults_false_when_field_missing_or_failed():
    """Старый формат ответа без поля и сетевой сбой — оба дают told=False:
    хуже показать лишнее приглашение рассказать, чем потерять рассказ."""
    res_missing = await classify_entry_type_intent(
        client=_client('{"outcome": "dream", "answer": "", "reason": "x"}'),
        model=_MODEL,
        user_text="сон",
    )
    assert res_missing.told is False

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    res_failed = await classify_entry_type_intent(
        client=client, model=_MODEL, user_text="эээ"
    )
    assert res_failed.outcome == "unclear"
    assert res_failed.told is False


# --- Классификатор ответа на пре-вопрос для НЕ-сновых типов ---


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["answer", "skip"])
async def test_entry_answer_outcomes_pass_through(outcome):
    res = await classify_entry_answer_intent(
        client=_client(f'{{"outcome": "{outcome}", "answer": "", "reason": "x"}}'),
        model=_MODEL,
        user_text="обиду и злость",
        bot_question="Что вы почувствовали в тот момент?",
    )
    assert res.outcome == outcome


@pytest.mark.asyncio
async def test_entry_answer_long_reply_skips_llm():
    """Гард длины как у сновой версии: длинный ответ — всегда answer, без вызова."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    res = await classify_entry_answer_intent(
        client=client, model=_MODEL, user_text="а" * 300,
        bot_question="Что вы почувствовали?",
    )
    assert res.outcome == "answer"
    assert res.input_tokens == 0 and res.output_tokens == 0
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_entry_answer_defaults_to_answer_on_failure():
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    res = await classify_entry_answer_intent(
        client=client, model=_MODEL, user_text="ммм", bot_question="Что почувствовали?"
    )
    assert res.outcome == "answer"  # контент терять нельзя


def test_entry_answer_system_prompt_is_dream_free():
    """Поле answer показывается юзеру ДОСЛОВНО: разбирая конфликт с женой, юзер
    не должен получить переспрос про сон и образы."""
    import re

    from llm.intent import _ENTRY_ANSWER_FEWSHOT, _ENTRY_ANSWER_SYSTEM

    blob = (
        _ENTRY_ANSWER_SYSTEM + " " + " ".join(u + " " + a for u, a in _ENTRY_ANSWER_FEWSHOT)
    ).lower()
    dream_words = re.findall(
        r"\b(?:сон|сна|сне|сну|сном|сны|снов|снам|сновидени\w*|приснил\w*|образ\w*)\b",
        blob,
    )
    assert dream_words == []
    assert INTENT_ENTRY_ANSWER_VERSION == "intent.entry_answer.v1"
