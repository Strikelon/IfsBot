"""Тесты intent-классификаторов (llm/intent.py).

Дешёвая модель (gpt-4o-mini) мокается как AsyncOpenAI-клиент: возвращает
strict-JSON, который classify() парсит. Проверяем маппинг outcome и
безопасные дефолты на сбое.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


def _or_response(content: str, p_tokens: int = 120, c_tokens: int = 8):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(prompt_tokens=p_tokens, completion_tokens=c_tokens),
    )


def _client(content: str) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(return_value=_or_response(content))
    return client


# ===================== LOOP intent (точки 1&2) =====================


@pytest.mark.asyncio
async def test_loop_intent_done_when_model_says_done():
    from llm.intent import classify_loop_intent

    client = _client('{"outcome": "done", "reason": "отказ"}')
    res = await classify_loop_intent(
        client=client, model="openai/gpt-4o-mini", user_text="нет, всё"
    )
    assert res.outcome == "done"
    assert res.input_tokens == 120
    assert res.output_tokens == 8


@pytest.mark.asyncio
async def test_loop_intent_continue_when_model_says_continue():
    from llm.intent import classify_loop_intent

    client = _client('{"outcome": "continue", "reason": "добавляет о себе"}')
    res = await classify_loop_intent(
        client=client, model="openai/gpt-4o-mini", user_text="ещё я работаю врачом"
    )
    assert res.outcome == "continue"


@pytest.mark.asyncio
async def test_loop_intent_defaults_to_continue_on_parse_error():
    """Сломанный JSON → НЕ теряем контент юзера, считаем продолжением."""
    from llm.intent import classify_loop_intent

    client = _client("это не json")
    res = await classify_loop_intent(
        client=client, model="m", user_text="что-то важное о себе"
    )
    assert res.outcome == "continue"
    assert res.input_tokens == 0
    assert res.output_tokens == 0


@pytest.mark.asyncio
async def test_loop_intent_defaults_to_continue_on_network_error():
    from llm.intent import classify_loop_intent

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("timeout"))
    res = await classify_loop_intent(client=client, model="m", user_text="ещё кое-что")
    assert res.outcome == "continue"


@pytest.mark.asyncio
async def test_loop_intent_messages_contain_fewshot_and_user_text():
    from llm.intent import classify_loop_intent

    client = _client('{"outcome": "continue", "reason": "x"}')
    await classify_loop_intent(client=client, model="m", user_text="моя реплика 123")

    messages = client.chat.completions.create.await_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    # есть few-shot (хотя бы одна пара user/assistant до реального ввода)
    assert any(m["role"] == "assistant" for m in messages)
    assert "моя реплика 123" in messages[-1]["content"]


# ===================== LOOP/CONFIRM: встречный вопрос =====================


@pytest.mark.asyncio
async def test_loop_intent_question_returns_answer():
    """Встречный вопрос («а надо?») → outcome=question + текст ответа в answer."""
    from llm.intent import classify_loop_intent

    client = _client(
        '{"outcome": "question", "answer": "Не обязательно — добавляйте, '
        'если есть что.", "reason": "встречный вопрос"}'
    )
    res = await classify_loop_intent(client=client, model="m", user_text="а надо?")
    assert res.outcome == "question"
    assert "Не обязательно" in res.answer


@pytest.mark.asyncio
async def test_loop_intent_answer_empty_for_continue():
    """continue/done не несут answer → пустая строка (не None)."""
    from llm.intent import classify_loop_intent

    client = _client('{"outcome": "continue", "reason": "новая инфо"}')
    res = await classify_loop_intent(client=client, model="m", user_text="ещё я врач")
    assert res.outcome == "continue"
    assert res.answer == ""


@pytest.mark.asyncio
async def test_confirm_intent_question_returns_answer():
    """Встречный вопрос на гейте → outcome=question + answer."""
    from llm.intent import classify_confirm_intent

    client = _client(
        '{"outcome": "question", "answer": "Разбор поможет понять смысл сна.", '
        '"reason": "вопрос про разбор"}'
    )
    res = await classify_confirm_intent(
        client=client, model="m", user_text="а что это даст?"
    )
    assert res.outcome == "question"
    assert "понять смысл" in res.answer


# ===================== QA intent (точка 4) =====================


@pytest.mark.asyncio
async def test_qa_intent_finish_when_model_says_finish():
    from llm.intent import classify_qa_intent

    client = _client('{"outcome": "finish", "reason": "закрывает"}')
    res = await classify_qa_intent(
        client=client, model="m", user_text="спасибо, на сегодня хватит"
    )
    assert res.outcome == "finish"


@pytest.mark.asyncio
async def test_qa_intent_continue_for_comment_not_just_question():
    from llm.intent import classify_qa_intent

    client = _client('{"outcome": "continue", "reason": "комментарий по сну"}')
    res = await classify_qa_intent(
        client=client, model="m", user_text="да, это про маму наверное"
    )
    assert res.outcome == "continue"


@pytest.mark.asyncio
async def test_qa_intent_new_dream():
    from llm.intent import classify_qa_intent

    client = _client('{"outcome": "new_dream", "reason": "хочет другой сон"}')
    res = await classify_qa_intent(
        client=client, model="m", user_text="давай разберём новый сон"
    )
    assert res.outcome == "new_dream"


@pytest.mark.asyncio
async def test_qa_intent_defaults_to_continue_on_error():
    """Ошибочно закрыть сессию хуже — дефолт continue."""
    from llm.intent import classify_qa_intent

    client = _client("garbage")
    res = await classify_qa_intent(client=client, model="m", user_text="ещё вопрос")
    assert res.outcome == "continue"
    assert res.input_tokens == 0


# ===================== CONFIRM intent (точка 5) =====================


@pytest.mark.asyncio
async def test_confirm_intent_analyze():
    from llm.intent import classify_confirm_intent

    client = _client('{"outcome": "analyze", "reason": "хочет разбор"}')
    res = await classify_confirm_intent(
        client=client, model="m", user_text="давай разберём"
    )
    assert res.outcome == "analyze"


@pytest.mark.asyncio
async def test_confirm_intent_save():
    from llm.intent import classify_confirm_intent

    client = _client('{"outcome": "save", "reason": "просто сохранить"}')
    res = await classify_confirm_intent(
        client=client, model="m", user_text="нет, просто сохрани"
    )
    assert res.outcome == "save"


@pytest.mark.asyncio
async def test_confirm_intent_defaults_to_unclear_on_error():
    """При сомнении не угадываем между платным разбором и сохранением."""
    from llm.intent import classify_confirm_intent

    client = _client("not json")
    res = await classify_confirm_intent(client=client, model="m", user_text="ммм")
    assert res.outcome == "unclear"


# ===================== BRANCH compression (точка 3) =====================


@pytest.mark.asyncio
async def test_compress_branch_short_request_passthrough_no_call():
    """Короткий запрос возвращается как есть, без вызова модели."""
    from llm.intent import compress_branch_request

    client = AsyncMock()
    client.chat.completions.create = AsyncMock()
    res = await compress_branch_request(
        client=client, model="m", user_text="хочу понять про маму"
    )
    assert res.text == "хочу понять про маму"
    assert res.input_tokens == 0
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_compress_branch_long_request_is_compressed():
    from llm.intent import BRANCH_REQUEST_MAX_CHARS, compress_branch_request

    long_text = "я хочу " + ("очень длинный запрос про мой сон и жизнь " * 20)
    assert len(long_text) > BRANCH_REQUEST_MAX_CHARS
    client = _client('{"request": "понять связь сна с жизнью"}')
    res = await compress_branch_request(client=client, model="m", user_text=long_text)
    assert res.text == "понять связь сна с жизнью"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_compress_branch_long_request_falls_back_to_truncate_on_error():
    from llm.intent import BRANCH_REQUEST_MAX_CHARS, compress_branch_request

    long_text = "ещё " + ("много текста про сон " * 40)
    assert len(long_text) > BRANCH_REQUEST_MAX_CHARS
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=RuntimeError("boom"))
    res = await compress_branch_request(client=client, model="m", user_text=long_text)
    assert len(res.text) <= BRANCH_REQUEST_MAX_CHARS
    assert res.input_tokens == 0


# ===================== QA intent: контекст вопроса бота =====================


@pytest.mark.asyncio
async def test_qa_intent_passes_bot_question_into_prompt():
    """bot_question прокидывается в промпт: классификатор видит и вопрос, и ответ."""
    from llm.intent import classify_qa_intent

    client = _client('{"outcome": "continue", "reason": "ответ на вопрос бота"}')
    res = await classify_qa_intent(
        client=client, model="openai/gpt-4o-mini",
        user_text="Я 3,5 года в проекте Олега и очень ему благодарен, он помог мне",
        bot_question="Как вы в реальности относитесь к Олегу и его подходу?",
    )
    assert res.outcome == "continue"
    sent = client.chat.completions.create.await_args.kwargs["messages"][-1]["content"]
    assert "вопрос бота" in sent and "Олегу" in sent
    assert "ответ юзера" in sent and "благодарен" in sent


@pytest.mark.asyncio
async def test_qa_intent_without_bot_question_uses_plain_reply():
    """Без bot_question — старый формат: в промпте нет маркеров вопрос/ответ."""
    from llm.intent import classify_qa_intent

    client = _client('{"outcome": "finish", "reason": "явное закрытие"}')
    res = await classify_qa_intent(
        client=client, model="openai/gpt-4o-mini", user_text="спасибо, на сегодня всё"
    )
    assert res.outcome == "finish"
    sent = client.chat.completions.create.await_args.kwargs["messages"][-1]["content"]
    assert "[вопрос бота]" not in sent
