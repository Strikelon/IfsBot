"""Тесты для llm/quick_summarizer.py и llm/prompts/quick_summary.py — опция C (rolling summary)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm.prompts.quick_summary import build_summary_messages
from llm.quick_summarizer import (
    RECENT_KEEP,
    SUMMARY_KEY,
    SUMMARY_THRESHOLD,
    compute_summary,
    should_summarize,
    split_history_for_summary,
)


def test_summary_threshold_and_recent_keep_constants() -> None:
    """Контракт значений — изменение требует обсуждения."""
    assert SUMMARY_THRESHOLD == 8
    assert RECENT_KEEP == 4
    assert SUMMARY_KEY == "quick_qa_summary"


def test_should_summarize_returns_false_below_threshold() -> None:
    for n in range(0, SUMMARY_THRESHOLD):
        assert should_summarize(n) is False


def test_should_summarize_returns_true_at_and_above_threshold() -> None:
    assert should_summarize(SUMMARY_THRESHOLD) is True
    assert should_summarize(SUMMARY_THRESHOLD + 1) is True
    assert should_summarize(100) is True


def test_split_history_keeps_all_when_below_or_equal_recent_keep() -> None:
    pairs = [("u%d" % i, "a%d" % i) for i in range(RECENT_KEEP)]
    to_compress, to_keep = split_history_for_summary(pairs)
    assert to_compress == []
    assert to_keep == pairs


def test_split_history_separates_old_and_recent() -> None:
    # 10 пар: первые 6 → compress, последние 4 → keep
    pairs = [("u%d" % i, "a%d" % i) for i in range(10)]
    to_compress, to_keep = split_history_for_summary(pairs)
    assert len(to_compress) == 10 - RECENT_KEEP
    assert len(to_keep) == RECENT_KEEP
    # Сохранён порядок и непересечение
    assert to_compress == pairs[:6]
    assert to_keep == pairs[6:]


def test_build_summary_messages_structure_without_previous() -> None:
    pairs = [
        ("[initial]", "первый IFS-разбор"),
        ("вопрос про ведро", "ответ про защитника"),
    ]
    msgs = build_summary_messages(pairs)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    user_content = msgs[1]["content"]
    # Initial попадает с особой меткой
    assert "Первый IFS-разбор сна" in user_content
    assert "первый IFS-разбор" in user_content
    # Обычная пара тоже на месте, со своими маркерами
    assert "--- Юзер ---" in user_content
    assert "--- Ассистент ---" in user_content
    assert "вопрос про ведро" in user_content
    assert "ответ про защитника" in user_content


def test_build_summary_messages_includes_previous_summary() -> None:
    """Если есть previous_summary — он попадает в user-payload первым."""
    pairs = [("q", "a")]
    msgs = build_summary_messages(pairs, previous_summary="прошлый summary текст")
    user_content = msgs[1]["content"]
    assert "ПРЕДЫДУЩИЙ SUMMARY" in user_content
    assert "прошлый summary текст" in user_content
    # И сам новый диалог тоже
    assert "ДИАЛОГ" in user_content


def test_build_summary_messages_omits_previous_block_when_none() -> None:
    pairs = [("q", "a")]
    msgs = build_summary_messages(pairs)
    user_content = msgs[1]["content"]
    assert "ПРЕДЫДУЩИЙ SUMMARY" not in user_content


@pytest.mark.asyncio
async def test_compute_summary_calls_openrouter_with_correct_args() -> None:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="сжатое содержание диалога"))
            ],
            usage=SimpleNamespace(prompt_tokens=1200, completion_tokens=180),
        )
    )
    pairs = [("[initial]", "разбор"), ("q1", "a1"), ("q2", "a2")]

    text, in_tok, out_tok = await compute_summary(
        client=client,
        model="google/gemini-2.5-flash",
        qa_pairs_to_compress=pairs,
        previous_summary=None,
    )

    assert text == "сжатое содержание диалога"
    assert in_tok == 1200
    assert out_tok == 180

    call = client.chat.completions.create.await_args
    assert call.kwargs["model"] == "google/gemini-2.5-flash"
    assert call.kwargs["temperature"] == 0.3  # фактологический, низкая
    # extra_body просит cost в usage
    assert call.kwargs["extra_body"] == {"usage": {"include": True}}
    # Messages построены из build_summary_messages — system+user
    msgs = call.kwargs["messages"]
    assert msgs[0]["role"] == "system"
    assert msgs[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_compute_summary_handles_empty_response() -> None:
    """Если Flash вернул empty content — отдаём пустую строку, не падаем."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=None))],
            usage=SimpleNamespace(prompt_tokens=500, completion_tokens=0),
        )
    )
    text, _, _ = await compute_summary(
        client=client,
        model="google/gemini-2.5-flash",
        qa_pairs_to_compress=[("q", "a")],
    )
    assert text == ""
