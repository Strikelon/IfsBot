"""Память: текст обращения для не-сновых типов; сны — байт-в-байт прежний вход."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from data.constants import PRE_QUESTION_MARKER
from domain.entry_types import KIND_SITUATION
from llm.memory_extract import (
    _ALLOWED_SOURCES,
    _build_user_message,
    extract_memory_items,
)


def test_sources_whitelist_extended():
    assert "entry_text" in _ALLOWED_SOURCES
    assert "pre_questions" in _ALLOWED_SOURCES


def test_build_user_message_includes_entry_block_for_situation():
    msg = _build_user_message(
        [(None, "ответ")], [], "переживания", [],
        entry_kind=KIND_SITUATION,
        entry_text="поссорилась с руководителем из-за дедлайна",
    )
    assert "ТЕКСТ ОБРАЩЕНИЯ" in msg
    assert 'source="entry_text"' in msg
    assert "поссорилась с руководителем" in msg


def test_build_user_message_dream_input_byte_identical():
    """Для снов (entry_text="") вход не отличается от вызова без новых kwargs."""
    legacy = _build_user_message([(None, "ответ")], [], "переживания", [])
    with_kwargs = _build_user_message(
        [(None, "ответ")], [], "переживания", [], entry_kind="dream", entry_text=""
    )
    assert legacy == with_kwargs
    assert "ТЕКСТ ОБРАЩЕНИЯ" not in legacy


def test_pre_question_pairs_labeled_pre_questions():
    msg = _build_user_message(
        [(PRE_QUESTION_MARKER + "Что вы почувствовали?", "обиду")],
        [], "", [],
    )
    assert 'source="pre_questions"' in msg


@pytest.mark.asyncio
async def test_real_life_from_entry_text_passes_guards():
    payload = (
        '{"items":[{"action":"add","target_id":null,"kind":"fact",'
        '"text":"Конфликт с руководителем из-за дедлайнов.",'
        '"realm":"real_life","entity_key":null,"source":"entry_text"}]}'
    )
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=payload))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )
    )
    res = await extract_memory_items(
        client=client, model="openai/gpt-4o-mini", existing_items=[],
        user_replies=[(None, "x")], grounded_labels=[], per_session_context="",
        entry_kind=KIND_SITUATION, entry_text="поссорилась с руководителем",
    )
    assert len(res.candidates) == 1
    assert res.candidates[0].source == "entry_text"
    assert res.candidates[0].realm == "real_life"
