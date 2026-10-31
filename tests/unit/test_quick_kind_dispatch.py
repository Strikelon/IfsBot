"""Диспетчеризация quick.py по kind: генератор, kind-резолвер, карточка."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from adapters.telegram.handlers.quick import _entry_kind_for_session, _run_generator
from data.models import RequestBranch, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import KIND_DREAM, KIND_SITUATION
from llm.prompts.quick import QuickContext


def _ctx() -> QuickContext:
    return QuickContext(
        user_display_name="Катя",
        user_life_context="меняю работу",
        per_session_context="боюсь подвести",
        request_branch=RequestBranch.UNCLEAR,
        dream_transcription="поссорилась с руководителем",
    )


@pytest.mark.asyncio
async def test_run_generator_dispatches_situation_prompts():
    captured = {}

    async def _fake_generate(*, client, messages):
        captured["messages"] = messages
        return AsyncMock(text="ok")

    with patch(
        "adapters.telegram.handlers.quick.generate_quick_response", new=_fake_generate
    ):
        await _run_generator(
            ctx=_ctx(), qa_history=[], user_followup=None,
            openrouter_client=AsyncMock(), kind=KIND_SITUATION,
        )
    user_block = captured["messages"][1]["content"]
    assert "ВОЛНУЮЩАЯ СИТУАЦИЯ" in user_block
    assert "СОН:" not in user_block


@pytest.mark.asyncio
async def test_run_generator_default_kind_is_dream():
    captured = {}

    async def _fake_generate(*, client, messages):
        captured["messages"] = messages
        return AsyncMock(text="ok")

    with patch(
        "adapters.telegram.handlers.quick.generate_quick_response", new=_fake_generate
    ):
        await _run_generator(
            ctx=_ctx(), qa_history=[], user_followup=None, openrouter_client=AsyncMock(),
        )
    assert "СОН:" in captured["messages"][1]["content"]


@pytest.mark.asyncio
async def test_entry_kind_for_session(test_session):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=9)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC),
        transcription="ссора", kind=KIND_SITUATION,
    )
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    assert await _entry_kind_for_session(
        test_session, user_id=user.id, session_id=sess.id
    ) == KIND_SITUATION
    # несуществующая сессия → безопасный дефолт dream
    assert await _entry_kind_for_session(
        test_session, user_id=user.id, session_id=999_999
    ) == KIND_DREAM
