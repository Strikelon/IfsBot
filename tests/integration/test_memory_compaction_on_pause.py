"""Уплотнение памяти на паузе сессии («сухой остаток», 2026-08-16).

Проход идёт ПОСЛЕ экстракции, в том же хуке `run_memory_extraction_on_pause`:
- запускается только когда память перестала влезать в бюджет блока промпта;
- поглощённые единицы гасятся (superseded) со ссылкой на новую плотную запись;
- сбой уплотнения не рушит ни паузу, ни уже записанную экстракцию.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from adapters.telegram.handlers._memory import run_memory_extraction_on_pause
from data.models import LlmUsageEvent
from data.repositories import UserMemoryItemRepository
from llm.memory_compact import (
    COMPACT_SOURCE,
    MEMORY_COMPACT_VERSION,
    CompactGroup,
    MemoryCompactResult,
)
from llm.memory_extract import MemoryExtractResult
from tests.integration.test_memory_on_pause import _make_session_with_qa, _make_user

_EXTRACT_PATCH = "adapters.telegram.handlers._memory.extract_memory_items"
_COMPACT_PATCH = "adapters.telegram.handlers._memory.compact_memory"

_NOOP_EXTRACT = MemoryExtractResult(
    candidates=[], input_tokens=0, output_tokens=0, reject_reason="empty_input"
)


# Виды чередуются: в блоке памяти у каждого вида своя квота строк, и десять записей
# ОДНОГО вида не поместились бы в блок из-за квоты, а не из-за бюджета.
_KINDS = ("fact", "attitude", "person", "pattern", "episode")


async def _fill_memory(
    session, user_id: int, n: int, *, realm: str = "real_life", text_len: int = 120
) -> list:
    """Память, которая ПЕРЕРОСЛА бюджет блока — иначе уплотнение не запускается."""
    repo = UserMemoryItemRepository(session)
    return [
        await repo.add(
            user_id=user_id, kind=_KINDS[i % len(_KINDS)],
            text=f"Разрозненная запись номер {i} " + "ц" * text_len,
            realm=realm, source="qa_answer",
        )
        for i in range(n)
    ]


async def _run_pause(session, user, sess, *, compact_result, budget_chars: int = 1600):
    with (
        patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_NOOP_EXTRACT)),
        patch(_COMPACT_PATCH, new=AsyncMock(return_value=compact_result)) as compact,
    ):
        await run_memory_extraction_on_pause(
            session=session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
            compact_budget_chars=budget_chars,
        )
    return compact


@pytest.mark.asyncio
async def test_compaction_merges_and_supersedes(test_session, schema) -> None:
    """Группа поглощает свои ячейки: одна новая запись, исходные погашены со ссылкой."""
    user = await _make_user(test_session, tg_id=201)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    items = await _fill_memory(test_session, user.id, 15)
    absorbed = [items[0].id, items[1].id, items[2].id]
    result = MemoryCompactResult(
        groups=[CompactGroup(tuple(absorbed), "fact", "Работает на двух работах, доход нестабилен")],
        input_tokens=300, output_tokens=60,
    )
    await _run_pause(test_session, user, sess, compact_result=result)

    repo = UserMemoryItemRepository(test_session)
    active = await repo.list_active(user_id=user.id)
    texts = {i.text for i in active}
    assert "Работает на двух работах, доход нестабилен" in texts
    assert len(active) == 13  # 15 − 3 поглощённых + 1 плотная
    new = next(i for i in active if i.source == COMPACT_SOURCE)
    assert new.source_session_id == sess.id
    assert new.dream_id == sess.dream_id
    for old in items[:3]:
        await test_session.refresh(old)
        assert old.status == "superseded"
        assert old.superseded_by_id == new.id


@pytest.mark.asyncio
async def test_compaction_not_called_while_memory_fits_the_budget(test_session, schema) -> None:
    """Память влезает в блок — модель уплотнения не зовётся вовсе (лишних трат нет)."""
    user = await _make_user(test_session, tg_id=202)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    await _fill_memory(test_session, user.id, 4)
    compact = await _run_pause(
        test_session, user, sess,
        compact_result=MemoryCompactResult(groups=[], input_tokens=0, output_tokens=0),
    )
    compact.assert_not_awaited()


@pytest.mark.asyncio
async def test_compaction_usage_recorded(test_session, schema) -> None:
    """Расход прохода пишется отдельной версией промпта — виден в лимите трат юзера."""
    user = await _make_user(test_session, tg_id=203)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    items = await _fill_memory(test_session, user.id, 15)
    result = MemoryCompactResult(
        groups=[CompactGroup((items[0].id, items[1].id), "fact", "Плотная запись")],
        input_tokens=300, output_tokens=60,
    )
    await _run_pause(test_session, user, sess, compact_result=result)

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert any(e.prompt_version == MEMORY_COMPACT_VERSION for e in events)


@pytest.mark.asyncio
async def test_compaction_keeps_association_realm(test_session, schema) -> None:
    """Плотная запись из ассоциаций остаётся ассоциацией, а не превращается в факт жизни."""
    user = await _make_user(test_session, tg_id=204)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    items = await _fill_memory(test_session, user.id, 15, realm="association")
    result = MemoryCompactResult(
        groups=[CompactGroup((items[0].id, items[1].id), "attitude", "Плотная ассоциация")],
        input_tokens=300, output_tokens=60,
    )
    await _run_pause(test_session, user, sess, compact_result=result)

    repo = UserMemoryItemRepository(test_session)
    new = next(
        i for i in await repo.list_active(user_id=user.id) if i.source == COMPACT_SOURCE
    )
    assert new.realm == "association"


@pytest.mark.asyncio
async def test_compaction_failure_does_not_break_pause(test_session, schema) -> None:
    """Взрыв уплотнения не бросает наружу и не гасит уже накопленную память."""
    user = await _make_user(test_session, tg_id=205)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    await _fill_memory(test_session, user.id, 15)
    with (
        patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_NOOP_EXTRACT)),
        patch(_COMPACT_PATCH, new=AsyncMock(side_effect=RuntimeError("boom"))),
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    active = await UserMemoryItemRepository(test_session).list_active(user_id=user.id)
    assert len(active) == 15


@pytest.mark.asyncio
async def test_compaction_can_be_switched_off(test_session, schema) -> None:
    """Kill-switch: с выключенным уплотнением проход не запускается."""
    user = await _make_user(test_session, tg_id=206)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    await _fill_memory(test_session, user.id, 15)
    with (
        patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_NOOP_EXTRACT)),
        patch(_COMPACT_PATCH, new=AsyncMock()) as compact,
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
            compact_enabled=False,
        )
    compact.assert_not_awaited()
