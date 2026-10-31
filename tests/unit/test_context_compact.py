"""Компоновка самоописания в своём модуле (Память v2, Task 7).

context.compact.v1 переезжает из удалённой «живой памяти» БЕЗ изменения текста промпта
(версия не бампается). Защиты те же: no-op под порогом, режект подозрительного
сжатия, no-op при любом сбое.
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from llm.context_compact import (
    COMPACT_TRIGGER_CHARS,
    CONTEXT_COMPACT_VERSION,
    compact_life_context,
)


@dataclass(frozen=True)
class _FakeClassifierResult:
    parsed: object
    input_tokens: int = 9
    output_tokens: int = 4


def _mock_classify(monkeypatch: pytest.MonkeyPatch, *, compacted: str) -> AsyncMock:
    from llm.context_compact import _Compacted

    spy = AsyncMock(
        return_value=_FakeClassifierResult(parsed=_Compacted(compacted=compacted))
    )
    monkeypatch.setattr("llm.context_compact.classify", spy)
    return spy


def test_version_not_bumped():
    """Переезд без изменения текста промпта — версия остаётся v1."""
    assert CONTEXT_COMPACT_VERSION == "context.compact.v1"


@pytest.mark.asyncio
async def test_under_threshold_noop():
    res = await compact_life_context(client=object(), model="m", text="короткий текст")
    assert res.changed is False
    assert res.reject_reason == "under_threshold"


@pytest.mark.asyncio
async def test_compacts_long_text(monkeypatch):
    long_text = "факт. " * 300  # 1800 символов > порога 1200
    assert len(long_text) > COMPACT_TRIGGER_CHARS
    _mock_classify(monkeypatch, compacted="сжатые факты " * 40)  # 520 симв.: > 25% и > 80
    res = await compact_life_context(client=object(), model="m", text=long_text)
    assert res.changed is True
    assert res.new_text is not None
    assert len(res.new_text) < len(long_text)


@pytest.mark.asyncio
async def test_suspicious_shrink_rejected(monkeypatch):
    long_text = "важный факт. " * 200
    _mock_classify(monkeypatch, compacted="х")
    res = await compact_life_context(client=object(), model="m", text=long_text)
    assert res.changed is False
    assert res.reject_reason == "suspicious_shrink"
