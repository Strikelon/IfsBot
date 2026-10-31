"""Юнит-тесты уплотнения памяти («сухой остаток», 2026-08-16).

Дизайн как у test_memory_extract: мокаем llm.memory_compact.classify, подсовываем
сырой parsed, проверяем кодовые защиты. Без БД и без реального LLM.

Отдельно проверяется отбор `select_for_compaction` — детерминированное правило
«пора ли уплотнять», единственное место, где решается частота прохода.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from llm.memory_compact import (
    MEMORY_COMPACT_VERSION,
    CompactItemView,
    compact_memory,
    select_for_compaction,
)


@dataclass(frozen=True)
class _FakeClassifierResult:
    parsed: object
    input_tokens: int = 200
    output_tokens: int = 40


def _mock_classify(monkeypatch: pytest.MonkeyPatch, *, groups: list[dict]) -> AsyncMock:
    from llm.memory_compact import _Compact

    parsed = _Compact.model_validate({"groups": groups})
    spy = AsyncMock(return_value=_FakeClassifierResult(parsed=parsed))
    monkeypatch.setattr("llm.memory_compact.classify", spy)
    return spy


def _views(n: int, *, start_id: int = 1, kind: str = "fact") -> list[CompactItemView]:
    return [
        CompactItemView(id=start_id + i, kind=kind, text=f"Факт номер {start_id + i}",
                        source="qa_answer")
        for i in range(n)
    ]


# Виды чередуются: в блоке памяти у каждого вида своя квота строк, и десять
# записей ОДНОГО вида в него не поместятся из-за квоты, а не из-за бюджета.
_MIXED_KINDS = ("fact", "attitude", "person", "pattern", "episode")


def _stored(
    n: int, *, start_id: int = 1, kind: str | None = None, source: str = "qa_answer",
    text_len: int = 20,
) -> list[SimpleNamespace]:
    """Единицы памяти как их отдаёт репозиторий (для отбора нужен рендер блока)."""
    return [
        SimpleNamespace(
            id=start_id + i, kind=kind or _MIXED_KINDS[i % len(_MIXED_KINDS)],
            source=source, status="active",
            text=f"Запись {start_id + i} " + "ц" * text_len,
            entity_key=None, realm="real_life", confidence=1,
            last_confirmed_at=datetime(2026, 8, 16, tzinfo=UTC) + timedelta(seconds=i),
        )
        for i in range(n)
    ]


async def _run(items=None, **overrides):
    kwargs = dict(client=object(), model="m", items=items if items is not None else _views(4))
    kwargs.update(overrides)
    return await compact_memory(**kwargs)


def test_version_constant():
    assert MEMORY_COMPACT_VERSION == "memory.compact.v1"


# === Отбор: когда вообще запускать проход ===


def test_select_waits_until_memory_outgrows_the_block_budget():
    """Триггер — не счётчик ячеек, а лимит блока памяти.

    Пока вся память влезает в бюджет промпта, уплотнять нечего: юзер ничего не теряет,
    а вызов модели стоил бы денег на каждой паузе. Как только память перестала
    помещаться — часть фактов в разбор уже не попадает, и пора уплотнять.
    """
    from llm.memory_select import (
        MEMORY_BLOCK_BUDGET,
        build_memory_block,
        rendered_item_count,
    )

    small = _stored(4)
    assert len(build_memory_block(small, budget_chars=10**6)) <= MEMORY_BLOCK_BUDGET
    assert select_for_compaction(small) == []

    # Виды подобраны так, чтобы никого не срезала квота: причина ровно одна —
    # длина. Иначе тест проверял бы второй повод (см. тест про квоты ниже).
    big = [
        *_stored(3, start_id=1, kind="fact", text_len=200),
        *_stored(3, start_id=10, kind="attitude", text_len=200),
        *_stored(3, start_id=20, kind="person", text_len=200),
        *_stored(3, start_id=30, kind="pattern", text_len=200),
    ]
    full = build_memory_block(big, budget_chars=10**6)
    assert rendered_item_count(full) == len(big), "квоты тут ни при чём — режет бюджет"
    assert len(full) > MEMORY_BLOCK_BUDGET
    assert select_for_compaction(big)


def test_select_triggers_when_quotas_hide_items():
    """Второй повод: часть записей не влезла в блок.

    Блок режется не только бюджетом, но и квотами по видам (fact — 6 строк). Десять
    фактов занимают ~700 символов, до лимита по символам не дотягивают, но четыре из них в
    разбор не попадут никогда. Это тоже «память переросла блок» — пора уплотнять.
    """
    from llm.memory_select import MEMORY_BLOCK_BUDGET, build_memory_block

    items = _stored(10, kind="fact")
    block = build_memory_block(items, budget_chars=10**6)
    assert len(block) < MEMORY_BLOCK_BUDGET, "бюджет тут ни при чём — режет квота"
    assert len(select_for_compaction(items)) == 10


def test_select_skips_core_and_parts():
    """Ядро (одна ячейка на категорию) и стороны (матчатся по entity_key) не трогаем."""
    items = [
        *_stored(15, text_len=120),
        *_stored(1, start_id=100, kind="core"),
        *_stored(1, start_id=101, kind="part"),
    ]
    selected = select_for_compaction(items)
    assert {i.kind for i in selected} <= set(_MIXED_KINDS)
    assert len(selected) == 15


def test_select_requires_new_items_since_last_compaction():
    """Проход не повторяется на каждой паузе: нужен приток новых ячеек.

    Иначе юзер, чья память осталась за лимитом (модель отказалась что-то сливать),
    оплачивал бы вызов уплотнения после каждого сна впустую.
    """
    old = _stored(15, text_len=120)
    compacted = _stored(1, start_id=50, source="compact")
    fresh_3 = _stored(3, start_id=51)
    assert select_for_compaction([*old, *compacted, *fresh_3], min_new=5) == []
    fresh_6 = _stored(6, start_id=51)
    assert select_for_compaction([*old, *compacted, *fresh_6], min_new=5)


def test_select_counts_previous_compaction_result_as_material():
    """Уплотнённая запись сама участвует в следующем уплотнении (иначе память
    расслаивается на «сухие остатки» и новьё, которые никогда не сойдутся)."""
    compacted = _stored(1, start_id=50, source="compact")
    selected = select_for_compaction([*compacted, *_stored(15, start_id=51, text_len=120)])
    assert 50 in {i.id for i in selected}


def test_select_ignores_superseded_items():
    """Погашенные единицы в промпт не идут — и в уплотнение тоже."""
    items = _stored(15, text_len=120)
    for i in items:
        i.status = "superseded"
    assert select_for_compaction(items) == []


# === Кодовые защиты вывода модели ===


@pytest.mark.asyncio
async def test_groups_parsed(monkeypatch):
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2, 3], "kind": "fact", "text": "Работает на двух работах"},
    ])
    res = await _run()
    assert len(res.groups) == 1
    assert res.groups[0].source_ids == (1, 2, 3)
    assert res.groups[0].kind == "fact"
    assert res.input_tokens == 200


@pytest.mark.asyncio
async def test_single_source_group_dropped(monkeypatch):
    """Группа из одной записи — не уплотнение, а переписывание чужого текста."""
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1], "kind": "fact", "text": "Работает на двух работах"},
    ])
    res = await _run()
    assert res.groups == []


@pytest.mark.asyncio
async def test_unknown_ids_dropped(monkeypatch):
    """Номера не из входа отбрасываются; если валидных осталось меньше двух — вся группа."""
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 999], "kind": "fact", "text": "Только одна настоящая ячейка"},
        {"source_ids": [2, 3, 777], "kind": "fact", "text": "Две настоящие ячейки"},
    ])
    res = await _run()
    assert len(res.groups) == 1
    assert res.groups[0].source_ids == (2, 3)


@pytest.mark.asyncio
async def test_id_used_by_one_group_only(monkeypatch):
    """Одна ячейка не может уехать в две уплотнённые записи — иначе она задвоится."""
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "fact", "text": "Первая тема, собранная из двух ячеек"},
        {"source_ids": [2, 3], "kind": "fact", "text": "Вторая тема, собранная из двух ячеек"},
    ])
    res = await _run()
    assert [g.source_ids for g in res.groups] == [(1, 2)]


@pytest.mark.asyncio
async def test_text_depersonalized_and_guarded(monkeypatch):
    """Текст уплотнённой записи проходит те же защиты, что и текст экстрактора."""
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "fact", "text": "Юзер работает на двух работах"},
        {"source_ids": [3, 4], "kind": "fact", "text": "x" * 501},
    ])
    res = await _run()
    assert [g.text for g in res.groups] == ["Работает на двух работах"]


@pytest.mark.asyncio
async def test_dream_content_rejected(monkeypatch):
    """Содержимое сна в уплотнённой записи режется так же, как в экстракторе."""
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "episode", "text": "Во сне был вампир на чердаке"},
    ])
    res = await _run()
    assert res.groups == []


@pytest.mark.asyncio
async def test_unknown_kind_falls_back_to_fact(monkeypatch):
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "core", "text": "Работает на двух работах"},
    ])
    res = await _run()
    assert res.groups[0].kind == "fact"


def test_system_prompt_forbids_mixing_themes():
    """Прод-прогон 2026-08-16: работа и семья слипались в одну запись."""
    from llm.memory_compact import _SYSTEM

    assert "РАЗНЫЕ ТЕМЫ НЕ СМЕШИВАЙ" in _SYSTEM


@pytest.mark.asyncio
async def test_group_cannot_absorb_two_compacted_records(monkeypatch):
    """Два «сухих остатка» в одной группе не сливаются — иначе идёт переуплотнение.

    Прод-прогон 2026-08-16: у самого активного юзера уплотнённая запись
    поглощалась снова и снова (6 проходов подряд), каждый раз теряя детали, —
    так из памяти исчез целый важный сюжет.
    """
    items = [
        CompactItemView(id=1, kind="fact", text="Сухой остаток про работу", source="compact"),
        CompactItemView(id=2, kind="fact", text="Сухой остаток про семью", source="compact"),
        CompactItemView(id=3, kind="fact", text="Свежая ячейка", source="qa_answer"),
    ]
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "fact", "text": "Работа и семья в одной записи"},
        {"source_ids": [1, 3], "kind": "fact",
         "text": "Сухой остаток про работу, дополненный свежей ячейкой"},
    ])
    res = await _run(items=items)
    assert [g.source_ids for g in res.groups] == [(1, 3)]


@pytest.mark.asyncio
async def test_group_shorter_than_longest_absorbed_is_dropped(monkeypatch):
    """Плотная запись короче самой длинной поглощаемой = пересказ с потерей деталей."""
    long_text = "Работает дизайнером удалённо, два постоянных заказчика и " \
                "подработка по выходным; основной доход в семье"
    items = [
        CompactItemView(id=1, kind="fact", text=long_text, source="qa_answer"),
        CompactItemView(id=2, kind="fact", text="Работает на двух работах", source="qa_answer"),
    ]
    _mock_classify(monkeypatch, groups=[
        {"source_ids": [1, 2], "kind": "fact", "text": "Работает на двух работах, основной доход"},
    ])
    res = await _run(items=items)
    assert res.groups == []


@pytest.mark.asyncio
async def test_call_failure_is_noop(monkeypatch):
    """Сбой модели — тихий no-op: память остаётся как была, расход не пишется."""
    monkeypatch.setattr(
        "llm.memory_compact.classify", AsyncMock(side_effect=RuntimeError("boom"))
    )
    res = await _run()
    assert res.groups == []
    assert res.reject_reason == "call_failed"
    assert res.input_tokens == 0


@pytest.mark.asyncio
async def test_empty_input_is_noop(monkeypatch):
    spy = _mock_classify(monkeypatch, groups=[])
    res = await _run(items=[])
    assert res.reject_reason == "empty_input"
    spy.assert_not_awaited()
