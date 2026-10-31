"""Тесты тематического отбора памяти в memory_block (Память v2, спека A2).

build_memory_block — чистая функция без LLM/БД: юниты создаём лёгкой фабрикой
на SimpleNamespace с теми же атрибутами, что у UserMemoryItem (kind, text,
status, confidence, last_confirmed_at, entity_key) — БД не трогаем.
"""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

from llm.memory_select import (
    HEADER,
    MEMORY_BLOCK_BUDGET,
    PARTS_HEADER,
    SECTION_HEADERS,
    build_memory_block,
)

_BASE_TS = datetime(2026, 7, 1, 12, 0, tzinfo=UTC)


def _item(
    kind: str = "fact",
    text: str = "работает на двух работах",
    *,
    status: str = "active",
    confidence: int = 1,
    last_confirmed_at: datetime | None = _BASE_TS,
    entity_key: str | None = None,
) -> SimpleNamespace:
    """Стенд-ин UserMemoryItem: только атрибуты, которые читает селектор."""
    return SimpleNamespace(
        kind=kind, text=text, status=status, confidence=confidence,
        last_confirmed_at=last_confirmed_at, entity_key=entity_key,
    )


def _core(key: str, text: str, **kw) -> SimpleNamespace:
    return _item(kind="core", text=text, entity_key=key, **kw)


def test_empty_returns_blank():
    assert build_memory_block([]) == ""


def test_rendered_item_count_ignores_headers():
    """Счётчик попавших в блок единиц: по нему уплотнение понимает, что часть
    памяти в промпт не доехала (квоты по видам режут её раньше бюджета)."""
    from llm.memory_select import rendered_item_count

    assert rendered_item_count("") == 0
    block = build_memory_block([_item(), _core("work", "работает врачом")])
    assert rendered_item_count(block) == 2
    # Десять фактов не поместятся: квота вида fact — шесть строк.
    many = build_memory_block([_item(text=f"факт {i}") for i in range(10)])
    assert rendered_item_count(many) == 6


def test_header_byte_identical_to_v1():
    # На HEADER завязаны system-промпты quick («ЧТО УЖЕ ИЗВЕСТНО») и тесты
    # test_quick_prompt/test_quick_handlers — заголовок менять нельзя.
    assert HEADER == (
        "ЧТО УЖЕ ИЗВЕСТНО ИЗ ПРОШЛЫХ РАЗБОРОВ (факты и наблюдения о РЕАЛЬНОЙ жизни):\n"
    )


def test_block_starts_with_header():
    assert build_memory_block([_item()]).startswith(HEADER)


def test_core_items_land_in_their_sections():
    items = [
        _core("work", "Работает продакт-менеджером"),
        _core("family", "Женат, двое детей"),
        _core("tone", "Просит без прямых советов"),
    ]
    block = build_memory_block(items)
    assert SECTION_HEADERS["family"] in block
    assert SECTION_HEADERS["work"] in block
    assert SECTION_HEADERS["tone"] in block
    assert block.index("О семье:") < block.index("О работе и деньгах:")
    assert block.index("О работе и деньгах:") < block.index("Как с вами общаться:")


def test_full_section_order():
    items = [
        _core("tone", "Просит без советов"),
        _item(kind="part", text="Внутренний критик"),
        _core("work", "Работает в найме"),
        _core("family", "Двое детей"),
        _core("age", "42 года"),
    ]
    block = build_memory_block(items)
    positions = [
        block.index("О вас:"),
        block.index("О семье:"),
        block.index("О работе и деньгах:"),
        block.index(PARTS_HEADER),
        block.index("Как с вами общаться:"),
    ]
    assert positions == sorted(positions)


def test_change_section_removed():
    """Прод-прогон 2026-08-15: категория change выпилена (однодневные обстоятельства),
    секции «О переменах» больше нет ни в порядке, ни в заголовках."""
    from llm.memory_select import SECTION_ORDER

    assert "change" not in SECTION_ORDER
    assert "change" not in SECTION_HEADERS
    assert all("О переменах" not in h for h in SECTION_HEADERS.values())


def test_retired_core_keys_fall_back_to_about():
    """Легаси-строки с entity_key=change/name (уже в проде) не теряются: идут в «О вас»."""
    block = build_memory_block([
        _core("change", "Ожидает интервью по гражданству"),
        _core("name", "Зовут Александр"),
    ])
    assert "О вас:" in block
    assert "Ожидает интервью по гражданству" in block
    assert "Зовут Александр" in block
    assert "О переменах:" not in block


def test_parts_render_under_parts_header():
    block = build_memory_block([
        _item(kind="part", text="Обесценивающий критик"),
        _item(kind="fact", text="живёт в Белграде"),
    ])
    assert PARTS_HEADER in block
    # Факт — в «О вас» (раньше), сторона — под своим заголовком (позже).
    assert block.index("живёт в Белграде") < block.index(PARTS_HEADER)
    assert block.index(PARTS_HEADER) < block.index("Обесценивающий критик")


def test_core_and_part_lines_have_no_kind_labels():
    block = build_memory_block([
        _core("work", "Работает продакт-менеджером"),
        _item(kind="part", text="Внутренний критик"),
    ])
    assert "– Работает продакт-менеджером" in block
    assert "– Внутренний критик" in block
    assert "[" not in block.replace(HEADER, "")


def test_regular_items_keep_kind_labels_in_about():
    block = build_memory_block([
        _item(kind="fact", text="работает на двух работах"),
        _item(kind="attitude", text="к подросткам держится подальше"),
    ])
    assert "О вас:" in block
    assert "– [факт] работает на двух работах" in block
    assert "– [отношение] к подросткам держится подальше" in block


def test_unknown_kind_goes_to_about_with_own_label():
    block = build_memory_block([_item(kind="habit", text="пьёт кофе по утрам")])
    assert "О вас:" in block
    assert "– [habit] пьёт кофе по утрам" in block


def test_empty_sections_omitted():
    block = build_memory_block([_item(kind="fact", text="один факт")])
    assert "О вас:" in block
    for name in ("О семье:", "О работе и деньгах:", "Как с вами общаться:"):
        assert name not in block
    assert PARTS_HEADER not in block


def test_core_age_ranks_before_health_inside_about():
    block = build_memory_block([
        _core("health", "Восстанавливается после операции"),
        _core("age", "42 года"),
    ])
    assert block.index("42 года") < block.index("Восстанавливается")


def test_core_before_regular_items_inside_about():
    block = build_memory_block([
        _item(kind="fact", text="работает на двух работах"),
        _core("age", "42 года"),
    ])
    assert block.index("42 года") < block.index("работает на двух работах")


def test_superseded_and_archived_excluded():
    block = build_memory_block([
        _item(text="актуальный факт"),
        _item(text="вытесненный факт", status="superseded"),
        _item(text="архивный факт", status="archived"),
    ])
    assert "актуальный факт" in block
    assert "вытесненный факт" not in block
    assert "архивный факт" not in block


def test_kind_quota_applies():
    items = [_item(kind="fact", text=f"факт номер {i}") for i in range(8)]
    block = build_memory_block(items)
    assert block.count("– [факт]") == 6  # квота fact = 6


def test_budget_respected():
    items = [_item(kind="fact", text="ф" * 200) for _ in range(10)]
    assert len(build_memory_block(items, budget_chars=700)) <= 700


def test_no_orphan_section_header_on_tight_budget():
    # HEADER + «О вас:» без единой строки не влезают → блок пуст (нет сироты-заголовка).
    tight = len(HEADER) + len("О вас:")
    assert build_memory_block([_item()], budget_chars=tight) == ""


def _overflowing_about_items() -> list[SimpleNamespace]:
    """18 обычных единиц (влезают по квотам видов), одна сторона и два ядровых.

    Обычные единицы все падают в catch-all «О вас» (kind не part и не core) и при
    длинах 70/85/100/120 символов переполняют бюджет блока внутри этой секции —
    ровно сценарий активного юзера после 3–4 сессий.
    """
    kinds = ["fact"] * 6 + ["attitude"] * 4 + ["person"] * 4 + ["pattern"] * 4
    lengths = (70, 85, 100, 120)
    items = [
        _item(kind=kind, text=f"{i:02d} " + "я" * lengths[i % 4])
        for i, kind in enumerate(kinds)
    ]
    items.append(_item(kind="part", text="Внутренний критик, обесценивающий результаты"))
    items.append(_core("tone", "Просит без прямых советов и без сюсюканья"))
    items.append(_core("family", "Женат, двое детей"))
    return items


def test_parts_and_tone_survive_when_about_overflows_budget():
    """Регрессия: «О вас» съедала весь бюджет, и хвост секций не рендерился вовсе.

    Знакомые стороны и tone — то, ради чего блок и делался: генератор обязан
    УЗНАВАТЬ части человека, а не открывать их заново.
    """
    block = build_memory_block(_overflowing_about_items())
    assert len(block) <= MEMORY_BLOCK_BUDGET
    assert PARTS_HEADER in block
    assert "Внутренний критик" in block
    assert SECTION_HEADERS["tone"] in block
    assert "без прямых советов" in block
    assert SECTION_HEADERS["family"] in block
    assert "Женат, двое детей" in block
    # «О вас» не должна исчезнуть в обмен — она просто добирает остаток.
    assert SECTION_HEADERS["about"] in block


def test_section_order_preserved_under_budget_pressure():
    """Порядок секций в готовом блоке — прежний, даже когда бюджет режет содержимое."""
    block = build_memory_block(_overflowing_about_items())
    positions = [
        block.index(SECTION_HEADERS["about"]),
        block.index(SECTION_HEADERS["family"]),
        block.index(PARTS_HEADER),
        block.index(SECTION_HEADERS["tone"]),
    ]
    assert positions == sorted(positions)


def test_budget_exhaustion_in_one_section_does_not_kill_the_rest():
    """Один длинный юнит в «О вас» больше не обрывает сборку глобально."""
    items = [
        _item(kind="fact", text="ф" * 400),
        _item(kind="fact", text="ф" * 400),
        _item(kind="fact", text="ф" * 400),
        _item(kind="part", text="Критик"),
        _core("tone", "Без советов"),
    ]
    block = build_memory_block(items, budget_chars=900)
    assert len(block) <= 900
    assert PARTS_HEADER in block
    assert SECTION_HEADERS["tone"] in block


def test_freshness_orders_within_section():
    old = _item(
        kind="fact", text="старый факт", last_confirmed_at=_BASE_TS - timedelta(days=30)
    )
    new = _item(kind="fact", text="свежий факт", last_confirmed_at=_BASE_TS)
    block = build_memory_block([old, new])
    assert block.index("свежий факт") < block.index("старый факт")


def test_deterministic():
    items = [
        _core("work", "Работает в найме"),
        _item(kind="part", text="Критик"),
        _item(kind="fact", text="двое детей и собака"),
    ]
    assert build_memory_block(items) == build_memory_block(items)


def test_freshness_note_follows_header_in_nonempty_block():
    """Фикс живого смоука 2026-08-15: профиль «мне 38» против ядрового «39» —
    генератору нужно правило старшинства. Примечание идёт сразу ПОСЛЕ заголовка."""
    from llm.memory_select import FRESHNESS_NOTE

    block = build_memory_block([_item()])
    assert block.startswith(HEADER + FRESHNESS_NOTE)
    assert FRESHNESS_NOTE.endswith("\n")


def test_freshness_note_absent_in_empty_block():
    from llm.memory_select import FRESHNESS_NOTE

    assert build_memory_block([]) == ""
    # И при бюджете, куда не влезает ни одна строка, примечание тоже не сиротеет.
    tight = len(HEADER) + len(FRESHNESS_NOTE) + len("О вас:")
    assert build_memory_block([_item()], budget_chars=tight) == ""


def test_freshness_note_counted_in_budget_and_header_untouched():
    from llm.memory_select import FRESHNESS_NOTE

    # Примечание живёт ОТДЕЛЬНО от константы заголовка (на HEADER завязан quick).
    assert HEADER == (
        "ЧТО УЖЕ ИЗВЕСТНО ИЗ ПРОШЛЫХ РАЗБОРОВ (факты и наблюдения о РЕАЛЬНОЙ жизни):\n"
    )
    assert FRESHNESS_NOTE not in HEADER
    # Длина примечания сидит в бюджете так же, как заголовок.
    items = [_item(kind="fact", text="ф" * 200) for _ in range(10)]
    assert len(build_memory_block(items, budget_chars=700)) <= 700
