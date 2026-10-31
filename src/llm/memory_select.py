"""Детерминированный отбор единиц памяти в тематический блок для генератора.

Память v2 (спека 2026-08-15, A2): вместо плоского списка с квотами — блок,
сгруппированный по темам с заголовками в фиксированном порядке:
«О вас» → «О семье» → «О работе и деньгах» →
«Знакомые стороны» (part-ы, как раньше) → «Как с вами общаться» (core:tone).

Чистая функция без LLM и без БД. Общий заголовок HEADER сохранён байт-в-байт
с v1 — на него завязаны system-промпты quick и тесты промптов. Бюджет 1600
символов сохранён и режется round-robin (см. _FILL_PRIORITY), чтобы catch-all
«О вас» не вытесняла из блока стороны юзера и tone. Пустая строка = не подставлять.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from data.models import UserMemoryItem

HEADER = "ЧТО УЖЕ ИЗВЕСТНО ИЗ ПРОШЛЫХ РАЗБОРОВ (факты и наблюдения о РЕАЛЬНОЙ жизни):\n"

# Бюджет блока в символах. Вынесен в константу: на него же смотрит уплотнение
# памяти (llm/memory_compact.py) — проход запускается, когда память перестала
# помещаться в этот бюджет. Два числа не должны разъезжаться.
# 1600 → 2500 (2026-08-16). Прод-срез: бюджет резал память у
# 4 юзеров из 31, на 2500 остаётся один (у него полный блок 4259 символов). В
# промпт доезжает на 19 записей больше; доплата на разборы — сотые доли цента за
# вызов генератора ($0.04 на всю трёхмесячную историю прода). Выше 2500 не идти:
# дальше растёт не польза, а стена фактов, в которой разбор тонет.
MEMORY_BLOCK_BUDGET = 2500

# Правило старшинства (смоук 2026-08-15): анкета «о себе» в профиле — застывший
# текст («мне 38»), ячейки памяти обновляются («исполнилось 39») — при расхождении
# генератор должен верить памяти. Живёт ОТДЕЛЬНО от HEADER: заголовок байт-в-байт
# завязан на system-промпты quick, менять его нельзя. В пустой блок не попадает.
FRESHNESS_NOTE = (
    "(это знание свежее анкеты «о себе» из профиля: "
    "если что-то расходится с ней — верь этому блоку)\n"
)

# Заголовок секции сторон (part) — сохранён из v1: генератор должен УЗНАВАТЬ
# знакомые части, а не открывать их заново.
PARTS_HEADER = "ЗНАКОМЫЕ СТОРОНЫ ЭТОГО ЧЕЛОВЕКА:"

# Порядок тем — фиксирован спекой (A2). Секция «О переменах» убрана по итогам
# прод-прогона 2026-08-15 вместе с категорией change (однодневные обстоятельства).
SECTION_ORDER = ("about", "family", "work", "parts", "tone")
SECTION_HEADERS = {
    "about": "О вас:",
    "family": "О семье:",
    "work": "О работе и деньгах:",
    "parts": PARTS_HEADER,
    "tone": "Как с вами общаться:",
}

# Ядровые категории (kind="core") → тема. age/place/health — «О вас».
# Легаси-ключи (name/change) в БД уже есть — они падают в catch-all «О вас».
_CORE_KEY_SECTIONS = {
    "age": "about",
    "place": "about",
    "health": "about",
    "family": "family",
    "work": "work",
    "tone": "tone",
}
# Порядок ядровых категорий внутри секции («возраст» раньше «здоровья»).
_CORE_KEY_RANK = {
    k: i for i, k in enumerate(("age", "place", "health", "family", "work", "tone"))
}

# Маркер строки-пункта. Вынесен в константу: по нему уплотнение считает, сколько
# единиц реально попало в блок (rendered_item_count).
_LINE_PREFIX = "– "

_KIND_LABELS = {
    "part": "сторона",
    "pattern": "паттерн",
    "insight": "инсайт",
    "attitude": "отношение",
    "person": "человек",
    "fact": "факт",
    "episode": "эпизод",
}

# Группы порядка внутри секции: меньше — раньше.
_KIND_GROUPS = {
    "part": 0,
    "pattern": 1,
    "insight": 1,
    "attitude": 2,
    "person": 2,
    "fact": 2,
}
_FALLBACK_GROUP = 3  # episode и любые новые виды — в хвост

# Квоты: сколько юнитов каждого вида максимум попадает в блок.
# core фактически ограничен вытеснением (одна active-ячейка на категорию).
_KIND_QUOTAS = {
    "core": 8,
    "part": 6,
    "pattern": 4,
    "insight": 4,
    "attitude": 4,
    "person": 4,
    "fact": 6,
    "episode": 2,
}
_FALLBACK_QUOTA = 3

# Порядок НАПОЛНЕНИЯ бюджета — НЕ порядок вывода (вывод всегда SECTION_ORDER).
# «О вас» — catch-all: в неё падает всё, что не part и не core, до 24 строк по
# квотам. Когда она наполнялась первой, бюджет 1600 кончался внутри неё, и весь
# хвост секций (в т.ч. ЗНАКОМЫЕ СТОРОНЫ и tone) не рендерился вовсе — регрессия
# против v1, где part-строки попадали в блок всегда. Поэтому дефицитные секции
# обслуживаются раньше, а catch-all добирает остаток последней.
_FILL_PRIORITY = ("parts", "tone", "family", "work", "about")


def _section_of(item: UserMemoryItem) -> str:
    if item.kind == "part":
        return "parts"
    if item.kind == "core":
        return _CORE_KEY_SECTIONS.get(getattr(item, "entity_key", None) or "", "about")
    return "about"


def _line_for(item: UserMemoryItem) -> str:
    # core и part живут под говорящим заголовком секции — ярлык вида избыточен.
    if item.kind in ("core", "part"):
        return f"{_LINE_PREFIX}{item.text}"
    return f"{_LINE_PREFIX}[{_KIND_LABELS.get(item.kind, item.kind)}] {item.text}"


def rendered_item_count(block: str) -> int:
    """Сколько единиц памяти реально попало в блок (строки-пункты, без заголовков).

    Нужен уплотнению: если в блок попало меньше, чем есть active-единиц, часть
    памяти в разбор не доезжает — и не обязательно из-за бюджета, режут ещё и
    квоты по видам (_KIND_QUOTAS). Формат строки знает только этот модуль,
    поэтому и счётчик живёт здесь.
    """
    return sum(1 for line in block.splitlines() if line.startswith(_LINE_PREFIX))


def _sort_key(pair: tuple[int, UserMemoryItem]) -> tuple:
    idx, item = pair
    is_core = item.kind == "core"
    core_rank = (
        _CORE_KEY_RANK.get(getattr(item, "entity_key", None) or "", 99) if is_core else 99
    )
    group = 0 if is_core else 1 + _KIND_GROUPS.get(item.kind, _FALLBACK_GROUP)
    return (group, core_rank, -_timestamp(item), -(item.confidence or 0), idx)


def build_memory_block(
    items: list[UserMemoryItem], *, budget_chars: int = MEMORY_BLOCK_BUDGET
) -> str:
    """Собирает тематический блок памяти из active-юнитов.

    Детерминированно (без случайности): секции ВЫВОДЯТСЯ в порядке SECTION_ORDER;
    внутри секции core-факты первыми (в порядке категорий), дальше по (группа
    вида, свежесть desc, confidence desc, исходный индекс). Заголовок секции
    печатается только вместе с её первой строкой (нет заголовков-сирот).

    Бюджет режется round-robin по _FILL_PRIORITY: в каждом круге каждая секция
    получает не больше одной строки, и дефицитные секции («знакомые стороны»,
    tone) обслуживаются раньше catch-all «О вас». Так первая строка достаётся
    каждой непустой секции — это и есть резерв бюджета под них. Юнит, который
    не влез, закрывает СВОЮ секцию, а не всю сборку (в v2.0 здесь был глобальный
    break, и хвост секций пропадал целиком).
    Пустой вход или бюджет, в который ничего не влезает, → "".
    """
    active = [it for it in items if getattr(it, "status", "active") in (None, "active")]
    if not active:
        return ""

    by_section: dict[str, list[tuple[int, UserMemoryItem]]] = {}
    for idx, item in enumerate(active):
        by_section.setdefault(_section_of(item), []).append((idx, item))
    queues = {
        section: sorted(entries, key=_sort_key) for section, entries in by_section.items()
    }

    picked: dict[str, list[str]] = {section: [] for section in SECTION_ORDER}
    cursors: dict[str, int] = {section: 0 for section in SECTION_ORDER}
    closed: dict[str, bool] = {section: section not in queues for section in SECTION_ORDER}
    taken_per_kind: dict[str, int] = {}
    # Примечание о старшинстве сидит в бюджете так же, как заголовок.
    used = len(HEADER) + len(FRESHNESS_NOTE)

    while not all(closed.values()):
        placed_any = False
        for section in _FILL_PRIORITY:
            if closed[section]:
                continue
            entries = queues[section]
            while cursors[section] < len(entries):
                item = entries[cursors[section]][1]
                kind = item.kind
                if taken_per_kind.get(kind, 0) >= _KIND_QUOTAS.get(kind, _FALLBACK_QUOTA):
                    cursors[section] += 1  # квота вида выбрана — юнит пропускаем
                    continue
                line = _line_for(item)
                cost = len(line) + 1  # + перевод строки
                if not picked[section]:
                    cost += len(SECTION_HEADERS[section]) + 1
                if used + cost > budget_chars:
                    # Секция закрыта, остальные продолжают наполняться.
                    closed[section] = True
                    break
                picked[section].append(line)
                taken_per_kind[kind] = taken_per_kind.get(kind, 0) + 1
                used += cost
                cursors[section] += 1
                placed_any = True
                break
            else:
                closed[section] = True  # юниты кончились
        if not placed_any:
            break

    lines: list[str] = []
    for section in SECTION_ORDER:
        if picked[section]:
            lines.append(SECTION_HEADERS[section])
            lines.extend(picked[section])

    if not lines:
        return ""
    return HEADER + FRESHNESS_NOTE + "\n".join(lines) + "\n"


def _timestamp(item: UserMemoryItem) -> float:
    """Свежесть юнита как число; None (несфлашенный default) — в хвост группы."""
    ts = item.last_confirmed_at
    if ts is None:
        return float("-inf")
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    return ts.timestamp()
