"""Уплотнение долговременной памяти — «сухой остаток» (2026-08-16).

Отдельный проход поверх накопленных единиц: разрозненные ячейки одной темы
собираются в одну плотную формулировку (десять фактов про работу → три). Честный
ответ на рост объёма памяти: бюджет блока 1600 символов не резиновый, а у самого
активного прод-юзера уже 48 разрозненных ячеек после 14 сессий — в окно начинает
попадать свежая мелочь, вытесняя важное. Побочно лечит перефразные дубли на
длинной дистанции, там где посимвольный дедуп в экстракторе бессилен.

Что НЕ уплотняется: ядровые факты (kind="core" — там и так одна ячейка на
категорию) и стороны (kind="part" — они матчатся между сессиями по entity_key,
слияние сломало бы узнавание).

Чистая функция без I/O в БД: вызывающий применяет группы через репозиторий
(новая запись + supersede исходных). Безопасный дефолт при любом сбое — no-op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from openai import AsyncOpenAI
from pydantic import BaseModel

from llm.classifier import ClassifierParseError, classify
from llm.memory_extract import _DREAM_MARKER_RE, _depersonalize
from llm.memory_select import (
    MEMORY_BLOCK_BUDGET,
    build_memory_block,
    rendered_item_count,
)

if TYPE_CHECKING:
    from data.models import UserMemoryItem

log = structlog.get_logger(__name__)

MEMORY_COMPACT_VERSION = "memory.compact.v1"

# source уплотнённой записи: и провенанс, и метка «после этой строки память уже
# уплотняли» — по ней считается приток новых ячеек (см. select_for_compaction).
COMPACT_SOURCE = "compact"

# Виды, которые можно сливать. core и part исключены сознательно (см. модульный
# docstring); insight оставлен, хотя экстрактор его больше не выдаёт — в проде
# такие строки есть с v1.
COMPACTABLE_KINDS = frozenset({"fact", "attitude", "episode", "pattern", "insight", "person"})
_ALLOWED_OUTPUT_KINDS = COMPACTABLE_KINDS

# Приток новых ячеек с прошлого уплотнения. Без него проход звался бы на каждой
# паузе у юзера, чья память осталась за лимитом (модель отказалась что-то сливать).
DEFAULT_MIN_NEW_SINCE_LAST = 5
# Рендер «сколько бы заняла ВСЯ память, если её не резать» — бюджет заведомо
# недостижимый, чтобы build_memory_block ничего не отбросил по месту.
_NO_BUDGET = 10**6

_MAX_TEXT = 500  # зеркало колонки user_memory_items.text
# Вход — до нескольких десятков коротких записей, выход — единицы плотных строк.
_COMPACT_MAX_TOKENS = 1500


@dataclass(frozen=True)
class CompactItemView:
    """Плоское представление active-единицы для промпта уплотнения."""

    id: int
    kind: str
    text: str
    source: str


@dataclass(frozen=True)
class CompactGroup:
    """Одна уплотнённая запись: чем заменяем и что именно она поглощает."""

    source_ids: tuple[int, ...]
    kind: str
    text: str


@dataclass(frozen=True)
class MemoryCompactResult:
    groups: list[CompactGroup]
    input_tokens: int
    output_tokens: int
    reject_reason: str | None = None


class _RawGroup(BaseModel):
    source_ids: list[int] = []
    kind: str = "fact"
    text: str = ""


class _Compact(BaseModel):
    groups: list[_RawGroup] = []


def _no_op(reason: str) -> MemoryCompactResult:
    return MemoryCompactResult(groups=[], input_tokens=0, output_tokens=0, reject_reason=reason)


def select_for_compaction(
    items: list[UserMemoryItem],
    *,
    budget_chars: int = MEMORY_BLOCK_BUDGET,
    min_new: int = DEFAULT_MIN_NEW_SINCE_LAST,
) -> list[CompactItemView]:
    """Какие active-единицы отдать на уплотнение; пустой список = ещё не пора.

    Триггер — ПАМЯТЬ ПЕРЕСТАЛА ВЛЕЗАТЬ В БЛОК, а не счётчик ячеек.
    Пока вся память доезжает до промпта, уплотнять нечего:
    юзер ничего не теряет, а вызов модели на каждой паузе стоил бы денег.
    Поводов два, и блок режется ровно по этим двум причинам:
    1) бюджет — память, собранная без ограничения, длиннее `budget_chars`;
    2) квоты по видам — в блок попало меньше строк, чем есть active-единиц
       (десять фактов занимают ~700 символов, до лимита не дотягивают, но четыре
       из них в разбор не попадут никогда: квота вида — шесть строк).

    Второе условие — приток: с прошлого уплотнения должно появиться не меньше
    `min_new` новых ячеек. «Новые» считаются по id (id монотонен), схемы в БД
    для этого не нужно. Сами уплотнённые записи в материал входят — иначе
    память расслоилась бы на «сухие остатки» и новьё, которые не сойдутся.
    """
    active = [i for i in items if getattr(i, "status", "active") in (None, "active")]
    pool = [i for i in active if i.kind in COMPACTABLE_KINDS]
    if len(pool) < 2:
        return []
    full = build_memory_block(active, budget_chars=_NO_BUDGET)
    over_budget = len(full) > budget_chars
    cut_by_quotas = rendered_item_count(full) < len(active)
    if not (over_budget or cut_by_quotas):
        return []
    last_compact_id = max(
        (i.id for i in active if (i.source or "") == COMPACT_SOURCE), default=0
    )
    if last_compact_id:
        fresh = [i for i in pool if i.id > last_compact_id]
        if len(fresh) < min_new:
            return []
    return [
        CompactItemView(id=i.id, kind=i.kind, text=i.text, source=i.source or "")
        for i in sorted(pool, key=lambda i: i.id)
    ]


_SYSTEM = """\
Ты уплотняешь долговременную память бота об одном человеке — делаешь «сухой остаток».
На входе — пронумерованные записи о РЕАЛЬНОЙ жизни этого человека, накопленные за \
несколько разговоров. Они разрозненные и часто говорят об одном и том же разными словами.
Задача: собрать записи ОБ ОДНОЙ И ТОЙ ЖЕ ТЕМЕ в одну плотную запись.

ГЛАВНОЕ ПРАВИЛО — НЕ ПОТЕРЯТЬ СМЫСЛ. Уплотнение это не сокращение и не пересказ: \
конкретика (имена, возраст, профессия, места, сроки, кто кому кем приходится) обязана \
целиком доехать в новую формулировку. Если две записи не укладываются в одну ясную \
фразу без потери деталей — НЕ объединяй их.
НЕ ВЫДУМЫВАЙ. В новой записи может быть только то, что есть во входных: никаких \
выводов, обобщений «в целом», оценок, догадок и интерпретаций.

ЧТО ОБЪЕДИНЯТЬ (в порядке приоритета):
- перефразы одного и того же факта, записанные в разные разы;
- записи об одной теме: работа и деньги; семья, партнёр и дети; здоровье; отношения с \
одним и тем же человеком; повторяющийся способ реагировать; переезд и место жизни.
ЧТО НЕ ТРОГАТЬ: записи, которым не с чем объединяться. Их просто не включай в ответ — \
они останутся в памяти как есть. Ответ без единой группы ({"groups":[]}) — нормальный \
исход, а не ошибка.
РАЗНЫЕ ТЕМЫ НЕ СМЕШИВАЙ: работа и семья, здоровье и деньги, отношения с разными людьми — \
это РАЗНЫЕ группы, даже если каждая мала. Одна группа = одна тема.
УЖЕ УПЛОТНЁННОЕ НЕ ПЕРЕСКАЗЫВАЙ: если запись выглядит как плотная сводка, её можно только \
ДОПОЛНИТЬ новой конкретикой, сохранив всё, что в ней было. Пересказ короче исходного — \
потеря памяти, а не уплотнение.

ФОРМА ГРУППЫ: source_ids — номера входных записей, которые поглощает новая; их должно \
быть НЕ МЕНЬШЕ ДВУХ. Один номер используй не больше чем в одной группе.
kind новой записи — из {fact, attitude, pattern, person, episode}: pattern — \
повторяющийся способ вести себя; person — про значимого человека; attitude — отношение \
к чему-то; episode — событие во времени; fact — устойчивый факт жизни, когда остальное \
не подходит.

ФОРМУЛИРОВКА — как в исходных записях: без слова «юзер», без подлежащего, начиная со \
сказуемого («Работает программистом и ведёт своё дело; доход нестабилен»). Пол человека \
НЕизвестен — род нигде не должен быть выражен: настоящее время, отглагольные \
существительные, краткое именное сказуемое. Никакого первого лица и местоимений \
«он»/«она» про самого человека. 1–3 предложения, до 500 символов.
Содержимого сна в памяти нет и быть не должно — не добавляй его.

Отвечай ТОЛЬКО строгим JSON без markdown:
{"groups":[{"source_ids":[1,2],"kind":"fact","text":"..."}]}\
"""


def _build_user_message(items: list[CompactItemView]) -> str:
    lines = ["ЗАПИСИ ПАМЯТИ (номер | вид | текст):"]
    lines += [f"{i.id} | {i.kind} | {i.text}" for i in items]
    return "\n".join(lines)


async def compact_memory(
    *,
    client: AsyncOpenAI,
    model: str,
    items: list[CompactItemView],
) -> MemoryCompactResult:
    """Собрать разрозненные единицы памяти в плотные группы. Никогда не бросает."""
    if not items:
        return _no_op("empty_input")

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": _build_user_message(items)},
    ]
    try:
        res = await classify(
            client=client,
            model=model,
            messages=messages,
            schema=_Compact,
            max_tokens=_COMPACT_MAX_TOKENS,
        )
    except ClassifierParseError:
        log.warning("memory_compact.parse_failed")
        return _no_op("parse_failed")
    except Exception:  # noqa: BLE001 — сеть/таймаут/прочее: память остаётся как была
        log.warning("memory_compact.call_failed", exc_info=True)
        return _no_op("call_failed")

    by_id = {i.id: i for i in items}
    valid_ids = set(by_id)
    taken: set[int] = set()
    groups: list[CompactGroup] = []
    for raw in res.parsed.groups:
        # Порядок сохраняем, дубли внутри группы схлопываем: одна ячейка не может
        # поглотиться дважды даже сама собой.
        ids = tuple(
            dict.fromkeys(
                i for i in raw.source_ids if i in valid_ids and i not in taken
            )
        )
        if len(ids) < 2:
            continue
        sources = [by_id[i] for i in ids]
        # Гард от ПЕРЕуплотнения (прод-прогон 2026-08-16). У самого активного юзера
        # уплотнённая запись поглощалась снова и снова — шесть проходов подряд, — и
        # каждый пересказ ронял детали, пока из памяти не исчез целый важный сюжет.
        # Заодно склеивались разные темы (работа + семья).
        # Два «сухих остатка» в одной группе — как раз этот случай.
        if sum(1 for s in sources if (s.source or "") == COMPACT_SOURCE) > 1:
            log.info("memory_compact.dropped_recompaction", ids=list(ids))
            continue
        text = _depersonalize((raw.text or "").strip())
        if not text or len(text) > _MAX_TEXT:
            continue
        # Вторая половина того же гарда: плотная запись обязана быть НЕ КОРОЧЕ самой
        # длинной из поглощаемых. Короче — значит не уплотнили, а пересказали, и
        # часть конкретики осталась за бортом.
        if len(text) < max(len(s.text or "") for s in sources):
            log.info("memory_compact.dropped_shorter_than_source", text=text[:80])
            continue
        if _DREAM_MARKER_RE.search(text):
            log.info("memory_compact.dropped_dream_marker", text=text[:80])
            continue
        kind = (raw.kind or "").strip()
        if kind not in _ALLOWED_OUTPUT_KINDS:
            kind = "fact"
        taken.update(ids)
        groups.append(CompactGroup(source_ids=ids, kind=kind, text=text))

    log.info("memory_compact.done", n_groups=len(groups), n_absorbed=len(taken))
    return MemoryCompactResult(
        groups=groups, input_tokens=res.input_tokens, output_tokens=res.output_tokens
    )
