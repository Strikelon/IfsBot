"""Оркестрация извлечения памяти на паузе сессии (fire-and-forget, shadow-режим).

Вызывается ПОСЛЕ `SessionRepository.pause(...)`: читает реплики юзера в Q&A
(+ per_session_context + уже известные единицы + заземлённые образы из qa_map),
зовёт чистый экстрактор `llm.memory_extract` (gpt-4o-mini) и применяет кандидатов
через `UserMemoryItemRepository` (add / confirm / supersede).

Shadow-режим: строки памяти пишутся, но в промпты НЕ подставляются — инъекция
включается отдельным флагом `memory_inject_enabled` (Task 9).

Никогда не бросает: сбой экстракции не должен ломать паузу/UX.
"""

from __future__ import annotations

import re

import structlog
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import Dream
from data.repositories import (
    SessionMessageRepository,
    SessionRepository,
    SessionStateRepository,
    UsageEventRepository,
    UserMemoryItemRepository,
)
from llm.memory_compact import (
    COMPACT_SOURCE,
    MEMORY_COMPACT_VERSION,
    compact_memory,
    select_for_compaction,
)
from llm.memory_extract import (
    MEMORY_EXTRACT_VERSION,
    ExistingItemView,
    extract_memory_items,
)
from llm.memory_select import MEMORY_BLOCK_BUDGET
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)

# Ключ карты беседы в session_state — дублирует QA_MAP_KEY из quick.py намеренно,
# чтобы не тянуть circular-import (quick.py импортирует этот модуль).
_QA_MAP_KEY = "qa_map"

# Ключ прогресса фичи image questions — дублирует IMGQ_STATE_KEY намеренно
# (image_questions.py импортирует quick.py, quick.py импортирует этот модуль).
_IMGQ_STATE_KEY = "image_questions"


def _norm_text(text: str) -> str:
    """Нормализация для сравнения «тот же ли это факт» (регистр/пробелы)."""
    return re.sub(r"\s+", " ", (text or "").strip().lower())


async def run_memory_extraction_on_pause(
    *,
    session: AsyncSession,
    user_id: int,
    session_id: int,
    dream_id: int | None,
    client: AsyncOpenAI,
    model: str,
    compact_enabled: bool = True,
    compact_budget_chars: int = MEMORY_BLOCK_BUDGET,
) -> None:
    """Извлечь единицы памяти из реплик юзера сессии и применить. Никогда не бросает.

    После записи новых единиц — проход уплотнения («сухой остаток»): разрозненные
    ячейки одной темы собираются в одну плотную запись. Запускается НЕ на каждой
    паузе, а только когда память перестала помещаться в бюджет блока промпта.
    Отдельный вызов модели, отдельный SAVEPOINT, отдельная строка расхода.
    """
    try:
        mem_repo = UserMemoryItemRepository(session)
        existing = [
            ExistingItemView(i.id, i.kind, i.text, i.entity_key)
            for i in await mem_repo.list_active(user_id=user_id)
        ]
        replies = await SessionMessageRepository(session).user_replies_with_prompts(
            session_id=session_id
        )
        state_data = await SessionStateRepository(session).get_data(session_id=session_id) or {}
        qa_map = state_data.get(_QA_MAP_KEY)
        qa_map = qa_map if isinstance(qa_map, dict) else {}
        grounded = [str(g) for g in (qa_map.get("grounded") or []) if g]
        imgq = state_data.get(_IMGQ_STATE_KEY)
        imgq = imgq if isinstance(imgq, dict) else {}
        answered_names = [
            str(i.get("name")) for i in (imgq.get("items") or [])
            if isinstance(i, dict) and i.get("answer") and i.get("name")
        ]
        # union с сохранением порядка (qa_map.grounded первыми)
        grounded = list(dict.fromkeys([*grounded, *answered_names]))
        # per_session_context — из РЕАЛЬНОЙ сессии (то, что юзер рассказал перед разбором).
        sess = await SessionRepository(session).get_by_id(session_id, user_id=user_id)
        per_session = (sess.per_session_context or "") if sess is not None else ""

        # Тип обращения: для ситуации/конфликта рассказ — это реальность (A2/B3),
        # подаём его архивариусу как entry_text. Для снов entry_text пуст —
        # вход экстрактора байт-в-байт прежний.
        entry_kind, entry_text = "dream", ""
        if dream_id is not None:
            dream_row = await session.get(Dream, dream_id)
            if dream_row is not None and dream_row.kind and dream_row.kind != "dream":
                entry_kind = dream_row.kind
                entry_text = dream_row.transcription or ""

        res = await extract_memory_items(
            client=client,
            model=model,
            existing_items=existing,
            user_replies=replies,
            grounded_labels=grounded,
            per_session_context=per_session,
            entry_kind=entry_kind,
            entry_text=entry_text,
        )
        # SAVEPOINT: сбой записи единиц (например StringDataRightTruncation на
        # Postgres) откатывает только вложенный блок и НЕ травит общую транзакцию
        # запроса (finish/pause, воронка, commit) — паттерн emit_funnel_step /
        # emit_daily_active (фикс I1, f3f6ab9).
        try:
            async with session.begin_nested():
                for c in res.candidates:
                    if c.action == "confirm" and c.target_id is not None:
                        await mem_repo.confirm(c.target_id)
                    elif c.action == "supersede" and c.target_id is not None:
                        new = await mem_repo.add(
                            user_id=user_id, kind=c.kind, text=c.text, realm=c.realm,
                            source=c.source, entity_key=c.entity_key,
                            source_session_id=session_id, dream_id=dream_id,
                        )
                        await mem_repo.supersede(c.target_id, new.id)
                    else:
                        # Ядровые факты: та же категория (entity_key) уже есть →
                        # дословный повтор подтверждает (confirm), явное обновление
                        # устаревшего факта вытесняет (supersede), а просто «ещё
                        # что-то из жизни» ячейку НЕ трогает. Прод-прогон
                        # 2026-08-15: вытеснение по любому новому тексту стирало
                        # устойчивые факты сиюминутными («работает в саду» поверх
                        # «соло-человек, учится опираться на людей»).
                        if c.kind == "core" and c.entity_key:
                            existing_core = await mem_repo.find_by_entity(
                                user_id, c.entity_key, kind="core"
                            )
                            if existing_core is not None:
                                if _norm_text(existing_core.text) == _norm_text(c.text):
                                    await mem_repo.confirm(existing_core.id)
                                elif c.replaces_outdated:
                                    new = await mem_repo.add(
                                        user_id=user_id, kind=c.kind, text=c.text,
                                        realm=c.realm, source=c.source,
                                        entity_key=c.entity_key,
                                        source_session_id=session_id,
                                        dream_id=dream_id,
                                    )
                                    await mem_repo.supersede(existing_core.id, new.id)
                                else:
                                    # Ядро не разрушаем, информацию не теряем:
                                    # обычная ячейка без категории.
                                    await mem_repo.add(
                                        user_id=user_id, kind="fact", text=c.text,
                                        realm=c.realm, source=c.source,
                                        entity_key=None,
                                        source_session_id=session_id,
                                        dream_id=dream_id,
                                    )
                                continue
                        # Части матчим по entity_key через сессии: та же роль во
                        # втором сне («критик-обесцениватель» узнался) → confirm
                        # существующей, не дубль.
                        if c.kind == "part" and c.entity_key:
                            existing_part = await mem_repo.find_by_entity(
                                user_id, c.entity_key, kind="part"
                            )
                            if existing_part is not None:
                                await mem_repo.confirm(existing_part.id)
                                continue
                        await mem_repo.add(
                            user_id=user_id, kind=c.kind, text=c.text, realm=c.realm,
                            source=c.source, entity_key=c.entity_key,
                            source_session_id=session_id, dream_id=dream_id,
                        )
        except Exception:  # noqa: BLE001 — единицы откатились к SAVEPOINT, usage ниже цел
            log.warning("memory.extraction.apply_failed", session_id=session_id, exc_info=True)
        # Zero-guard: usage пишем только при реальном вызове модели.
        if res.input_tokens or res.output_tokens:
            await UsageRecorder(UsageEventRepository(session)).record_openrouter_context(
                user_id=user_id,
                session_id=session_id,
                model=model,
                input_tokens=res.input_tokens,
                output_tokens=res.output_tokens,
                prompt_version=MEMORY_EXTRACT_VERSION,
            )
        log.info("memory.extraction.applied", session_id=session_id, n=len(res.candidates))
    except Exception:  # noqa: BLE001 — память не критична, пауза уже состоялась
        log.warning("memory.extraction.failed", session_id=session_id, exc_info=True)

    if compact_enabled:
        await _run_memory_compaction(
            session=session,
            user_id=user_id,
            session_id=session_id,
            dream_id=dream_id,
            client=client,
            model=model,
            budget_chars=compact_budget_chars,
        )


async def _run_memory_compaction(
    *,
    session: AsyncSession,
    user_id: int,
    session_id: int,
    dream_id: int | None,
    client: AsyncOpenAI,
    model: str,
    budget_chars: int,
) -> None:
    """Уплотнить накопленные единицы памяти в «сухой остаток». Никогда не бросает.

    Момент решает `select_for_compaction`: пока память влезает в бюджет блока
    (или с прошлого уплотнения не набежало новых ячеек) — модель не зовётся
    вообще, лишних трат нет. Поглощённые единицы гасятся supersede-ссылкой на
    плотную запись, поэтому история не теряется и решение разбирается постфактум.
    """
    try:
        mem_repo = UserMemoryItemRepository(session)
        active = await mem_repo.list_active(user_id=user_id)
        by_id = {i.id: i for i in active}
        selected = select_for_compaction(active, budget_chars=budget_chars)
        if not selected:
            return

        res = await compact_memory(client=client, model=model, items=selected)
        try:
            async with session.begin_nested():
                for group in res.groups:
                    sources = [by_id[i] for i in group.source_ids if i in by_id]
                    if len(sources) < 2:
                        continue
                    # Ассоциации не переезжают в факты реальной жизни: realm
                    # повышаем до real_life, только если он там был у источников.
                    realm = (
                        "association"
                        if all(s.realm == "association" for s in sources)
                        else "real_life"
                    )
                    new = await mem_repo.add(
                        user_id=user_id, kind=group.kind, text=group.text, realm=realm,
                        source=COMPACT_SOURCE, entity_key=None,
                        source_session_id=session_id, dream_id=dream_id,
                    )
                    for old in sources:
                        await mem_repo.supersede(old.id, new.id)
        except Exception:  # noqa: BLE001 — группы откатились к SAVEPOINT, usage ниже цел
            log.warning("memory.compaction.apply_failed", session_id=session_id, exc_info=True)
        if res.input_tokens or res.output_tokens:
            await UsageRecorder(UsageEventRepository(session)).record_openrouter_context(
                user_id=user_id,
                session_id=session_id,
                model=model,
                input_tokens=res.input_tokens,
                output_tokens=res.output_tokens,
                prompt_version=MEMORY_COMPACT_VERSION,
            )
        log.info(
            "memory.compaction.applied",
            session_id=session_id,
            n_in=len(selected),
            n_groups=len(res.groups),
        )
    except Exception:  # noqa: BLE001 — уплотнение не критично, пауза уже состоялась
        log.warning("memory.compaction.failed", session_id=session_id, exc_info=True)
