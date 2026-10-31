"""Хелперы, общие для нескольких handlers, и оторванные от UI-конкретики.

Здесь живёт логика, которую невозможно положить в конкретный handler без
циклических импортов (например, `auto_pause_active_quick` зовут и из
dreams.py, и из profile.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog
from aiogram.fsm.context import FSMContext
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers._memory import run_memory_extraction_on_pause
from app.config import Settings
from data.repositories import (
    DreamRepository,
    SessionRepository,
    UsageEventRepository,
    UserRepository,
)
from domain.entry_types import KIND_DREAM
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)

# Дублируем ключ из newdream.py намеренно — чтобы не тянуть circular-import
# (newdream.py импортирует из _common). Должен совпадать с _FSM_KEY_NEWDREAM_DREAM_ID.
_NEWDREAM_DREAM_ID_KEY = "newdream_dream_id"

# FSM-ключ типа обращения (EntryStates + знакомство). Живёт здесь, а не в
# entry_flow.py: onboarding.py тоже читает его, а entry_flow импортирует
# onboarding — константа в _common разрывает цикл.
FSM_KEY_ENTRY_KIND = "entry_kind"

# Ключ dream_id рассказа (тот же, что в onboarding.py/_FSM_KEY_ONBOARDING_DREAM_ID).
_ENTRY_DREAM_ID_KEY = "onboarding_dream_id"


async def discard_pending_newdream(
    *,
    state: FSMContext,
    session: AsyncSession,
    telegram_user_id: int,
) -> bool:
    """Если юзер был в процессе рассказа нового сна (NewDreamStates) и переключился
    командой (/dreams, /profile, /newdream, /start) — выкидываем недосказанный сон
    («как будто его и не было») и чистим FSM, чтобы команда отработала с чистого листа.

    Returns: True если что-то сбросили (юзер был в NewDreamStates), иначе False.
    No-op вне NewDreamStates.
    """
    current = await state.get_state()
    if not (current and current.startswith("NewDreamStates")):
        return False
    data = await state.get_data()
    dream_id = data.get(_NEWDREAM_DREAM_ID_KEY)
    await state.clear()
    if dream_id is not None:
        user = await UserRepository(session).get_by_tg_id(telegram_user_id)
        if user is not None:
            await DreamRepository(session).hard_delete_with_cascade(
                dream_id=dream_id, user_id=user.id
            )
            log.info(
                "newdream.discarded_on_command_switch", dream_id=dream_id, user_id=user.id
            )
    return True


async def record_intent_usage(
    *,
    session: AsyncSession,
    telegram_user_id: int,
    session_id: int | None,
    model: str,
    input_tokens: int,
    output_tokens: int,
    prompt_version: str,
) -> None:
    """Записать расход на intent-вызов (gpt-4o-mini) в llm_usage_events.

    No-op, если токенов 0 (вызова к модели не было / он упал на fallback) или
    юзер не найден. Так дашборд расходов учитывает и дешёвые intent-классификаторы.
    """
    if not input_tokens and not output_tokens:
        return
    user = await UserRepository(session).get_by_tg_id(telegram_user_id)
    if user is None:
        return
    await UsageRecorder(UsageEventRepository(session)).record_openrouter_intent(
        user_id=user.id,
        session_id=session_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_version=prompt_version,
    )


async def record_context_usage(
    *,
    session: AsyncSession,
    telegram_user_id: int,
    session_id: int | None,
    model: str,
    input_tokens: int,
    output_tokens: int,
    prompt_version: str,
) -> None:
    """Записать расход на context-вызов («живая память»/компоновка) в llm_usage_events.

    No-op, если токенов 0 (вызова не было / он упал на safe-default) или юзер не найден.
    """
    if not input_tokens and not output_tokens:
        return
    user = await UserRepository(session).get_by_tg_id(telegram_user_id)
    if user is None:
        return
    await UsageRecorder(UsageEventRepository(session)).record_openrouter_context(
        user_id=user.id,
        session_id=session_id,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_version=prompt_version,
    )


@dataclass(frozen=True)
class PausedEntry:
    """Что именно запаузили: id сессии и тип обращения.

    Тип нужен caller'у для текста уведомления: «вернётесь через /dreams» уводит
    в список, где ситуации и конфликта нет (списки фильтруются по kind).
    """

    session_id: int
    kind: str


async def entry_kind_for_dream(
    session: AsyncSession, *, dream_id: int, user_id: int
) -> str:
    """kind обращения по dream_id; запись не найдена → dream (fail-safe)."""
    dream = await DreamRepository(session).get_by_id(dream_id, user_id=user_id)
    return dream.kind if dream is not None else KIND_DREAM


async def auto_pause_active_quick(
    *,
    session: AsyncSession,
    state: FSMContext,
    user_id: int,
    excluded_dream_id: int | None = None,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> PausedEntry | None:
    """Если у юзера есть IN_PROGRESS Quick-сессия (и не на excluded_dream_id),
    ставит её на pause + clear FSM-state. Возвращает PausedEntry или None.

    Caller сам решает, шлёт ли он уведомление — для этого проверяет, что
    возвращённое значение не None, и берёт текст из entry_config(paused.kind).

    Args:
        excluded_dream_id: если активная сессия принадлежит этому сну — не
            паузим. Используется в `/dream_<same_id>`, чтобы юзер мог открыть
            карточку сна, по которому он уже в Q&A, без auto-pause.
        settings + openrouter_classifier_client: для экстрактора памяти на паузе
            (shadow-режим, за флагом memory_write_enabled). None = skip экстракции.
    """
    sess_repo = SessionRepository(session)
    active = await sess_repo.latest_in_progress_for_user(user_id=user_id)
    if active is None:
        return None
    if excluded_dream_id is not None and active.dream_id == excluded_dream_id:
        return None

    await sess_repo.pause(session_id=active.id)
    if (
        openrouter_classifier_client is not None
        and settings is not None
        and settings.memory_write_enabled
    ):
        await run_memory_extraction_on_pause(
            session=session,
            user_id=user_id,
            session_id=active.id,
            dream_id=active.dream_id,
            client=openrouter_classifier_client,
            model=settings.openrouter_context_model,
            compact_enabled=settings.memory_compact_enabled,
            compact_budget_chars=settings.memory_compact_budget_chars,
        )
    await state.clear()
    kind = await entry_kind_for_dream(session, dream_id=active.dream_id, user_id=user_id)
    log.info(
        "auto_pause.quick",
        paused_session_id=active.id,
        paused_dream_id=active.dream_id,
        paused_kind=kind,
        user_id=user_id,
    )
    return PausedEntry(session_id=active.id, kind=kind)


async def note_content_filter(
    *,
    session: AsyncSession,
    session_id: int,
    where: str,
    user_id: int | None = None,
    dream_id: int | None = None,
    pause: bool,
) -> None:
    """Зафиксировать в логе, что юзеру показали сообщение о блокировке контента.

    `where` — точка потока: "analysis" | "onboarding" | "qa". По этому событию
    считается частота блокировок в journald, без подъёма базы.

    `pause=True` — сессию на паузу (обычный разбор и онбординг: юзер выходит из
    режима разбора). Именно pause, а не complete: COMPLETED в этом проекте
    означает «упёрся в лимит трат» и используется в метриках пейвола.
    Экстрактор памяти на этой паузе НЕ запускается — в сессии нет ни одного
    круга Q&A, извлекать нечего, а вызов платный.

    `pause=False` — круг Q&A: разбор уже выдан, юзер остаётся в IN_QA и может
    переформулировать ответ.
    """
    if pause:
        await SessionRepository(session).pause(session_id=session_id)
    log.info(
        "content_filter.user_notified",
        where=where,
        session_id=session_id,
        dream_id=dream_id,
        user_id=user_id,
        paused=pause,
    )


async def discard_pending_entry(
    *,
    state: FSMContext,
    session: AsyncSession,
    telegram_user_id: int,
) -> bool:
    """Аналог discard_pending_newdream для EntryStates: недосказанная ситуация/
    конфликт при переключении командой выкидывается, FSM чистится.

    Returns: True если что-то сбросили. No-op вне EntryStates.
    """
    current = await state.get_state()
    if not (current and current.startswith("EntryStates")):
        return False
    data = await state.get_data()
    dream_id = data.get(_ENTRY_DREAM_ID_KEY)
    await state.clear()
    if dream_id is not None:
        user = await UserRepository(session).get_by_tg_id(telegram_user_id)
        if user is not None:
            await DreamRepository(session).hard_delete_with_cascade(
                dream_id=dream_id, user_id=user.id
            )
            log.info(
                "entry.discarded_on_command_switch",
                dream_id=dream_id, user_id=user.id,
            )
    return True
