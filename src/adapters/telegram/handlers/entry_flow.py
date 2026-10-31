"""Рассказ и loop для типов «волнующая ситуация» / «внутренний конфликт».

Спека B3, шаги 1–2. Одна группа EntryStates обслуживает и онбординг, и повторные
обращения: развилка в done-ветке loop'а — не-onboarded уходит в знакомство
(OnboardingStates.ASKING_LIFE_CONTEXT), onboarded сразу получает per-session
вопрос (QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT, дальше общий конвейер
quick.py/image_questions.py, kind резолвится по dreams.kind).

Команды /new_external_situation, /new_internal_conflict — Task 16 (этот же модуль).
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as dt

import structlog
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    FSM_KEY_ENTRY_KIND,
)
from adapters.telegram.handlers._loop_prompt import (
    extinguish_loop_button,
    send_loop_prompt,
)
from adapters.telegram.handlers._new_entry import start_new_entry
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.handlers.onboarding import (
    _FSM_KEY_ONBOARDING_DREAM_ID,
    _get_text_from_message,
    _resolve_loop_reply,
)
from adapters.telegram.keyboards import LOOP_DONE_CALLBACK, remove_keyboard
from adapters.telegram.states import EntryStates, OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import (
    ASK_LIFE_CONTEXT_ONB,
    LOOP_ADD_BTN_DONE,
    NOT_ONBOARDED_PROMPT,
    THINKING_GUARD_NOTICE,
)
from app.config import Settings
from data.analytics_events import AnalyticsEventType, emit_funnel_step
from data.models import DreamStatus, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import KIND_CONFLICT, KIND_DREAM, KIND_SITUATION, entry_config

log = structlog.get_logger(__name__)


def _fsm_kind(data: dict) -> str:
    """kind из FSM-data; отсутствие ключа → dream (fail-safe, легаси не ломаем)."""
    return data.get(FSM_KEY_ENTRY_KIND) or KIND_DREAM


async def handle_entry_input(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """ASKING_ENTRY: первая реплика — создаём Dream с kind, переходим в ENTRY_LOOP."""
    if message.from_user is None:
        return
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    data = await state.get_data()
    kind = _fsm_kind(data)
    cfg = entry_config(kind)
    user = await UserRepository(session).get_or_create_by_tg_id(
        telegram_user_id=message.from_user.id
    )
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription=text, kind=kind
    )
    # Имя шага воронки НЕ меняем (B8): тот же onb_dream_told, тип — в метаданных.
    await emit_funnel_step(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_DREAM_TOLD,
        metadata={"dream_id": dream.id, "entry_type": kind},
    )
    await state.update_data(**{_FSM_KEY_ONBOARDING_DREAM_ID: dream.id})
    await state.set_state(EntryStates.ENTRY_LOOP)
    await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)


async def _entry_loop_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    """ENTRY_LOOP: done → развилка онбординг/повтор; continue → append; question → ответ."""
    data = await state.get_data()
    kind = _fsm_kind(data)
    cfg = entry_config(kind)
    res = await _resolve_loop_reply(
        text=text, message=message, session=session, client=client, model=model,
        telegram_user_id=telegram_user_id, statuses=list(cfg.thinking_statuses),
    )
    if res.outcome == "question":
        if res.answer:
            await message.answer(res.answer)
        await state.set_state(EntryStates.ENTRY_LOOP)
        await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)
        return
    if res.outcome == "done":
        user = await UserRepository(session).get_by_tg_id(telegram_user_id)
        if user is None:
            return
        if not UserRepository.is_onboarded(user):
            # Онбординг: знакомство один раз на юзера, независимо от типа (B3 п.3).
            await state.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)
            await message.answer(ASK_LIFE_CONTEXT_ONB, reply_markup=remove_keyboard())
            return
        # Повторное обращение: знакомство не спрашиваем (B6) — сразу сессия и
        # per-session вопрос; дальше общий конвейер quick/image_questions.
        dream_id = data.get(_FSM_KEY_ONBOARDING_DREAM_ID)
        if dream_id is None:
            await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)
            return
        dream = await DreamRepository(session).get_by_id(dream_id, user_id=user.id)
        if dream is None:
            await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)
            return
        sess_repo = SessionRepository(session)
        new_session = await sess_repo.create(
            dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK,
            is_onboarding_session=False,
        )
        if dream.status == DreamStatus.RECORDED:
            dream.status = DreamStatus.IN_ANALYSIS
        await state.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
        await state.set_data({"dream_id": dream.id, "session_id": new_session.id})
        await message.answer(cfg.per_session_question, reply_markup=remove_keyboard())
        log.info(
            "entry.session.created",
            session_id=new_session.id, dream_id=dream.id, user_id=user.id, kind=kind,
        )
        return
    # continue — append к transcription + stay in ENTRY_LOOP
    dream_id = data.get(_FSM_KEY_ONBOARDING_DREAM_ID)
    if dream_id is None:
        await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)
        return
    user = await UserRepository(session).get_or_create_by_tg_id(
        telegram_user_id=telegram_user_id
    )
    dream = await DreamRepository(session).get_by_id(dream_id, user_id=user.id)
    if dream is None:
        await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)
        return
    dream.transcription = f"{dream.transcription}\n\n{text}"
    await state.set_state(EntryStates.ENTRY_LOOP)
    await send_loop_prompt(message=message, state=state, text=cfg.loop_prompt)


async def handle_entry_loop(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """ENTRY_LOOP (текст): гасим старую кнопку; кнопка-текст без LLM; иначе intent под qa_lock."""
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = (message.text or "").strip()
    if not text:
        data = await state.get_data()
        await send_loop_prompt(
            message=message, state=state, text=entry_config(_fsm_kind(data)).loop_prompt
        )
        return
    if text == LOOP_ADD_BTN_DONE:
        await _entry_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _entry_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )


async def handle_entry_loop_done_cb(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """Тап inline «Нет, не хочу» в ENTRY_LOOP → гасим кнопку → done-ветка."""
    if callback.from_user is None or callback.message is None:
        return
    await safe_answer(callback)
    await extinguish_loop_button(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
    )
    await _entry_loop_act(
        text=LOOP_ADD_BTN_DONE, message=callback.message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=callback.from_user.id,
    )


async def handle_entry_loop_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """ENTRY_LOOP + голос: STT (под stt_lock middleware) → intent → append/advance."""
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _entry_loop_act(
        text=text, message=message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )


async def _handle_new_entry_command(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    bot: Bot | None,
    openrouter_classifier_client: AsyncOpenAI | None,
    *,
    kind: str,
) -> None:
    """/new_external_situation | /new_internal_conflict — зеркало /newdream.

    Общее тело (гейты онбординга, авто-пауза активной сессии, выброс черновиков)
    живёт в `_new_entry.start_new_entry`. Отличие от /newdream осталось дальше по
    цепочке: гейта «Разобрать сейчас?» у новых типов нет — после рассказа и loop'а
    сразу per-session вопрос (done-ветка в _entry_loop_act).
    """
    if message.from_user is None:
        return
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None:
        await message.answer(NOT_ONBOARDED_PROMPT)
        return
    if not UserRepository.is_onboarded(user):
        current_state = await state.get_state()
        if current_state is None:
            await message.answer(NOT_ONBOARDED_PROMPT)
            return
    await start_new_entry(
        kind=kind,
        message=message,
        state=state,
        session=session,
        settings=settings,
        bot=bot,
        openrouter_classifier_client=openrouter_classifier_client,
        user=user,
    )


async def handle_new_situation_command(
    message: Message, state: FSMContext, session: AsyncSession,
    settings: Settings, bot: Bot | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    await _handle_new_entry_command(
        message, state, session, settings, bot, openrouter_classifier_client,
        kind=KIND_SITUATION,
    )


async def handle_new_conflict_command(
    message: Message, state: FSMContext, session: AsyncSession,
    settings: Settings, bot: Bot | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    await _handle_new_entry_command(
        message, state, session, settings, bot, openrouter_classifier_client,
        kind=KIND_CONFLICT,
    )


def create_router() -> Router:
    """Роутер EntryStates. Команды исключены из текстовых хэндлеров (B7 п.4 —
    новый флоу не наследует дефект онбординга, глотавшего /dreams как текст)."""
    router = Router(name="entry_flow")
    router.message.register(
        handle_new_situation_command, Command("new_external_situation")
    )
    router.message.register(
        handle_new_conflict_command, Command("new_internal_conflict")
    )
    router.message.register(
        handle_entry_input,
        EntryStates.ASKING_ENTRY,
        (F.text & ~F.text.startswith("/")) | F.voice,
    )
    router.callback_query.register(
        handle_entry_loop_done_cb, EntryStates.ENTRY_LOOP, F.data == LOOP_DONE_CALLBACK
    )
    router.message.register(
        handle_entry_loop, EntryStates.ENTRY_LOOP, F.text & ~F.text.startswith("/")
    )
    router.message.register(handle_entry_loop_voice, EntryStates.ENTRY_LOOP, F.voice)
    return router
