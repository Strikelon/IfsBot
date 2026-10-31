"""Экран выбора типа обращения (спека B3): 3 кнопки + свободный текст/голос.

Кнопка — без LLM. Текст/голос → classify_entry_type_intent (gpt-4o-mini):
dream/external_situation/internal_conflict → ветка типа; question → ответ и снова
кнопки; unclear → «Не понял вас…» и снова кнопки. Во время классификации —
thinking_status (человеческие фразы, не сновые); параллельный спам ловят
qa_user_lock / stt_lock (BusyGuardMiddleware).

Если классификатор пометил реплику как уже РАССКАЗ (`told`), шаг «расскажите»
перепрыгивается: рассказ сохраняется как обращение и юзер сразу попадает в loop
«хотите добавить ещё».
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as dt
from html import escape as html_escape

import structlog
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    FSM_KEY_ENTRY_KIND,
    record_intent_usage,
)
from adapters.telegram.handlers._loop_prompt import (
    extinguish_inline_kb,
    send_loop_prompt,
)
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.handlers.onboarding import _FSM_KEY_ONBOARDING_DREAM_ID
from adapters.telegram.keyboards import (
    ENTRY_TYPE_CALLBACK_PREFIX,
    entry_type_keyboard,
)
from adapters.telegram.quick_markup import render_quick_markup
from adapters.telegram.states import EntryStates, OnboardingStates
from adapters.telegram.texts import (
    ENTRY_CAPTURED_MAX_CHARS,
    ENTRY_CAPTURED_TMPL,
    ENTRY_CHOICE_REPEAT,
    ENTRY_CHOICE_UNCLEAR,
    PROFILE_THINKING_STATUSES,
    THINKING_GUARD_NOTICE,
)
from adapters.telegram.thinking_status import thinking_status
from app.config import Settings
from data.analytics_events import AnalyticsEventType, emit_funnel_step
from data.models import User
from data.repositories import DreamRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_DREAM, entry_config
from llm.intent import INTENT_ENTRY_TYPE_VERSION, classify_entry_type_intent

log = structlog.get_logger(__name__)

# FSM-ключ message_id экрана выбора (гашение кнопок по паттерну _loop_prompt).
_FSM_KEY_CHOICE_MSG_ID = "entry_choice_msg_id"


async def _reshow_choice(message: Message, state: FSMContext, text: str) -> None:
    """Переспрос: text + те же 3 кнопки; message_id запоминаем для гашения."""
    sent = await message.answer(text, reply_markup=entry_type_keyboard())
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(**{_FSM_KEY_CHOICE_MSG_ID: msg_id})


def _as_quote(text: str) -> str:
    """Текст юзера → строки-цитаты «> » (render_quick_markup склеит их в blockquote)."""
    trimmed = text.strip()
    if len(trimmed) > ENTRY_CAPTURED_MAX_CHARS:
        trimmed = trimmed[:ENTRY_CAPTURED_MAX_CHARS].rstrip() + "…"
    return "\n".join(f"> {line.strip()}" for line in trimmed.splitlines() if line.strip())


async def _capture_told_entry(
    *,
    kind: str,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
) -> None:
    """Юзер сразу рассказал историю: создать обращение, показать его и уйти в loop.

    Полный аналог `handle_dream_input` (onboarding.py), только точка входа — экран
    выбора: та же запись в dreams, то же funnel-событие, тот же FSM-ключ dream_id и тот
    же loop-каркас. Для сна состояние — существующее `OnboardingStates.DREAM_LOOP`,
    для новых типов — `EntryStates.ENTRY_LOOP` (B7: существующие имена не трогаем).
    """
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription=text, kind=kind
    )
    await emit_funnel_step(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_DREAM_TOLD,
        metadata={"dream_id": dream.id, "entry_type": kind},
    )
    await state.update_data(**{_FSM_KEY_ONBOARDING_DREAM_ID: dream.id})
    if kind == KIND_DREAM:
        await state.set_state(OnboardingStates.DREAM_LOOP)
    else:
        await state.set_state(EntryStates.ENTRY_LOOP)
        await state.update_data(**{FSM_KEY_ENTRY_KIND: kind})
    await send_loop_prompt(
        message=message,
        state=state,
        text=render_quick_markup(ENTRY_CAPTURED_TMPL.format(quote=_as_quote(text))),
    )
    log.info("entry_choice.told_captured", kind=kind, dream_id=dream.id, user_id=user.id)


async def _apply_choice(
    *,
    kind: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    told_text: str | None = None,
) -> None:
    """Выбор сделан: событие + переход в ветку типа (спека B3 «Дальше по типу»).

    `told_text` — рассказ, который юзер прислал ВМЕСТО нажатия кнопки: тогда шаг
    «расскажите» перепрыгиваем. Тап по кнопке всегда
    передаёт None: кнопка содержания не несёт.
    """
    cfg = entry_config(kind)
    await emit_funnel_step(  # идемпотентно (once-per-user), имя шага — onb_type_chosen
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_TYPE_CHOSEN,
        metadata={"entry_type": kind, "told": told_text is not None},
    )
    if told_text is not None:
        await _capture_told_entry(
            kind=kind, text=told_text, message=message, state=state,
            session=session, user=user,
        )
        return
    invite = cfg.onb_invite.format(name=html_escape(user.display_name or "друг"))
    if kind == KIND_DREAM:
        # Ветка сна — существующие состояние и хэндлеры, без изменений (B3).
        await state.set_state(OnboardingStates.ASKING_DREAM)
        await message.answer(invite)
        return
    await state.set_state(EntryStates.ASKING_ENTRY)
    await state.update_data(**{FSM_KEY_ENTRY_KIND: kind})
    await message.answer(invite)
    log.info("entry_choice.applied", kind=kind, user_id=user.id)


async def handle_entry_choice_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot | None = None,
) -> None:
    """Тап одной из трёх кнопок: entry_type:<kind>."""
    if callback.from_user is None or callback.message is None or callback.data is None:
        return
    await safe_answer(callback)
    kind = callback.data.removeprefix(ENTRY_TYPE_CALLBACK_PREFIX)
    if kind not in ENTRY_TYPES:
        return
    await extinguish_inline_kb(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
        fsm_key=_FSM_KEY_CHOICE_MSG_ID,
    )
    user = await UserRepository(session).get_or_create_by_tg_id(
        telegram_user_id=callback.from_user.id
    )
    await _apply_choice(
        kind=kind, message=callback.message, state=state, session=session, user=user
    )


async def _choice_text_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot | None,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    user = await UserRepository(session).get_or_create_by_tg_id(
        telegram_user_id=telegram_user_id
    )
    await extinguish_inline_kb(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None,
        fsm_key=_FSM_KEY_CHOICE_MSG_ID,
    )
    async with thinking_status(message, statuses=PROFILE_THINKING_STATUSES):
        res = await classify_entry_type_intent(client=client, model=model, user_text=text)
    await record_intent_usage(
        session=session, telegram_user_id=telegram_user_id, session_id=None,
        model=model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        prompt_version=INTENT_ENTRY_TYPE_VERSION,
    )
    if res.outcome in ENTRY_TYPES:
        await _apply_choice(
            kind=res.outcome, message=message, state=state, session=session, user=user,
            told_text=text if res.told else None,
        )
        return
    if res.outcome == "question":
        # Ответили — и снова показываем три кнопки (спека B3).
        if res.answer:
            await message.answer(res.answer)
        await _reshow_choice(message, state, ENTRY_CHOICE_REPEAT)
        return
    # unclear
    await _reshow_choice(message, state, ENTRY_CHOICE_UNCLEAR)


async def handle_entry_choice_text(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    if message.from_user is None or message.text is None:
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _choice_text_act(
            text=message.text.strip(), message=message, state=state, session=session,
            bot=bot, client=openrouter_classifier_client,
            model=openrouter_intent_model, telegram_user_id=message.from_user.id,
        )


async def handle_entry_choice_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """Голос на экране выбора: STT (под stt_lock middleware) → тот же act."""
    if message.from_user is None or message.voice is None:
        return
    from adapters.telegram.handlers.onboarding import _get_text_from_message

    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _choice_text_act(
        text=text, message=message, state=state, session=session, bot=bot,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )


def create_router() -> Router:
    router = Router(name="entry_choice")
    router.callback_query.register(
        handle_entry_choice_callback,
        OnboardingStates.CHOOSING_ENTRY_TYPE,
        F.data.startswith(ENTRY_TYPE_CALLBACK_PREFIX),
    )
    router.message.register(
        handle_entry_choice_text,
        OnboardingStates.CHOOSING_ENTRY_TYPE,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_entry_choice_voice, OnboardingStates.CHOOSING_ENTRY_TYPE, F.voice
    )
    return router
