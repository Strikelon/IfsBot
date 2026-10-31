"""Онбординг — FSM-состояния заполнения профиля + Quick + опц. Deep.

Текстовые и голосовые обработчики. Голос принимается в ASKING_LIFE_CONTEXT
и ASKING_DREAM через Whisper.
"""

from __future__ import annotations

from datetime import UTC
from datetime import datetime as dt

import structlog
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    FSM_KEY_ENTRY_KIND,
    note_content_filter,
    record_context_usage,
    record_intent_usage,
)
from adapters.telegram.handlers._loop_prompt import (
    extinguish_loop_button,
    send_loop_prompt,
)
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.handlers.quick import (
    _build_quick_context,
    _entry_kind_for_session,
    _run_generator,
)
from adapters.telegram.keyboards import (
    LOOP_DONE_CALLBACK,
    remove_keyboard,
)
from adapters.telegram.spend_guard import exit_with_paywall
from adapters.telegram.states import EntryStates, OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import (
    ANALYSIS_START_NOTICE,
    ASK_LIFE_CONTEXT_ONB,
    LOOP_ADD_BTN_DONE,
    LOOP_ADD_PROMPT_DREAM,
    LOOP_ADD_PROMPT_LIFE,
    PROFILE_THINKING_STATUSES,
    QUICK_LLM_FAILED,
    THINKING_GUARD_NOTICE,
    llm_error_message,
)
from adapters.telegram.thinking_status import thinking_status
from adapters.telegram.voice_helper import transcribe_with_placeholder
from app.config import Settings
from data.analytics_events import AnalyticsEventType, emit_funnel_step
from data.models import DreamStatus, RequestBranch, SessionType, User
from data.repositories import (
    DreamRepository,
    SessionRepository,
    UsageEventRepository,
    UserRepository,
)
from domain.entry_types import KIND_DREAM, entry_config
from llm.context_compact import (
    COMPACT_TRIGGER_CHARS,
    CONTEXT_COMPACT_VERSION,
    compact_life_context,
)
from llm.generator import _QUICK_FALLBACK_CHAIN, ErrorCategory
from llm.intent import (
    INTENT_LOOP_VERSION,
    IntentResult,
    classify_loop_intent,
)
from llm.prompts.entry_prompts import prompt_bundle
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)

_FSM_KEY_ONBOARDING_DREAM_ID = "onboarding_dream_id"
_FSM_KEY_ONBOARDING_SESSION_ID = "onboarding_session_id"


async def _transcribe_message_voice(
    message: Message,
    bot: Bot,
    openai_client: AsyncOpenAI,
    usage_recorder: UsageRecorder,
    user_id: int,
    session: AsyncSession,
    settings: Settings,
) -> str | None:
    """Скачать voice → Whisper (с плейсхолдером) → записать usage. Возвращает текст или None."""
    result = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client, log_prefix="onboarding",
        session=session, settings=settings,
    )
    if result is None:
        return None
    await usage_recorder.record_whisper(user_id=user_id, audio_seconds=result.duration_seconds)
    return result.text


async def _get_text_from_message(
    message: Message,
    bot: Bot,
    openai_client: AsyncOpenAI,
    session: AsyncSession,
    settings: Settings,
) -> str | None:
    """Извлечь текст из text- или voice-сообщения.

    Возвращает None если ни text, ни voice, или транскрипция упала
    (в этом случае пользователь уже видел сообщение об ошибке).

    stt_user_lock для voice захватывается в BusyGuardMiddleware на ВЕСЬ
    handler — здесь дополнительно не лочим.
    """
    if message.from_user is None:
        return None
    if message.text is not None:
        return message.text.strip()
    if message.voice is not None:
        repo = UserRepository(session)
        user = await repo.get_or_create_by_tg_id(telegram_user_id=message.from_user.id)
        recorder = UsageRecorder(UsageEventRepository(session))
        return await _transcribe_message_voice(
            message, bot, openai_client, recorder, user.id, session, settings
        )
    return None


async def handle_life_context_input(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """ASKING_LIFE_CONTEXT: первая реплика юзера. Сохраняем и выводим loop-add prompt."""
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=message.from_user.id)
    await repo.update_profile(user_id=user.id, profile_life_context=text)
    kind = (await state.get_data()).get(FSM_KEY_ENTRY_KIND) or KIND_DREAM
    await emit_funnel_step(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_LIFE_CONTEXT,
        metadata={"entry_type": kind},
    )
    await state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_LIFE)


async def _resolve_loop_reply(
    *,
    text: str,
    message: Message,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
    statuses: list[str] | None = None,
) -> IntentResult:
    """Классифицировать реплику в loop: done / continue / question.

    Точная кнопка «Это всё»/«Нет, не хочу» — без LLM (синтетический done). Любой
    другой текст/голос — через дешёвый intent-классификатор (дефолт continue при
    сбое, чтобы не терять контент). Расход пишется в usage (session_id=None).

    `statuses` — кастомный набор фраз плейсхолдера (для шага знакомства — про
    человека, не про сны).
    """
    if text == LOOP_ADD_BTN_DONE:
        return IntentResult(outcome="done", input_tokens=0, output_tokens=0)
    async with thinking_status(message, statuses=statuses):
        res = await classify_loop_intent(client=client, model=model, user_text=text)
    await record_intent_usage(
        session=session, telegram_user_id=telegram_user_id, session_id=None,
        model=model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        prompt_version=INTENT_LOOP_VERSION,
    )
    return res


async def _life_context_loop_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    """Общая логика LIFE_CONTEXT_LOOP: done/continue/question → продвинуть/дописать/ответить.

    `message` используется только для ответов (в callback-пути это
    `callback.message`), юзер берётся по явному `telegram_user_id`.
    """
    repo = UserRepository(session)
    res = await _resolve_loop_reply(
        text=text, message=message, session=session, client=client, model=model,
        telegram_user_id=telegram_user_id, statuses=PROFILE_THINKING_STATUSES,
    )
    if res.outcome == "question":
        # Встречный вопрос — отвечаем и повторяем тот же вопрос, ничего не записывая.
        if res.answer:
            await message.answer(res.answer)
        await state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_LIFE)
        return
    if res.outcome == "done":
        user = await repo.get_by_tg_id(telegram_user_id)
        # Guard на размер: если за loop контекст разросся — компоновка (не обрезание),
        # чтобы держать profile_life_context компактным. Падёт — оставляем как есть.
        if user and user.profile_life_context and len(user.profile_life_context) > COMPACT_TRIGGER_CHARS:
            comp = await compact_life_context(
                client=client, model=model, text=user.profile_life_context
            )
            if comp.changed and comp.new_text:
                await repo.update_profile(user_id=user.id, profile_life_context=comp.new_text)
            await record_context_usage(
                session=session, telegram_user_id=telegram_user_id, session_id=None,
                model=model, input_tokens=comp.input_tokens, output_tokens=comp.output_tokens,
                prompt_version=CONTEXT_COMPACT_VERSION,
            )
        # Знакомство завершено → per-session вопрос перед разбором (per-kind;
        # для dream текст байт-в-байт прежний ASK_PER_SESSION_CONTEXT).
        kind = (await state.get_data()).get(FSM_KEY_ENTRY_KIND) or KIND_DREAM
        await state.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
        await message.answer(
            entry_config(kind).per_session_question, reply_markup=remove_keyboard()
        )
        return
    # continue — append к profile_life_context + stay in LIFE_CONTEXT_LOOP
    user = await repo.get_or_create_by_tg_id(telegram_user_id=telegram_user_id)
    existing = user.profile_life_context or ""
    combined = f"{existing}\n\n{text}" if existing else text
    await repo.update_profile(user_id=user.id, profile_life_context=combined)
    await state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_LIFE)


async def handle_life_context_loop(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """LIFE_CONTEXT_LOOP (текст): новое сообщение гасит старую кнопку; «всё» → ASKING_DREAM; иначе append + stay.

    Свободный текст-ввод проходит через intent-классификатор внутри qa_lock —
    второе сообщение во время классификации ловит busy-notice (защита от спама).
    """
    if message.from_user is None:
        return
    # Юзер прислал реплику → гасим устаревшую inline-кнопку «Это всё».
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = (message.text or "").strip()
    if not text:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_LIFE)
        return
    if text == LOOP_ADD_BTN_DONE:
        # Точный текст «Это всё» — без LLM, lock не нужен.
        await _life_context_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _life_context_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )


async def handle_life_context_loop_done_cb(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """Тап inline «Это всё» в LIFE_CONTEXT_LOOP → гасим кнопку → ASKING_DREAM."""
    if callback.from_user is None or callback.message is None:
        return
    await safe_answer(callback)
    await extinguish_loop_button(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
    )
    await _life_context_loop_act(
        text=LOOP_ADD_BTN_DONE, message=callback.message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=callback.from_user.id,
    )


async def handle_life_context_loop_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """LIFE_CONTEXT_LOOP + голос: гасим старую кнопку → STT → intent → append/advance.

    Весь handler идёт под stt_lock (захвачен middleware на voice), поэтому
    STT + классификатор + ответ атомарны — отдельный qa_lock не нужен.
    """
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _life_context_loop_act(
        text=text, message=message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )



async def handle_dream_input(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """ASKING_DREAM: первая реплика юзера. Создаём Dream record, переходим к DREAM_LOOP."""
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=message.from_user.id)

    dream_repo = DreamRepository(session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=dt.now(UTC),
        transcription=text,
    )
    await emit_funnel_step(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.ONB_DREAM_TOLD,
        # entry_type и здесь: этот шаг воронки у ситуации/конфликта уходит из
        # entry_flow с типом, и разрез был бы дырявым именно на снах.
        metadata={"dream_id": dream.id, "entry_type": KIND_DREAM},
    )
    await state.update_data(**{_FSM_KEY_ONBOARDING_DREAM_ID: dream.id})
    await state.set_state(OnboardingStates.DREAM_LOOP)
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)


async def _dream_loop_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    """Общая логика DREAM_LOOP онбординга: done → per-session; continue → append.

    `message` — только для ответов (в callback-пути `callback.message`),
    юзер берётся по явному `telegram_user_id`.
    """
    res = await _resolve_loop_reply(
        text=text, message=message, session=session, client=client, model=model,
        telegram_user_id=telegram_user_id,
    )
    if res.outcome == "question":
        if res.answer:
            await message.answer(res.answer)
        await state.set_state(OnboardingStates.DREAM_LOOP)
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    if res.outcome == "done":
        # Сон рассказан → шаг знакомства (рассказ о себе).
        await state.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)
        await message.answer(ASK_LIFE_CONTEXT_ONB, reply_markup=remove_keyboard())
        return
    # continue — append к Dream.transcription + stay in DREAM_LOOP
    data = await state.get_data()
    dream_id = data.get(_FSM_KEY_ONBOARDING_DREAM_ID)
    if dream_id is None:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    user_repo = UserRepository(session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=telegram_user_id)
    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    if dream is None:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    dream.transcription = f"{dream.transcription}\n\n{text}"
    await state.set_state(OnboardingStates.DREAM_LOOP)
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)


async def handle_dream_loop(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """DREAM_LOOP (текст): новое сообщение гасит старую кнопку; «всё» → per-session; иначе append.

    Классификатор под qa_lock.
    """
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = (message.text or "").strip()
    if not text:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    if text == LOOP_ADD_BTN_DONE:
        await _dream_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _dream_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )


async def handle_dream_loop_done_cb(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """Тап inline «Это всё» в DREAM_LOOP → гасим кнопку → ASKING_PER_SESSION_CONTEXT_ONB."""
    if callback.from_user is None or callback.message is None:
        return
    await safe_answer(callback)
    await extinguish_loop_button(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
    )
    await _dream_loop_act(
        text=LOOP_ADD_BTN_DONE, message=callback.message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=callback.from_user.id,
    )


async def handle_dream_loop_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """DREAM_LOOP + голос: гасим старую кнопку → STT → intent → append/advance (под stt_lock)."""
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _dream_loop_act(
        text=text, message=message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )



async def handle_per_session_context_onb(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    settings: Settings,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """ASKING_PER_SESSION_CONTEXT_ONB: собрать per-session контекст и сразу запустить разбор.

    Шаг выбора ветки (6 кнопок) убран — ставим дефолт `RequestBranch.UNCLEAR`.
    """
    if message.from_user is None:
        return
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None:
        return
    dream_id = (await state.get_data()).get(_FSM_KEY_ONBOARDING_DREAM_ID)
    if dream_id is None:
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await emit_funnel_step(
            session,
            user_id=user.id,
            telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.ONB_INTENT_GIVEN,
            metadata={
                "dream_id": dream_id,
                "entry_type": (await state.get_data()).get(FSM_KEY_ENTRY_KIND)
                or KIND_DREAM,
            },
        )
        await _start_onboarding_quick(
            message=message,
            state=state,
            session=session,
            user=user,
            dream_id=dream_id,
            per_session_context=text,
            branch=RequestBranch.UNCLEAR,
            custom_text=None,
            openrouter_client=openrouter_client,
            context_client=openrouter_classifier_client,
            context_model=openrouter_context_model,
            settings=settings,
        )


async def _start_onboarding_quick(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    dream_id: int,
    per_session_context: str,
    branch: RequestBranch,
    custom_text: str | None,
    openrouter_client: AsyncOpenAI,
    context_client: AsyncOpenAI,
    context_model: str,
    settings: Settings,
) -> None:
    """Создать онбординг-Quick-сессию (+кастомный запрос если есть) и запустить разбор."""
    # Spend cap: над лимитом не запускаем платный онбординг-разбор — помечаем
    # онбординг завершённым (чтобы не зациклить) и показываем пейвол.
    if await exit_with_paywall(message, state, session, user, settings, mark_onboarded=True):
        return
    user_id = user.id
    sess_repo = SessionRepository(session)
    new_session = await sess_repo.create(
        dream_id=dream_id,
        user_id=user_id,
        type_=SessionType.QUICK,
        is_onboarding_session=True,
    )
    await sess_repo.set_per_session_context(
        session_id=new_session.id, context=per_session_context
    )
    await sess_repo.set_request_branch(session_id=new_session.id, branch=branch)

    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user_id)
    if dream is None:
        return
    if dream.status == DreamStatus.RECORDED:
        dream.status = DreamStatus.IN_ANALYSIS

    await state.update_data(**{_FSM_KEY_ONBOARDING_SESSION_ID: new_session.id})
    if settings.image_questions_enabled:
        # Вопросы по образам ПЕРЕД первым разбором (общий оркестратор с обычным
        # потоком; is_onboarding=True добавляет funnel-шаги onb_img_q_*).
        # Приглашение идёт с inline-клавой, поэтому remove_keyboard() к нему
        # прикрепить нельзя (один reply_markup на сообщение) — снятие reply-клавы
        # остаётся за ANALYSIS_START_NOTICE при фактическом старте разбора.
        from adapters.telegram.handlers.image_questions import start_image_questions_flow

        await start_image_questions_flow(
            message=message, state=state, session=session, user=user,
            session_id=new_session.id, dream_id=dream_id, is_onboarding=True,
            settings=settings, bot=None,
        )
        return
    await _run_quick_for_onboarding(
        message=message,
        state=state,
        session=session,
        user_id=user_id,
        telegram_user_id=user.telegram_user_id,
        session_id=new_session.id,
        dream_id=dream_id,
        openrouter_client=openrouter_client,
        context_client=context_client,
        context_model=context_model,
        settings=settings,
    )


async def _run_quick_for_onboarding(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user_id: int,
    telegram_user_id: int,
    session_id: int,
    dream_id: int,
    openrouter_client: AsyncOpenAI,
    context_client: AsyncOpenAI,
    context_model: str,
    settings: Settings,
    openrouter_generator_model: str = "",  # deprecated, ignored — generator owns chain
) -> None:
    """Build Quick context → call generator → log usage → handoff to IN_QA.

    Onboarding completes only on user's first «Завершить» tap
    (handled in handle_finish_quick_reply via is_onboarding_session check).
    """
    ctx = await _build_quick_context(
        session=session, user_id=user_id, session_id=session_id, settings=settings
    )
    if ctx is None:
        await message.answer(QUICK_LLM_FAILED)
        return

    kind = await _entry_kind_for_session(session, user_id=user_id, session_id=session_id)
    cfg = entry_config(kind)
    bundle = prompt_bundle(kind)

    # Снимаем reply-клаву (от LIFE_CONTEXT_LOOP / DREAM_LOOP) отдельным сообщением
    # ДО thinking_status. Если бы передали reply_markup=remove_keyboard() в сам
    # thinking_status, Telegram бы запретил editMessageText на placeholder
    # (BUG #1 round 2, 2026-05-21: «message can't be edited» на каждую ротацию).
    try:
        await message.answer(ANALYSIS_START_NOTICE, reply_markup=remove_keyboard())
    except Exception:  # noqa: BLE001
        log.debug("onboarding.remove_keyboard_failed", exc_info=True)

    async with thinking_status(
        message,
        state=state,
        thinking_state=OnboardingStates.GENERATING_INITIAL_ONB,
        statuses=list(cfg.thinking_statuses),
    ):
        try:
            result = await _run_generator(
                ctx=ctx,
                qa_history=[],
                user_followup=None,
                openrouter_client=openrouter_client,
                kind=kind,
            )
        except Exception:  # noqa: BLE001
            log.exception("onboarding.quick.generator.failed", session_id=session_id)
            await message.answer(QUICK_LLM_FAILED)
            return

    if not result.text:
        if result.error_category == ErrorCategory.CONTENT_FILTER:
            # Зарубили все модели. Онбординг не завершён и завершиться не может –
            # возвращаем юзера на первый шаг (рассказ сна). Текст ошибки сам
            # работает как приглашение, повторное приветствие не шлём.
            await note_content_filter(
                session=session,
                session_id=session_id,
                where="onboarding",
                user_id=user_id,
                dream_id=dream_id,
                pause=True,
            )
            if kind == KIND_DREAM:
                await state.set_state(OnboardingStates.ASKING_DREAM)
                await state.set_data({})
            else:
                # Возврат на первый шаг СВОЕЙ ветки: рассказ заново, kind сохраняем.
                await state.set_state(EntryStates.ASKING_ENTRY)
                await state.set_data({FSM_KEY_ENTRY_KIND: kind})
            await message.answer(cfg.cf_onboarding, reply_markup=remove_keyboard())
            return
        await message.answer(llm_error_message(result.error_category))
        return

    recorder = UsageRecorder(UsageEventRepository(session))
    await recorder.record_openrouter_generator(
        user_id=user_id,
        session_id=session_id,
        model=result.model_used or _QUICK_FALLBACK_CHAIN[0],
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        prompt_version=bundle.version_initial,
    )
    # Handoff to QuickAnalysisStates.IN_QA BEFORE delivery so that
    # _deliver_analysis's update_data(qa_options_msg_id=...) is not wiped.
    # set_data (replace) runs here once; _deliver_analysis then adds
    # qa_options_msg_id on top via update_data (merge).
    await state.set_state(QuickAnalysisStates.IN_QA)
    await emit_funnel_step(
        session,
        user_id=user_id,
        telegram_user_id=telegram_user_id,
        event_type=AnalyticsEventType.ONB_ANALYSIS_DELIVERED,
        metadata={"session_id": session_id, "entry_type": kind},
    )
    await state.set_data({
        "session_id": session_id,
        "dream_id": dream_id,
    })
    # Deliver analysis: split tail → log prose-only → send with inline chips.
    # Mirrors _run_initial_quick / follow-up delivery in quick.py.
    from adapters.telegram.handlers.quick import _deliver_analysis
    from adapters.telegram.session_log import QUICK_PHASE_INITIAL

    await _deliver_analysis(
        message=message,
        state=state,
        session=session,
        bot=None,  # nothing to extinguish on first delivery
        session_id=session_id,
        raw_text=result.text,
        phase_label=QUICK_PHASE_INITIAL,
    )
    log.info("onboarding.handoff.in_qa", session_id=session_id, dream_id=dream_id)


def create_router() -> Router:
    """Создать новый экземпляр Router с зарегистрированными хэндлерами."""
    router = Router(name="onboarding")
    router.message.register(
        handle_life_context_input,
        OnboardingStates.ASKING_LIFE_CONTEXT,
        F.text | F.voice,
    )
    router.callback_query.register(
        handle_life_context_loop_done_cb,
        OnboardingStates.LIFE_CONTEXT_LOOP,
        F.data == LOOP_DONE_CALLBACK,
    )
    router.message.register(
        handle_life_context_loop,
        OnboardingStates.LIFE_CONTEXT_LOOP,
        F.text,
    )
    router.message.register(
        handle_life_context_loop_voice,
        OnboardingStates.LIFE_CONTEXT_LOOP,
        F.voice,
    )
    router.message.register(handle_dream_input, OnboardingStates.ASKING_DREAM, F.text | F.voice)
    router.callback_query.register(
        handle_dream_loop_done_cb,
        OnboardingStates.DREAM_LOOP,
        F.data == LOOP_DONE_CALLBACK,
    )
    router.message.register(handle_dream_loop, OnboardingStates.DREAM_LOOP, F.text)
    router.message.register(handle_dream_loop_voice, OnboardingStates.DREAM_LOOP, F.voice)
    router.message.register(
        handle_per_session_context_onb,
        OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB,
        F.text | F.voice,
    )
    return router
