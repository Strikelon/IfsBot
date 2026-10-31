"""Quick analysis FSM-handlers (Phase 2 + quick.v2.1).

Состояния:
    ASKING_PER_SESSION_CONTEXT — юзер отвечает текстом или голосом, затем сразу разбор
    IN_QA                      — юзер задаёт уточнения, бот отвечает (без лимита)

В state.data: {dream_id, session_id}. qa_history больше не в FSM —
source of truth теперь session_messages (E2 из quick.v2.1).
"""

from __future__ import annotations

from html import escape as html_escape

import structlog
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message, ReplyKeyboardRemove
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    entry_kind_for_dream,
    note_content_filter,
    record_intent_usage,
)
from adapters.telegram.handlers._loop_prompt import extinguish_inline_kb
from adapters.telegram.handlers._memory import run_memory_extraction_on_pause
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.keyboards import (
    QA_CALLBACK_CUSTOM,
    QA_CALLBACK_FINISH,
    QA_CALLBACK_SKIP,
    dream_actions_keyboard,
    qa_options_keyboard,
    remove_keyboard,
)
from adapters.telegram.menu import sync_user_commands
from adapters.telegram.quick_markup import render_quick_markup
from adapters.telegram.sender import answer_long, split_for_telegram
from adapters.telegram.session_log import (
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_QA,
    load_quick_qa_history,
    log_bot_message,
    log_user_message,
)
from adapters.telegram.spend_guard import deny_entry, exit_with_paywall, is_over_limit
from adapters.telegram.states import DreamCardStates, QuickAnalysisStates
from adapters.telegram.texts import (
    DREAM_ACTION_START_ANALYSIS,
    END_OF_ONBOARDING_TEXT,
    FINISH_BUTTON_TEXT,
    NOT_ONBOARDED_PROMPT,
    QA_CONFIRM_YES_BTN,
    QA_CUSTOM_QUESTION_PROMPT,
    QUICK_CONTINUE_RESUMED_INTRO,
    QUICK_LLM_FAILED,
    QUICK_USE_BUTTONS,
    SESSION_PAUSED_NOTICE,
    THINKING_GUARD_NOTICE,
    llm_error_message,
)
from adapters.telegram.thinking_status import thinking_status
from adapters.telegram.voice_helper import transcribe_with_placeholder
from app.config import Settings
from data.analytics_events import AnalyticsEventType, emit_funnel_step
from data.models import (
    Dream,
    DreamStatus,
    RequestBranch,
    SessionMessageContentType,
    SessionStatus,
    SessionType,
    User,
)
from data.repositories import (
    DreamRepository,
    SessionMessageRepository,
    SessionRepository,
    SessionStateRepository,
    UsageEventRepository,
    UserMemoryItemRepository,
    UserRepository,
)
from domain.entry_types import KIND_DREAM, entry_config
from llm.generator import _QUICK_FALLBACK_CHAIN, ErrorCategory, generate_quick_response
from llm.intent import (
    INTENT_CONFIRM_VERSION,
    INTENT_QA_VERSION,
    classify_confirm_intent,
    classify_qa_intent,
)
from llm.memory_select import build_memory_block
from llm.prompts.entry_prompts import prompt_bundle
from llm.prompts.quick import (
    QuickContext,
    split_qa_options,
)
from llm.prompts.quick_summary import SUMMARY_PROMPT_VERSION
from llm.quick_summarizer import (
    SUMMARY_KEY,
    compute_summary,
    should_summarize,
    split_history_for_summary,
)
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)

# Ключ в session_state (JSON) для свободного запроса юзера на ветке (RequestBranch.CUSTOM).
# Шаг выбора ветки убран из воронки, но ключ оставлен для совместимости со старыми сессиями.
CUSTOM_BRANCH_KEY = "custom_branch_request"
# Ключ в FSM-data для dream_id открытой карточки сна (DreamCardStates.DECIDING_ANALYZE).
_DREAM_CARD_DREAM_ID = "dream_card_dream_id"
# session_state: лейблы+kind последней QA-клавы (для возврата через /dreams).
QA_OPTIONS_KEY = "qa_options"
# followup-сигнал, когда юзер пропускает уточняющий вопрос — генератор отвечает из того, что уже есть.
QA_SKIP_FOLLOWUP = "(пропускаю этот вопрос – ответьте из того, что уже есть)"
# session_state: карта беседы IFS (focus/type/stage/covered), пишется генератором хвостом.
QA_MAP_KEY = "qa_map"
# FSM: message_id последней QA-клавы (для гашения через extinguish_inline_kb).
_FSM_KEY_QA_MSG_ID = "qa_options_msg_id"


def _labels_from_split(kind: str, options: list[str]) -> list[str]:
    """Лейблы чипов для клавы: confirm → ["Да"]; choice → options (может быть [])."""
    return [QA_CONFIRM_YES_BTN] if kind == "confirm" else options


# === Callback: тап «Разобрать сейчас» на карточке сна ===


async def handle_start_analysis_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
) -> None:
    """callback_data = 'start_analysis:<dream_id>'. Создаёт сессию и спрашивает per-session context."""
    if callback.from_user is None or callback.data is None or callback.message is None:
        return

    await safe_answer(callback)  # убираем «часики» с кнопки

    try:
        dream_id = int(callback.data.split(":", 1)[1])
    except (IndexError, ValueError):
        log.warning("quick.start_analysis.bad_callback", data=callback.data)
        return

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(callback.from_user.id)
    if user is None or not UserRepository.is_onboarded(user):
        await callback.message.answer(NOT_ONBOARDED_PROMPT)
        return

    # Spend cap: блокируем вход в разбор над лимитом (пейвол, в анализ не входим).
    if await deny_entry(callback, session, user, settings):
        return

    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    if dream is None:
        return  # сон не его — молча игнорим

    sess_repo = SessionRepository(session)
    active = await sess_repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id)
    if active is not None:
        await callback.message.answer(entry_config(dream.kind).already_active)
        return

    # Если у этого сна уже есть какая-либо завершённая Quick-сессия — кнопка
    # «Разобрать сейчас» на карточке /dream_<id> не показывается, юзер видит
    # вход в Q&A. Stale-callback может прилететь только из старого сообщения —
    # отвечаем «уже идёт разбор» по типу обращения, чтобы юзер понимал что произошло.
    if await sess_repo.latest_inactive_quick_for_dream(dream_id=dream.id, user_id=user.id):
        await callback.message.answer(entry_config(dream.kind).already_active)
        return

    await _start_new_quick_session(
        callback_message=callback.message,
        state=state,
        session=session,
        sess_repo=sess_repo,
        user_id=user.id,
        dream=dream,
    )


async def _start_new_quick_session(
    *,
    callback_message: Message,
    state: FSMContext,
    session: AsyncSession,
    sess_repo: SessionRepository,
    user_id: int,
    dream: Dream,
) -> None:
    """Создать новую Quick-сессию и перейти в ASKING_PER_SESSION_CONTEXT."""
    new_session = await sess_repo.create(
        dream_id=dream.id, user_id=user_id, type_=SessionType.QUICK
    )
    dream.status = DreamStatus.IN_ANALYSIS

    log.info(
        "quick.session.created",
        session_id=new_session.id,
        dream_id=dream.id,
        user_id=user_id,
    )

    await state.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await state.set_data({"dream_id": dream.id, "session_id": new_session.id})

    await callback_message.answer(entry_config(dream.kind).per_session_question)


# === DreamCardStates: свободный текст/голос на карточке сна «Разобрать сейчас» ===


async def _dream_card_decide_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    settings: Settings,
) -> None:
    """Классифицируем реплику на карточке сна (analyze/save/unclear) и действуем.

    analyze → запускаем разбор этого сна; save → оставляем сохранённым; unclear →
    переспрашиваем кнопкой. Прогресс-бар через thinking_status.
    """
    if message.from_user is None:
        return
    data = await state.get_data()
    dream_id = data.get(_DREAM_CARD_DREAM_ID)
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None or dream_id is None:
        await state.clear()
        return

    # Тексты карточки — по типу обращения (сон / ситуация / конфликт).
    card_dream = await DreamRepository(session).get_by_id(dream_id, user_id=user.id)
    cfg = entry_config(card_dream.kind if card_dream is not None else None)

    # Spend cap: над лимитом не начинаем разбор с карточки (до платного классификатора).
    if await deny_entry(message, session, user, settings):
        return

    async with thinking_status(message, statuses=list(cfg.thinking_statuses)):
        res = await classify_confirm_intent(client=client, model=model, user_text=text)
    await record_intent_usage(
        session=session, telegram_user_id=message.from_user.id, session_id=None,
        model=model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
        prompt_version=INTENT_CONFIRM_VERSION,
    )

    if res.outcome == "analyze":
        dream = card_dream
        if dream is None:
            await state.clear()
            await message.answer(QUICK_USE_BUTTONS)
            return
        sess_repo = SessionRepository(session)
        active = await sess_repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id)
        if active is not None or await sess_repo.latest_inactive_quick_for_dream(
            dream_id=dream.id, user_id=user.id
        ):
            # Карточка устарела (появилась активная/прошлая сессия) — не плодим вторую.
            await state.clear()
            await message.answer(cfg.already_active)
            return
        await _start_new_quick_session(
            callback_message=message, state=state, session=session,
            sess_repo=sess_repo, user_id=user.id, dream=dream,
        )
        return

    if res.outcome == "save":
        await state.clear()
        await message.answer(cfg.saved_without_analysis)
        return

    # unclear → переспрашиваем кнопкой (state остаётся в DECIDING_ANALYZE)
    await message.answer(
        cfg.confirm_analyze_prompt,
        reply_markup=dream_actions_keyboard(
            dream_id=dream_id,
            text=DREAM_ACTION_START_ANALYSIS,
            callback_action="start_analysis",
        ),
    )


async def handle_dream_card_text(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """DreamCardStates + свободный текст: analyze/save/unclear (под qa_lock)."""
    if message.from_user is None or message.text is None:
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _dream_card_decide_act(
            text=message.text.strip(), message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            settings=settings,
        )


async def handle_dream_card_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """DreamCardStates + голос: STT → analyze/save/unclear. Под stt_lock (middleware)."""
    if message.from_user is None or message.voice is None:
        return
    # Lazy import — quick.py ←→ onboarding.py циклическая зависимость на уровне модуля.
    from adapters.telegram.handlers.onboarding import _get_text_from_message

    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _dream_card_decide_act(
        text=text, message=message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        settings=settings,
    )


# === ASKING_PER_SESSION_CONTEXT → разбор (шаг выбора ветки убран) ===


async def _run_quick_after_context(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_client: AsyncOpenAI,
    context_client: AsyncOpenAI,
    context_model: str,
    settings: Settings,
    bot: Bot | None = None,
) -> None:
    """Проставить дефолтную ветку и сразу запустить разбор после per-session контекста.

    Шаг выбора ветки (6 кнопок) убран из воронки — ставим нейтральный дефолт
    `RequestBranch.UNCLEAR` («просто проанализируй»).
    """
    if message.from_user is None:
        return
    data = await state.get_data()
    session_id = data.get("session_id")
    if session_id is None:
        return
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None:
        return
    await SessionRepository(session).set_request_branch(
        session_id=session_id, branch=RequestBranch.UNCLEAR
    )
    # Spend cap: над лимитом не запускаем платный разбор.
    if await deny_entry(message, session, user, settings):
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        if settings.image_questions_enabled:
            # Лениво: image_questions ↔ quick — взаимные импорты (см. модуль фичи).
            from adapters.telegram.handlers.image_questions import (
                start_image_questions_flow,
            )

            await start_image_questions_flow(
                message=message, state=state, session=session, user=user,
                session_id=session_id, dream_id=data.get("dream_id"),
                is_onboarding=False, settings=settings, bot=bot,
            )
            return
        await _run_initial_quick(
            message=message, state=state, session=session, user_id=user.id,
            session_id=session_id, openrouter_client=openrouter_client,
            context_client=context_client, context_model=context_model,
            settings=settings, bot=bot,
        )


async def handle_per_session_context_text(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    settings: Settings,
    bot: Bot | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    if message.from_user is None or message.text is None:
        return
    data = await state.get_data()
    session_id: int = data["session_id"]
    context_text = message.text.strip()[:2048]
    await SessionRepository(session).set_per_session_context(
        session_id=session_id, context=context_text
    )
    await _run_quick_after_context(
        message=message, state=state, session=session,
        openrouter_client=openrouter_client, context_client=openrouter_classifier_client,
        context_model=openrouter_context_model, settings=settings, bot=bot,
    )


async def handle_per_session_context_voice(
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
    if message.from_user is None or message.voice is None:
        return
    data = await state.get_data()
    session_id: int = data["session_id"]
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None:
        return
    result = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client, log_prefix="quick.per_session",
        session=session, settings=settings,
    )
    if result is None:
        return
    recorder = UsageRecorder(UsageEventRepository(session))
    await recorder.record_whisper(
        user_id=user.id, audio_seconds=result.duration_seconds, session_id=session_id
    )
    context_text = result.text.strip()[:2048]
    await SessionRepository(session).set_per_session_context(
        session_id=session_id, context=context_text
    )
    await _run_quick_after_context(
        message=message, state=state, session=session,
        openrouter_client=openrouter_client, context_client=openrouter_classifier_client,
        context_model=openrouter_context_model, settings=settings, bot=bot,
    )


async def _deliver_analysis(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot | None,
    session_id: int,
    raw_text: str,
    phase_label: str,
) -> None:
    """Сплит хвоста → лог прозы → отправка прозы с inline-клавой → сохранить
    labels+kind в session_state и message_id клавы в FSM.

    Используется и первичным разбором, и follow-up — единый путь доставки.
    """
    prose, kind, options, qa_map = split_qa_options(raw_text)
    labels = _labels_from_split(kind, options)

    # В БД и историю пишем ТОЛЬКО прозу (без служебного хвоста).
    await log_bot_message(
        session, session_id=session_id, content=prose, phase_label=phase_label
    )

    # labels+kind+kb_html → session_state (merge, чтобы не затереть summary/custom_branch).
    rendered = render_quick_markup(prose)
    kb_html = split_for_telegram(rendered)[-1]
    state_repo = SessionStateRepository(session)
    existing = await state_repo.get_data(session_id=session_id) or {}
    # Сид map.grounded отвеченными образами фичи (NEVER-инвариант §4.4): образ,
    # на вопрос про который юзер уже ответил, не должен получить повторный
    # заземляющий вопрос при выборе его веткой в Q&A. Скипнутые НЕ сидируются.
    if qa_map is not None:
        from adapters.telegram.handlers.image_questions import IMGQ_STATE_KEY

        imgq = existing.get(IMGQ_STATE_KEY) or {}
        answered_names = [
            str(i.get("name")) for i in (imgq.get("items") or [])
            if isinstance(i, dict) and i.get("answer") and i.get("name")
        ]
        if answered_names:
            grounded = [str(g) for g in (qa_map.get("grounded") or []) if g]
            for name in answered_names:
                if name not in grounded:
                    grounded.append(name)
            qa_map = {**qa_map, "grounded": grounded}
    new_state = {
        **existing,
        QA_OPTIONS_KEY: {"labels": labels, "kind": kind, "kb_html": kb_html},
    }
    if qa_map is not None:
        new_state[QA_MAP_KEY] = qa_map
    await state_repo.upsert(session_id=session_id, data=new_state)

    sent = await answer_long(
        message,
        rendered,
        reply_markup=qa_options_keyboard(labels, kind=kind),
        escape_html=False,
    )
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(**{_FSM_KEY_QA_MSG_ID: msg_id})


async def _run_initial_quick(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user_id: int,
    session_id: int,
    openrouter_client: AsyncOpenAI,
    context_client: AsyncOpenAI,
    context_model: str,
    settings: Settings,
    bot: Bot | None = None,
) -> None:
    """Построить контекст → generator → записать usage → IN_QA → ответ юзеру.

    Лок (qa_lock/stt_lock) берёт вызывающий. Используется и кнопочным выбором
    ветки, и свободным кастомным запросом (text/voice).
    """
    ctx = await _build_quick_context(
        session=session, user_id=user_id, session_id=session_id, settings=settings
    )
    if ctx is None:
        log.warning("quick.branch.missing_context", session_id=session_id)
        await message.answer(QUICK_LLM_FAILED)
        return

    kind = await _entry_kind_for_session(session, user_id=user_id, session_id=session_id)
    cfg = entry_config(kind)
    bundle = prompt_bundle(kind)

    async with thinking_status(
        message,
        state=state,
        thinking_state=QuickAnalysisStates.GENERATING_INITIAL,
        statuses=list(cfg.thinking_statuses),
    ):
        try:
            result = await _run_generator(
                ctx=ctx, qa_history=[], user_followup=None,
                openrouter_client=openrouter_client, kind=kind,
            )
        except Exception:  # noqa: BLE001
            log.exception("quick.generator.failed", session_id=session_id)
            await message.answer(QUICK_LLM_FAILED)
            return

    if not result.text:
        if result.error_category == ErrorCategory.CONTENT_FILTER:
            # Зарубили все модели цепочки. Выводим юзера из режима разбора:
            # оставлять его в GENERATING_INITIAL/IN_QA без разбора – тупик.
            await note_content_filter(
                session=session,
                session_id=session_id,
                where="analysis",
                user_id=user_id,
                pause=True,
            )
            await state.clear()
            await message.answer(cfg.cf_analysis, reply_markup=remove_keyboard())
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
    await state.set_state(QuickAnalysisStates.IN_QA)
    log.info(
        "quick.generator.done",
        session_id=session_id,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )
    # usage уже записан выше; FSM в IN_QA
    await _deliver_analysis(
        message=message, state=state, session=session, bot=bot,
        session_id=session_id, raw_text=result.text, phase_label=QUICK_PHASE_INITIAL,
    )


# === Хелперы LLM ===


async def _entry_kind_for_session(
    session: AsyncSession, *, user_id: int, session_id: int
) -> str:
    """dreams.kind сессии; при любой недостаче данных — dream (fail-safe).

    Fail-safe гарантирует: любой легаси-путь (старые сессии, гонки) ведёт себя
    ровно как ветка сна до фичи.
    """
    sess = await SessionRepository(session).get_by_id(session_id, user_id=user_id)
    if sess is None:
        return KIND_DREAM
    dream = await DreamRepository(session).get_by_id(sess.dream_id, user_id=user_id)
    if dream is None or not dream.kind:
        return KIND_DREAM
    return dream.kind


async def _run_generator(
    *,
    ctx: QuickContext,
    qa_history: list[tuple[str, str]],
    user_followup: str | None,
    openrouter_client: AsyncOpenAI,
    model: str = "",  # deprecated, ignored — generator owns chain
    qa_summary: str | None = None,
    qa_map: dict | None = None,
    kind: str = KIND_DREAM,
):
    """Сборка messages по типу обращения + вызов generator. Возвращает GeneratorResult.

    kind выбирает промпт-бандл (PROMPTS_BY_KIND); дефолт dream = прежнее поведение
    байт-в-байт (dream-бандл указывает на функции quick.py).

    Если user_followup is None → первый разбор; иначе — follow-up.
    qa_summary используется только в follow-up (rolling summary, опция C).
    qa_map используется только в follow-up (карта беседы IFS, опция C).
    """
    bundle = prompt_bundle(kind)
    if user_followup is None:
        messages = bundle.build_initial_messages(ctx)
    else:
        messages = bundle.build_followup_messages(
            ctx, qa_history=qa_history, user_followup=user_followup,
            qa_summary=qa_summary, qa_map=qa_map,
        )
    return await generate_quick_response(client=openrouter_client, messages=messages)


async def _build_quick_context(
    *, session: AsyncSession, user_id: int, session_id: int, settings: Settings
) -> QuickContext | None:
    """Собрать QuickContext из БД. Вернёт None, если профиль/сон/сессия не найдены/не заполнены."""
    user = await session.get(User, user_id)
    if user is None or user.profile_life_context is None:
        return None
    sess_repo = SessionRepository(session)
    sess = await sess_repo.get_by_id(session_id, user_id=user_id)
    if sess is None or sess.request_branch is None or sess.per_session_context is None:
        return None
    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(sess.dream_id, user_id=user_id)
    if dream is None:
        return None
    state_repo = SessionStateRepository(session)
    state_data = await state_repo.get_data(session_id=session_id) or {}
    custom_branch: str | None = None
    if sess.request_branch == RequestBranch.CUSTOM:
        raw = state_data.get(CUSTOM_BRANCH_KEY)
        custom_branch = raw if isinstance(raw, str) else None
    img_block = ""
    if settings.image_questions_enabled:
        from adapters.telegram.handlers.image_questions import IMGQ_STATE_KEY

        imgq = state_data.get(IMGQ_STATE_KEY)
        if isinstance(imgq, dict) and isinstance(imgq.get("items"), list):
            img_block = prompt_bundle(dream.kind).build_answers_block(imgq["items"])
    mem_block = ""
    if settings.memory_inject_enabled:
        mem_items = await UserMemoryItemRepository(session).list_active(user_id=user_id)
        mem_block = build_memory_block(mem_items)
    return QuickContext(
        user_display_name=user.display_name or "вы",
        user_life_context=user.profile_life_context,
        per_session_context=sess.per_session_context,
        request_branch=sess.request_branch,
        dream_transcription=dream.transcription,
        request_branch_custom=custom_branch,
        memory_block=mem_block,
        image_answers_block=img_block,
    )


# === IN_QA: уточняющие вопросы ===

# Сигнал «завершить» по природе КОРОТКИЙ («хватит», «спасибо, достаточно», «на сегодня
# всё»). Длинная реплика — это содержательный ответ/рассказ, а не закрытие. Детерминированный
# гард поверх classify_qa_intent: finish на длинной реплике — почти всегда ложное
# срабатывание (классификатор читает рефлексивно-итоговый тон как «подведение черты»).
_QA_FINISH_MAX_CHARS = 240


def _is_real_finish(outcome: str, user_text: str) -> bool:
    """True только если outcome=finish И реплика достаточно короткая для сигнала закрытия."""
    return outcome == "finish" and len(user_text.strip()) <= _QA_FINISH_MAX_CHARS


async def _handle_qa_turn(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openrouter_client: AsyncOpenAI,
    openrouter_generator_model: str,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_classifier_model: str,
    openrouter_intent_model: str,
    settings: Settings,
    user_followup: str,
    user_content_type: SessionMessageContentType = SessionMessageContentType.TEXT,
    audio_file_id: str | None = None,
    session_id_override: int | None = None,
    telegram_user_id: int | None = None,
    classify_intent: bool = True,
) -> None:
    """Общая логика QA-turn: запись user-input в БД, чтение истории из БД,
    rolling summary при 8+ turns, вызов LLM, запись ответа в БД.

    qa_history больше НЕ хранится в FSM — source of truth теперь session_messages.
    Опция C: при достижении 8+ пар активируется rolling summary через Flash.

    `session_id_override` — для случаев, когда вызываем не из стандартного IN_QA
    (например, из онбординга, где session_id хранится под другим ключом).
    `telegram_user_id` — явный tg-id юзера; нужен при вызове из callback, где
    `message.from_user` — бот (Task 7). Если задан — перекрывает `message.from_user.id`.
    """
    tg_user_id = telegram_user_id if telegram_user_id is not None else (
        message.from_user.id if message.from_user else None
    )
    if tg_user_id is None:
        return
    data = await state.get_data()
    if session_id_override is not None:
        session_id = session_id_override
    else:
        session_id = data["session_id"]

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(tg_user_id)
    if user is None:
        return

    # Тип обращения этой сессии: промпты, статусы и CF-тексты берём по нему.
    # Дефолт dream = поведение до фичи байт-в-байт.
    kind = await _entry_kind_for_session(session, user_id=user.id, session_id=session_id)
    cfg = entry_config(kind)
    bundle = prompt_bundle(kind)

    # Погасить прошлую QA-клаву — устаревшие чипы нельзя нажать повторно.
    await extinguish_inline_kb(
        state=state, bot=bot,
        chat_id=message.chat.id if message.chat else None,
        fsm_key=_FSM_KEY_QA_MSG_ID,
    )

    async with qa_user_lock(tg_user_id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return

        # Точка 4: свободная реплика может быть НЕ вопросом, а сигналом «завершить».
        # Классифицируем дешёвой моделью ПОД локом (спам во время вызова → busy-guard).
        # continue = вопрос/комментарий/реакция; finish = явное закрытие. Дефолт continue.
        #
        # classify_intent=False (chip/skip tap): чип — детерминированный «explore»,
        # классификатор здесь не нужен. Пропускаем его, чтобы исключить ложные
        # срабатывания finish/new_dream на лейблах вроде «Завершить разбор» или
        # «Исчезновение маяка», а также бот-id mislookup при вызове из callback.
        if classify_intent:
            # Даём классификатору ВОПРОС бота, на который отвечает юзер (последняя реплика
            # бота): развёрнутый ответ по существу = continue, а не «завершить». Вопрос —
            # это проза с вовлекающим вопросом в конце, поэтому берём её хвост.
            last_bot = await SessionMessageRepository(session).last_bot_message(
                session_id=session_id
            )
            bot_question = last_bot[-500:] if last_bot else None
            async with thinking_status(message, statuses=list(cfg.thinking_statuses)):
                intent = await classify_qa_intent(
                    client=openrouter_classifier_client,
                    model=openrouter_intent_model,
                    user_text=user_followup,
                    bot_question=bot_question,
                )
            await record_intent_usage(
                session=session, telegram_user_id=tg_user_id, session_id=session_id,
                model=openrouter_intent_model, input_tokens=intent.input_tokens,
                output_tokens=intent.output_tokens, prompt_version=INTENT_QA_VERSION,
            )
            if _is_real_finish(intent.outcome, user_followup):
                # qa_lock уже держим — зовём тело напрямую, иначе гард от двойного
                # тапа проглотит завершение по свободной реплике.
                await _finish_quick_impl(
                    message=message, state=state, session=session, bot=bot,
                    tg_user_id=tg_user_id, settings=settings,
                    openrouter_classifier_client=openrouter_classifier_client,
                )
                return
            if intent.outcome == "finish":
                # Длинная реплика ≠ сигнал закрытия — не завершаем, продолжаем разбор.
                log.info(
                    "quick.qa.finish_ignored_long_answer",
                    session_id=session_id,
                    chars=len(user_followup.strip()),
                )
            if intent.outcome == "new_dream":
                # Юзер свободной формой попросил разобрать новый сон → закрываем текущий
                # разбор и запускаем сбор нового сна (как команда /newdream).
                # Если это НЕзавершённый онбординг — сначала завершаем онбординг
                # (mark_onboarded + меню, как при «Завершить»), иначе гейт /newdream
                # отбил бы юзера сообщением NOT_ONBOARDED_PROMPT в тупик (он ведь
                # как раз внутри онбординга).
                sess_repo = SessionRepository(session)
                sess = await sess_repo.get_by_id(session_id, user_id=user.id)
                if (
                    sess is not None
                    and sess.is_onboarding_session
                    and user.onboarding_completed_at is None
                ):
                    await user_repo.mark_onboarded(user_id=user.id)
                    await session.flush()
                    await sync_user_commands(
                        bot=bot, session=session, user=user, onboarded=True,
                        settings=settings,
                    )
                    log.info("quick.onboarding.completed_via_new_dream", session_id=session_id)

                from adapters.telegram.handlers.newdream import handle_newdream_command

                await handle_newdream_command(
                    message, state, session, settings, bot,
                    openrouter_classifier_client=openrouter_classifier_client,
                )
                return

        ctx = await _build_quick_context(
            session=session, user_id=user.id, session_id=session_id, settings=settings
        )
        if ctx is None:
            await message.answer(QUICK_LLM_FAILED)
            return

        # Подтягиваем накопленную историю (initial + предыдущие Q&A пары) ИЗ БД.
        # ВАЖНО: пишем текущий user-message в БД ТОЛЬКО после успешного LLM-ответа
        # (см. ниже). Иначе при сбое LLM в БД оседает «висячий» вопрос без ответа,
        # который load_quick_qa_history молча игнорирует — и юзер не понимает,
        # почему его вопрос не учитывается в дальнейшем контексте.
        initial_text, prior_pairs = await load_quick_qa_history(session, session_id=session_id)
        full_qa_history: list[tuple[str, str]] = []
        if initial_text is not None:
            full_qa_history.append(("[initial]", initial_text))
        full_qa_history.extend(prior_pairs)

        # Rolling summary (опция C): если ≥ SUMMARY_THRESHOLD пар, сжимаем старые
        # через Flash и держим только последние RECENT_KEEP пар + summary в system.
        qa_summary: str | None = None
        qa_history_for_llm: list[tuple[str, str]] = full_qa_history

        state_repo = SessionStateRepository(session)
        existing_state = await state_repo.get_data(session_id=session_id) or {}
        previous_summary = existing_state.get(SUMMARY_KEY)
        qa_map_saved = existing_state.get(QA_MAP_KEY)
        qa_map: dict | None = qa_map_saved if isinstance(qa_map_saved, dict) else None

        if should_summarize(len(full_qa_history)):
            to_compress, to_keep = split_history_for_summary(full_qa_history)
            try:
                summary_text, sum_in, sum_out = await compute_summary(
                    client=openrouter_classifier_client,
                    model=openrouter_classifier_model,
                    qa_pairs_to_compress=to_compress,
                    previous_summary=previous_summary if isinstance(previous_summary, str) else None,
                )
            except Exception:  # noqa: BLE001
                log.exception("quick.qa.summary_failed", session_id=session_id)
                # Fallback: используем старый summary если есть, и полную историю
                qa_summary = previous_summary if isinstance(previous_summary, str) else None
            else:
                qa_summary = summary_text
                qa_history_for_llm = to_keep
                new_state = {**existing_state, SUMMARY_KEY: summary_text}
                await state_repo.upsert(session_id=session_id, data=new_state)
                recorder_sum = UsageRecorder(UsageEventRepository(session))
                await recorder_sum.record_openrouter_classifier(
                    user_id=user.id,
                    session_id=session_id,
                    model=openrouter_classifier_model,
                    input_tokens=sum_in,
                    output_tokens=sum_out,
                    prompt_version=SUMMARY_PROMPT_VERSION,
                )

        async with thinking_status(
            message,
            state=state,
            thinking_state=QuickAnalysisStates.GENERATING_FOLLOWUP,
            statuses=list(cfg.thinking_statuses),
        ):
            try:
                result = await _run_generator(
                    ctx=ctx,
                    qa_history=qa_history_for_llm,
                    user_followup=user_followup,
                    openrouter_client=openrouter_client,
                    qa_summary=qa_summary,
                    qa_map=qa_map,
                    kind=kind,
                )
            except Exception:  # noqa: BLE001
                log.exception("quick.qa.generator_failed", session_id=session_id)
                await message.answer(QUICK_LLM_FAILED)
                return

        if not result.text:
            if result.error_category == ErrorCategory.CONTENT_FILTER:
                # Разбор уже выдан – из сессии НЕ выводим, состояние не трогаем.
                # thinking_status на выходе сам вернул IN_QA; юзер может ответить
                # иначе, инлайн-кнопки под прошлым вопросом остаются рабочими.
                await note_content_filter(
                    session=session,
                    session_id=session_id,
                    where="qa",
                    user_id=user.id,
                    pause=False,
                )
                await message.answer(cfg.cf_qa)
                return
            await message.answer(llm_error_message(result.error_category))
            return

        # Q&A turn — followup-версия бандла (отличается от initial для аналитики)
        recorder = UsageRecorder(UsageEventRepository(session))
        await recorder.record_openrouter_generator(
            user_id=user.id,
            session_id=session_id,
            model=result.model_used or _QUICK_FALLBACK_CHAIN[0],
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            prompt_version=bundle.version_followup,
        )

        # ONB_QA_ANSWERED: первый QA-ответ юзера в онбординге (guard дешёвый —
        # extra query только раз в жизни; idempotency emit_funnel_once — страховка).
        if user.qa_answers_total == 0:
            _sess_check = await SessionRepository(session).get_by_id(session_id, user_id=user.id)
            if _sess_check is not None and _sess_check.is_onboarding_session:
                await emit_funnel_step(
                    session,
                    user_id=user.id,
                    telegram_user_id=user.telegram_user_id,
                    event_type=AnalyticsEventType.ONB_QA_ANSWERED,
                    metadata={"session_id": session_id, "entry_type": kind},
                )

        # Пишем пару (user-msg + bot-ответ) ВМЕСТЕ — только после успешного LLM,
        # чтобы пара была атомарной и load_quick_qa_history никогда не видела
        # user без ответа. _deliver_analysis сам пишет прозу бота (без хвоста).
        await log_user_message(
            session,
            session_id=session_id,
            content=user_followup,
            content_type=user_content_type,
            audio_file_id=audio_file_id,
        )
        await _deliver_analysis(
            message=message, state=state, session=session, bot=bot,
            session_id=session_id, raw_text=result.text, phase_label=QUICK_PHASE_QA,
        )


async def _qa_paywall_if_over(
    *,
    target: Message | CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    settings: Settings,
    session_id: int | None,
) -> bool:
    """In-flow paywall для IN_QA: при превышении лимита сохраняет сон и выходит.

    Если активная сессия — онбординговая (и юзер ещё не onboarded), дополнительно
    помечает онбординг завершённым, чтобы не зациклить юзера обратно в онбординг.
    """
    if not await is_over_limit(session, user, settings):
        return False
    sess = (
        await SessionRepository(session).get_by_id(session_id, user_id=user.id)
        if session_id is not None
        else None
    )
    mark = bool(sess and sess.is_onboarding_session and user.onboarding_completed_at is None)
    await exit_with_paywall(
        target, state, session, user, settings, session_id=session_id, mark_onboarded=mark
    )
    return True


async def handle_quick_qa_text(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openrouter_client: AsyncOpenAI,
    openrouter_generator_model: str,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_classifier_model: str,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    if message.text is None or message.from_user is None:
        return
    # Spend cap: лимит перейдён в Q&A → сохраняем сон и выходим (а не отвечаем).
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    session_id = (await state.get_data()).get("session_id")
    if user is not None and await _qa_paywall_if_over(
        target=message, state=state, session=session, user=user,
        settings=settings, session_id=session_id,
    ):
        return
    await _handle_qa_turn(
        message=message,
        state=state,
        session=session,
        bot=bot,
        openrouter_client=openrouter_client,
        openrouter_generator_model=openrouter_generator_model,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_classifier_model=openrouter_classifier_model,
        openrouter_intent_model=openrouter_intent_model,
        settings=settings,
        user_followup=message.text.strip(),
    )


async def handle_quick_qa_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_client: AsyncOpenAI,
    openrouter_generator_model: str,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_classifier_model: str,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    if message.voice is None or message.from_user is None:
        return
    data = await state.get_data()
    session_id: int = data["session_id"]

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(message.from_user.id)
    if user is None:
        return

    # Spend cap: лимит перейдён → сохраняем сон и выходим ДО транскрипции (Whisper платный).
    if await _qa_paywall_if_over(
        target=message, state=state, session=session, user=user,
        settings=settings, session_id=session_id,
    ):
        return

    stt = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client, log_prefix="quick.qa",
        session=session, settings=settings,
    )
    if stt is None:
        return
    recorder = UsageRecorder(UsageEventRepository(session))
    await recorder.record_whisper(
        user_id=user.id, audio_seconds=stt.duration_seconds, session_id=session_id
    )

    await _handle_qa_turn(
        message=message,
        state=state,
        session=session,
        bot=bot,
        openrouter_client=openrouter_client,
        openrouter_generator_model=openrouter_generator_model,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_classifier_model=openrouter_classifier_model,
        openrouter_intent_model=openrouter_intent_model,
        settings=settings,
        user_followup=stt.text.strip(),
        user_content_type=SessionMessageContentType.VOICE,
        audio_file_id=message.voice.file_id,
    )


# === Closing ===


async def _pause_session(
    *,
    session: AsyncSession,
    session_id: int,
    user_id: int | None = None,
    dream_id: int | None = None,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    """Поставить quick-сессию на паузу: status=paused. dream.status остаётся IN_ANALYSIS.

    Если прокинуты `settings` + `openrouter_classifier_client` (+ `user_id`) и
    `settings.memory_write_enabled` — после паузы запускается экстрактор памяти
    (shadow-режим, fire-and-forget). Без них — graceful skip (легаси-вызовы/тесты).
    """
    sess_repo = SessionRepository(session)
    await sess_repo.pause(session_id=session_id)
    if (
        openrouter_classifier_client is not None
        and settings is not None
        and settings.memory_write_enabled
        and user_id is not None
    ):
        await run_memory_extraction_on_pause(
            session=session,
            user_id=user_id,
            session_id=session_id,
            dream_id=dream_id,
            client=openrouter_classifier_client,
            model=settings.openrouter_context_model,
            compact_enabled=settings.memory_compact_enabled,
            compact_budget_chars=settings.memory_compact_budget_chars,
        )


async def handle_finish_quick_reply(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    telegram_user_id: int | None = None,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    """Reply «Завершить» в IN_QA → pause + clear FSM + ReplyKeyboardRemove.

    Если сессия — онбординг-сессия и onboarding_completed_at ещё не выставлен,
    ставит его сейчас и активирует команды меню (☰) через update_user_commands.

    `telegram_user_id` — явный tg-id юзера; нужен при вызове из callback, где
    `message.from_user` — бот (Task 7). Если задан — перекрывает `message.from_user.id`.
    `settings` + `openrouter_classifier_client` — для экстрактора памяти на паузе
    (aiogram инжектит их по имени при вызове как хэндлер; при прямых вызовах
    прокидывают вызывающие; None = skip экстракции).
    """
    tg_user_id = telegram_user_id if telegram_user_id is not None else (
        message.from_user.id if message.from_user else None
    )
    if tg_user_id is None:
        return
    # Двойной тап inline-кнопки «Завершить» приходит как CallbackQuery и проходит
    # мимо BusyGuardMiddleware (он фильтрует только Message). Без lock'а оба вызова
    # видят сессию IN_PROGRESS (первый ещё не закоммичен) и оба платно гоняют
    # экстракцию памяти. Второму тапу отвечать нечем — первый всё сделал.
    async with qa_user_lock(tg_user_id) as acquired:
        if not acquired:
            log.info("quick.finish.double_tap_dropped", telegram_user_id=tg_user_id)
            return
        await _finish_quick_impl(
            message=message, state=state, session=session, bot=bot,
            tg_user_id=tg_user_id, settings=settings,
            openrouter_classifier_client=openrouter_classifier_client,
        )


async def _finish_quick_impl(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    tg_user_id: int,
    settings: Settings | None,
    openrouter_classifier_client: AsyncOpenAI | None,
) -> None:
    """Тело завершения без взятия qa_lock.

    Отдельно от хэндлера, потому что путь «свободная реплика → классификатор finish»
    (_handle_qa_turn) уже держит lock: повторный вход в него заглушил бы завершение.
    """
    data = await state.get_data()
    session_id = data.get("session_id")
    if session_id is None:
        await state.clear()
        return

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(tg_user_id)
    if user is None:
        return

    sess_repo = SessionRepository(session)
    sess = await sess_repo.get_by_id(session_id, user_id=user.id)
    if sess is None or sess.status != SessionStatus.IN_PROGRESS:
        await state.clear()
        return

    await _pause_session(
        session=session,
        session_id=session_id,
        user_id=user.id,
        dream_id=sess.dream_id,
        settings=settings,
        openrouter_classifier_client=openrouter_classifier_client,
    )

    # Onboarding completes only on user's first «Завершить» tap
    notice_text = SESSION_PAUSED_NOTICE
    if sess.is_onboarding_session and user.onboarding_completed_at is None:
        entry_kind = await _entry_kind_for_session(
            session, user_id=user.id, session_id=session_id
        )
        await emit_funnel_step(
            session,
            user_id=user.id,
            telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.ONB_COMPLETED,
            metadata={"session_id": session_id, "entry_type": entry_kind},
        )
        await user_repo.mark_onboarded(user_id=user.id)
        await session.flush()
        await sync_user_commands(
            bot=bot, session=session, user=user, onboarded=True, settings=settings,
        )
        notice_text = END_OF_ONBOARDING_TEXT

    await state.clear()
    await message.answer(notice_text, reply_markup=ReplyKeyboardRemove())
    log.info(
        "quick.session.paused_by_user",
        session_id=session_id,
        was_onboarding=sess.is_onboarding_session,
    )



async def _resume_qa_for_dream(
    *,
    callback_message: Message,
    state: FSMContext,
    session: AsyncSession,
    user_id: int,
    dream_id: int,
    auto_pause_other: bool,
    intro_text: str | None,
    include_last_reply: bool,
    source: str,
    settings: Settings,
    openrouter_client: AsyncOpenAI | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """Общая логика возобновления Q&A на сне.

    Args:
        auto_pause_other: если True — паузит активную IN_PROGRESS другого сна
            и шлёт уведомление по типу того обращения. Для dream_card_open False (auto-pause
            сделан раньше, на уровне handle_dream_by_id).
        intro_text: что пишем юзеру при входе. Для continue_quick это
            QUICK_CONTINUE_RESUMED_INTRO. Для dream_card_open — None (без приписки;
            клава «Завершить» вешается на карточку в _show_dream_by_id).
        include_last_reply: если True — после intro дублируем последнюю реплику
            бота из истории. Для continue_quick True (юзер давно не видел разбор),
            для dream_card_open False (handle_dream_by_id уже отправил его
            в карточке).
    """
    sess_repo = SessionRepository(session)
    if auto_pause_other:
        other_active = await sess_repo.latest_in_progress_for_user(user_id=user_id)
        if other_active is not None and other_active.dream_id != dream_id:
            await sess_repo.pause(session_id=other_active.id)
            paused_kind = await entry_kind_for_dream(
                session, dream_id=other_active.dream_id, user_id=user_id
            )
            await callback_message.answer(entry_config(paused_kind).auto_paused_notice)
            log.info(
                "quick.resume.auto_paused_other",
                paused_session_id=other_active.id,
                resumed_dream_id=dream_id,
            )

    last = await sess_repo.latest_inactive_quick_for_dream(dream_id=dream_id, user_id=user_id)
    if last is None:
        log.warning("quick.resume.no_inactive", dream_id=dream_id, user_id=user_id)
        return

    await sess_repo.resume_session(session_id=last.id)

    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user_id)
    if dream is not None:
        dream.status = DreamStatus.IN_ANALYSIS

    # Фича image questions: сессия могла быть запаузена ДО initial-разбора
    # (юзер ушёл посреди вопросов). Тогда ре-энтри — в фичу, не в IN_QA.
    if settings.image_questions_enabled:
        initial_probe, _ = await load_quick_qa_history(session, session_id=last.id)
        if initial_probe is None:
            from adapters.telegram.handlers.image_questions import (
                IMGQ_STATE_KEY,
                resume_image_questions,
            )

            imgq_data = (await SessionStateRepository(session).get_data(
                session_id=last.id) or {}).get(IMGQ_STATE_KEY)
            user_obj = await session.get(User, user_id)
            if isinstance(imgq_data, dict) and user_obj is not None:
                handled = await resume_image_questions(
                    callback_message=callback_message, state=state, session=session,
                    user=user_obj, session_id=last.id, dream_id=dream_id,
                    imgq=imgq_data, settings=settings, bot=None,
                    openrouter_client=openrouter_client,
                    context_client=openrouter_classifier_client,
                    context_model=openrouter_context_model,
                )
                if handled:
                    return

    await state.set_state(QuickAnalysisStates.IN_QA)
    await state.set_data({"dream_id": dream_id, "session_id": last.id})

    # intro_text=None (dream_card_open) — не шлём приписку; клава «Завершить» уже
    # на карточке. Для continue_quick — короткое intro без reply-клавы
    # («Завершить» теперь живёт inline под последним разбором).
    if intro_text is not None:
        await callback_message.answer(intro_text)

    prior_turns: int | None = None
    if include_last_reply:
        initial_text, qa_pairs = await load_quick_qa_history(session, session_id=last.id)
        prior_turns = len(qa_pairs)
        last_bot_reply = qa_pairs[-1][1] if qa_pairs else initial_text
        if last_bot_reply:
            state_repo = SessionStateRepository(session)
            saved = (await state_repo.get_data(session_id=last.id) or {}).get(QA_OPTIONS_KEY) or {}
            raw_labels = saved.get("labels")
            labels = raw_labels if isinstance(raw_labels, list) else []
            rendered = render_quick_markup(last_bot_reply)
            kb_html = split_for_telegram(rendered)[-1]
            # Update session_state with kb_html for the keyboard-bearing message.
            existing_full = await state_repo.get_data(session_id=last.id) or {}
            existing_qa = existing_full.get(QA_OPTIONS_KEY) or {}
            await state_repo.upsert(
                session_id=last.id,
                data={**existing_full, QA_OPTIONS_KEY: {**existing_qa, "kb_html": kb_html}},
            )
            sent = await answer_long(
                callback_message, rendered,
                reply_markup=qa_options_keyboard(labels), escape_html=False,
            )
            msg_id = getattr(sent, "message_id", None)
            if isinstance(msg_id, int):
                await state.update_data(**{_FSM_KEY_QA_MSG_ID: msg_id})

    log.info(
        "quick.resumed",
        session_id=last.id,
        dream_id=dream_id,
        user_id=user_id,
        source=source,
        prior_turns=prior_turns,
    )


async def handle_continue_quick_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_client: AsyncOpenAI | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """callback_data='continue_quick:<dream_id>'. Поднять последнюю Quick-сессию
    в IN_PROGRESS, перевести FSM в IN_QA, показать последнюю реплику бота."""
    if callback.data is None or callback.message is None or callback.from_user is None:
        return
    await safe_answer(callback)

    try:
        dream_id = int(callback.data.split(":", 1)[1])
    except (IndexError, ValueError):
        return

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(callback.from_user.id)
    if user is None:
        return

    # Spend cap: над лимитом не возобновляем платный Q&A (пейвол, FSM чистим).
    session_id = (await state.get_data()).get("session_id")
    if await exit_with_paywall(
        callback, state, session, user, settings, session_id=session_id
    ):
        return

    await _resume_qa_for_dream(
        callback_message=callback.message,
        state=state,
        session=session,
        user_id=user.id,
        dream_id=dream_id,
        auto_pause_other=True,
        intro_text=QUICK_CONTINUE_RESUMED_INTRO,
        include_last_reply=True,
        source="continue_quick",
        settings=settings,
        openrouter_client=openrouter_client,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_context_model=openrouter_context_model,
    )


async def _mark_qa_choice(
    *,
    state: FSMContext,
    bot: Bot | None,
    session: AsyncSession,
    session_id: int | None,
    chat_id: int | None,
    choice: str,
) -> None:
    """Edit the QA keyboard message to show the chosen option and remove keyboard."""
    data = await state.get_data()
    msg_id = data.get(_FSM_KEY_QA_MSG_ID)
    kb_html: str | None = None
    if session_id is not None:
        saved = (await SessionStateRepository(session).get_data(session_id=session_id) or {}).get(QA_OPTIONS_KEY) or {}
        raw = saved.get("kb_html")
        kb_html = raw if isinstance(raw, str) else None
    if isinstance(msg_id, int) and bot is not None and chat_id is not None and kb_html:
        new_text = f"{kb_html}\n\n👉 <b>Ваш выбор:</b> {html_escape(choice)}"
        try:
            await bot.edit_message_text(chat_id=chat_id, message_id=msg_id, text=new_text, parse_mode="HTML")
            await state.update_data(**{_FSM_KEY_QA_MSG_ID: None})
            return
        except Exception:  # noqa: BLE001
            log.debug("qa.mark_choice.edit_failed", exc_info=True)
    # Fallback: just remove the keyboard
    await extinguish_inline_kb(state=state, bot=bot, chat_id=chat_id, fsm_key=_FSM_KEY_QA_MSG_ID)


async def handle_qa_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openrouter_client: AsyncOpenAI,
    openrouter_generator_model: str,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_classifier_model: str,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """Тап inline-кнопки Q&A: qa:N (чип) / qa:custom / qa:finish.

    qa:N → followup = лейбл из session_state (дословно) → _handle_qa_turn.
    qa:custom → подсказка «Слушаю ваш вопрос», остаёмся в IN_QA.
    qa:finish → как reply «Завершить» (pause+сохранить).
    Перед действием редактируем сообщение — показываем «Ваш выбор: …» (или гасим клаву).
    """
    if callback.from_user is None or callback.message is None or callback.data is None:
        return
    await safe_answer(callback)
    chat_id = callback.message.chat.id if callback.message.chat else None

    # Получаем session_id заранее — нужен для _mark_qa_choice во всех ветках.
    fsm_data = await state.get_data()
    session_id: int | None = fsm_data.get("session_id")

    data = callback.data
    if data == QA_CALLBACK_FINISH:
        await _mark_qa_choice(
            state=state, bot=bot, session=session,
            session_id=session_id, chat_id=chat_id,
            choice="сохранить и завершить",
        )
        await handle_finish_quick_reply(
            callback.message, state, session, bot,
            telegram_user_id=callback.from_user.id,
            settings=settings,
            openrouter_classifier_client=openrouter_classifier_client,
        )
        return
    if data == QA_CALLBACK_CUSTOM:
        await _mark_qa_choice(
            state=state, bot=bot, session=session,
            session_id=session_id, chat_id=chat_id,
            choice="свой вопрос",
        )
        await callback.message.answer(QA_CUSTOM_QUESTION_PROMPT)
        return

    if data == QA_CALLBACK_SKIP:
        await _mark_qa_choice(
            state=state, bot=bot, session=session,
            session_id=session_id, chat_id=chat_id, choice="пропустить",
        )
        if session_id is None:
            return
        user = await UserRepository(session).get_by_tg_id(callback.from_user.id)
        if user is not None and await _qa_paywall_if_over(
            target=callback, state=state, session=session, user=user,
            settings=settings, session_id=session_id,
        ):
            return
        await _handle_qa_turn(
            message=callback.message, state=state, session=session, bot=bot,
            openrouter_client=openrouter_client,
            openrouter_generator_model=openrouter_generator_model,
            openrouter_classifier_client=openrouter_classifier_client,
            openrouter_classifier_model=openrouter_classifier_model,
            openrouter_intent_model=openrouter_intent_model,
            settings=settings, user_followup=QA_SKIP_FOLLOWUP,
            telegram_user_id=callback.from_user.id,
            classify_intent=False,
        )
        return

    # qa:N — достаём лейбл из session_state
    try:
        idx = int(data.split(":", 1)[1])
    except (IndexError, ValueError):
        return
    if session_id is None:
        return
    state_repo = SessionStateRepository(session)
    saved = (await state_repo.get_data(session_id=session_id) or {}).get(QA_OPTIONS_KEY) or {}
    labels = saved.get("labels") or []
    if not isinstance(labels, list) or idx < 0 or idx >= len(labels):
        # Устаревший тап — клава уже погашена; просто гасим без редактирования.
        await extinguish_inline_kb(state=state, bot=bot, chat_id=chat_id, fsm_key=_FSM_KEY_QA_MSG_ID)
        return
    followup = str(labels[idx])

    await _mark_qa_choice(
        state=state, bot=bot, session=session,
        session_id=session_id, chat_id=chat_id,
        choice=followup,
    )

    # Пейвол: тап = платный follow-up.
    user = await UserRepository(session).get_by_tg_id(callback.from_user.id)
    if user is not None and await _qa_paywall_if_over(
        target=callback, state=state, session=session, user=user,
        settings=settings, session_id=session_id,
    ):
        return

    await _handle_qa_turn(
        message=callback.message, state=state, session=session, bot=bot,
        openrouter_client=openrouter_client,
        openrouter_generator_model=openrouter_generator_model,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_classifier_model=openrouter_classifier_model,
        openrouter_intent_model=openrouter_intent_model,
        settings=settings, user_followup=followup,
        telegram_user_id=callback.from_user.id,
        classify_intent=False,
    )


# === Router ===


def create_router() -> Router:
    router = Router(name="quick_analysis")

    router.callback_query.register(
        handle_start_analysis_callback, F.data.startswith("start_analysis:")
    )

    router.message.register(
        handle_per_session_context_text,
        QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_per_session_context_voice,
        QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT,
        F.voice,
    )

    # IN_QA: inline-чипы qa:N / qa:custom / qa:finish — BEFORE general text handler
    router.callback_query.register(
        handle_qa_callback,
        QuickAnalysisStates.IN_QA,
        F.data.startswith("qa:"),
    )
    # IN_QA: reply «Завершить» BEFORE general text handler
    router.message.register(
        handle_finish_quick_reply,
        QuickAnalysisStates.IN_QA,
        F.text == FINISH_BUTTON_TEXT,
    )
    # IN_QA: text/voice/end_session
    router.message.register(
        handle_quick_qa_text,
        QuickAnalysisStates.IN_QA,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_quick_qa_voice,
        QuickAnalysisStates.IN_QA,
        F.voice,
    )
    router.callback_query.register(
        handle_continue_quick_callback, F.data.startswith("continue_quick:")
    )

    # Карточка сна без разбора: свободный текст/голос вместо тапа «Разобрать сейчас».
    router.message.register(
        handle_dream_card_text,
        DreamCardStates.DECIDING_ANALYZE,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_dream_card_voice,
        DreamCardStates.DECIDING_ANALYZE,
        F.voice,
    )

    return router
