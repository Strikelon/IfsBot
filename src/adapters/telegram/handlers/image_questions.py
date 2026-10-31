"""Опциональные вопросы по образам сна перед initial-разбором (offer-first).

Флоу (спека 2026-07-05): приглашение да/нет ДО выделения образов («Не хочу» =
ноль LLM-затрат) → после «Я отвечу» один вызов image_questions.v1 по цепочке
генератора → 1-3 открытых заземляющих вопроса → ответы в session_messages
(phase image_q, память подхватит) и в session_state.data.image_questions
(инжект в initial + сид map.grounded).

Единый оркестратор для обычного разбора и онбординга (is_onboarding решает
только funnel-события и диспетчеризацию старта разбора). За kill-switch
settings.image_questions_enabled: выключен → модуль не вызывается вовсе.

Циклические импорты: quick.py/onboarding.py вызывают start_image_questions_flow
лениво; этот модуль лениво импортирует _run_initial_quick/_build_quick_context/
_run_quick_for_onboarding (паттерн существующей пары quick↔onboarding).
"""

from __future__ import annotations

from html import escape as html_escape

import structlog
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import record_intent_usage
from adapters.telegram.handlers._loop_prompt import extinguish_inline_kb
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.keyboards import (
    IMGQ_CALLBACK_NO,
    IMGQ_CALLBACK_SKIP,
    IMGQ_CALLBACK_SKIP_ALL,
    IMGQ_CALLBACK_YES,
    image_offer_keyboard,
    image_question_keyboard,
)
from adapters.telegram.quick_markup import normalize_punctuation, render_quick_markup
from adapters.telegram.session_log import (
    log_bot_message,
    log_user_message,
)
from adapters.telegram.spend_guard import deny_entry, is_over_limit, show_paywall
from adapters.telegram.states import QuickAnalysisStates
from adapters.telegram.texts import (
    IMG_Q_ANSWER_BTN,
    QA_SKIP_BTN,
    THINKING_GUARD_NOTICE,
)
from adapters.telegram.thinking_status import thinking_status
from adapters.telegram.voice_helper import transcribe_with_placeholder
from app.config import Settings
from data.analytics_events import (
    AnalyticsEventType,
    emit_append_event,
    emit_funnel_step,
)
from data.models import SessionMessageContentType, User
from data.repositories import (
    SessionRepository,
    SessionStateRepository,
    UsageEventRepository,
    UserRepository,
)
from domain.entry_types import KIND_DREAM, EntryTypeConfig, entry_config
from llm.generator import _QUICK_FALLBACK_CHAIN, generate_quick_response
from llm.intent import (
    INTENT_ENTRY_ANSWER_VERSION,
    INTENT_ENTRY_OFFER_VERSION,
    INTENT_IMAGE_ANSWER_VERSION,
    INTENT_IMAGE_OFFER_VERSION,
    classify_entry_answer_intent,
    classify_entry_offer_intent,
    classify_image_answer_intent,
    classify_image_offer_intent,
)
from llm.prompts.entry_prompts import EntryPromptBundle, prompt_bundle
from llm.usage import UsageRecorder

log = structlog.get_logger(__name__)

# Ключ прогресса фичи в session_state.data (рядом с qa_options/qa_map/summary).
IMGQ_STATE_KEY = "image_questions"
# FSM: message_id и html последней inline-клавы фичи (гашение + «Ваш выбор: …»).
_FSM_KEY_IMGQ_MSG_ID = "imgq_msg_id"
_FSM_KEY_IMGQ_MSG_HTML = "imgq_msg_html"


async def _load_imgq(session: AsyncSession, session_id: int) -> dict:
    data = await SessionStateRepository(session).get_data(session_id=session_id) or {}
    imgq = data.get(IMGQ_STATE_KEY)
    return imgq if isinstance(imgq, dict) else {}


async def _save_imgq(session: AsyncSession, session_id: int, imgq: dict) -> None:
    """Merge-запись: не затираем summary/qa_options/qa_map (паттерн _deliver_analysis)."""
    repo = SessionStateRepository(session)
    existing = await repo.get_data(session_id=session_id) or {}
    await repo.upsert(session_id=session_id, data={**existing, IMGQ_STATE_KEY: imgq})


async def _remember_imgq_message(state: FSMContext, sent, html: str) -> None:
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(
            **{_FSM_KEY_IMGQ_MSG_ID: msg_id, _FSM_KEY_IMGQ_MSG_HTML: html}
        )


async def _mark_imgq_choice(
    *, state: FSMContext, bot: Bot | None, chat_id: int | None, choice: str
) -> None:
    """Дописать «Ваш выбор: …» в сообщение фичи и снять клаву; fallback — гашение.

    Паттерн _mark_qa_choice (quick.py:1298), html берём из FSM (мы сами его слали).
    """
    data = await state.get_data()
    msg_id = data.get(_FSM_KEY_IMGQ_MSG_ID)
    base_html = data.get(_FSM_KEY_IMGQ_MSG_HTML)
    if (
        isinstance(msg_id, int) and bot is not None and chat_id is not None
        and isinstance(base_html, str) and base_html
    ):
        try:
            await bot.edit_message_text(
                chat_id=chat_id, message_id=msg_id,
                text=f"{base_html}\n\n👉 <b>Ваш выбор:</b> {html_escape(choice)}",
                parse_mode="HTML",
            )
            await state.update_data(**{_FSM_KEY_IMGQ_MSG_ID: None})
            return
        except Exception:  # noqa: BLE001
            log.debug("imgq.mark_choice.edit_failed", exc_info=True)
    await extinguish_inline_kb(
        state=state, bot=bot, chat_id=chat_id, fsm_key=_FSM_KEY_IMGQ_MSG_ID
    )


async def _entry_setup(
    session: AsyncSession, *, user_id: int, session_id: int
) -> tuple[str, EntryTypeConfig, EntryPromptBundle]:
    """(kind, конфиг текстов, промпт-бандл) по dreams.kind сессии; дефолт dream."""
    from adapters.telegram.handlers.quick import _entry_kind_for_session

    kind = await _entry_kind_for_session(session, user_id=user_id, session_id=session_id)
    return kind, entry_config(kind), prompt_bundle(kind)


async def _send_offer_message(
    message: Message, state: FSMContext, cfg: EntryTypeConfig
) -> None:
    """Показать приглашение с inline [Я отвечу][Не хочу отвечать], запомнить msg_id."""
    sent = await message.answer(cfg.preq_offer_text, reply_markup=image_offer_keyboard())
    await _remember_imgq_message(state, sent, html_escape(cfg.preq_offer_text))


async def start_image_questions_flow(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    session_id: int,
    dream_id: int,
    is_onboarding: bool,
    settings: Settings,
    bot: Bot | None,
) -> None:
    """Точка входа фичи (offer-first): status=offered → приглашение → OFFERING.

    Ноль LLM-затрат до выбора юзера. Кладёт в FSM стандартные session_id/dream_id —
    хэндлеры состояний едины для обычного потока и онбординга.
    """
    kind, cfg, _bundle = await _entry_setup(
        session, user_id=user.id, session_id=session_id
    )
    await _save_imgq(session, session_id, {"status": "offered"})
    await state.update_data(session_id=session_id, dream_id=dream_id)
    await state.set_state(QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS)
    await _send_offer_message(message, state, cfg)
    await emit_append_event(
        session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_OFFERED,
        metadata={"session_id": session_id, "is_onboarding": is_onboarding,
                  "entry_type": kind},
    )
    if is_onboarding:
        await emit_funnel_step(
            session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.ONB_IMG_Q_OFFERED,
            metadata={"session_id": session_id, "entry_type": kind},
        )
    log.info("imgq.offered", session_id=session_id, is_onboarding=is_onboarding)


async def _start_analysis(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    session_id: int,
    dream_id: int,
    openrouter_client: AsyncOpenAI,
    context_client: AsyncOpenAI,
    context_model: str,
    settings: Settings,
    bot: Bot | None,
) -> None:
    """Диспетчер старта разбора: онбординг → _run_quick_for_onboarding (со своим
    ANALYSIS_START_NOTICE), обычный → _run_initial_quick (thinking_status).
    Никаких новых сообщений перед разбором (спека §1.9)."""
    sess = await SessionRepository(session).get_by_id(session_id, user_id=user.id)
    if sess is not None and sess.is_onboarding_session:
        from adapters.telegram.handlers.onboarding import _run_quick_for_onboarding

        await _run_quick_for_onboarding(
            message=message, state=state, session=session, user_id=user.id,
            telegram_user_id=user.telegram_user_id, session_id=session_id,
            dream_id=dream_id, openrouter_client=openrouter_client,
            context_client=context_client, context_model=context_model,
            settings=settings,
        )
    else:
        from adapters.telegram.handlers.quick import _run_initial_quick

        await _run_initial_quick(
            message=message, state=state, session=session, user_id=user.id,
            session_id=session_id, openrouter_client=openrouter_client,
            context_client=context_client, context_model=context_model,
            settings=settings, bot=bot,
        )


async def _ask_current_question(
    *, message: Message, state: FSMContext, imgq: dict, cfg: EntryTypeConfig
) -> None:
    """Показать вопрос items[idx]; шаблон per-kind (счётчик только при total>=2)."""
    idx, total = int(imgq["idx"]), int(imgq["total"])
    item = imgq["items"][idx]
    tmpl = cfg.preq_tmpl_many if total >= 2 else cfg.preq_tmpl_one
    text = tmpl.format(
        n=idx + 1, total=total, name=item.get("name", ""), question=item["question"]
    )
    rendered = render_quick_markup(text)  # **…**→<b>, нормализация тире/кавычек LLM
    sent = await message.answer(
        rendered, reply_markup=image_question_keyboard(total=total)
    )
    await _remember_imgq_message(state, sent, rendered)


async def resume_image_questions(
    *, callback_message: Message, state: FSMContext, session: AsyncSession,
    user: User, session_id: int, dream_id: int, imgq: dict,
    settings: Settings, bot: Bot | None,
    openrouter_client: AsyncOpenAI | None = None,
    context_client: AsyncOpenAI | None = None,
    context_model: str = "openai/gpt-4o-mini",
) -> bool:
    """Ре-энтри в фичу при resume сессии БЕЗ initial-разбора (спека §5.3).

    answering → переспросить текущий вопрос (вопросы не «протухают» — сон тот же).
    offered → повторить приглашение. done/no_images/declined без initial (paywall
    на финише §7.8 или упавший разбор) → deny_entry → разбор с собранными
    ответами (нужны клиенты; без них — False, старый путь). False = не наш случай.

    Аудит гардов (Task 15): отдельных guard-списков для правки НЕТ.
    - Авто-пауза (`auto_pause_active_quick`) ищет IN_PROGRESS-сессию по БД без
      оглядки на FSM-состояние → новые состояния фичи покрыты автоматически;
      экстрактор памяти на паузе отрабатывает по уже отвеченным парам.
    - BusyGuardMiddleware работает по in-memory локам (не по спискам состояний) —
      хэндлеры фичи берут те же `qa_user_lock`/`stt_lock`.
    - Thinking-guard покрывает своп в GENERATING_INITIAL через `thinking_status`.
    """
    status = imgq.get("status")
    _kind, cfg, _bundle = await _entry_setup(
        session, user_id=user.id, session_id=session_id
    )
    if status == "answering" and imgq.get("items"):
        await state.set_state(QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS)
        await state.set_data({"dream_id": dream_id, "session_id": session_id})
        await _ask_current_question(
            message=callback_message, state=state, imgq=imgq, cfg=cfg
        )
        log.info("imgq.resumed_answering", session_id=session_id, idx=imgq.get("idx"))
        return True
    if status == "offered":
        await state.set_state(QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS)
        await state.set_data({"dream_id": dream_id, "session_id": session_id})
        await _send_offer_message(callback_message, state, cfg)
        log.info("imgq.resumed_offered", session_id=session_id)
        return True
    if status in ("done", "no_images", "declined"):
        # Разбора нет (paywall на финише / упавший разбор) → deny_entry → разбор
        # с уже собранными ответами. Ответы не теряются (§5.3).
        if openrouter_client is None or context_client is None:
            return False  # нет клиентов (легаси-вызов) — старый путь
        if await deny_entry(callback_message, session, user, settings):
            return True  # пейвол показан, состояние не трогаем
        async with qa_user_lock(user.telegram_user_id) as acquired:
            if not acquired:
                await callback_message.answer(THINKING_GUARD_NOTICE)
                return True
            await state.set_data({"dream_id": dream_id, "session_id": session_id})
            await _start_analysis(
                message=callback_message, state=state, session=session, user=user,
                session_id=session_id, dream_id=dream_id,
                openrouter_client=openrouter_client, context_client=context_client,
                context_model=context_model, settings=settings, bot=bot,
            )
        return True
    return False


async def _emit_onb_done_if_onboarding(
    *, session: AsyncSession, user: User, session_id: int, choice: str, kind: str
) -> None:
    """onb_img_q_done = «юзер прошёл шаг, не застрял» (decline ИЛИ flow_done)."""
    sess = await SessionRepository(session).get_by_id(session_id, user_id=user.id)
    if sess is not None and sess.is_onboarding_session:
        await emit_funnel_step(
            session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.ONB_IMG_Q_DONE,
            metadata={"choice": choice, "entry_type": kind},
        )


async def _apply_decline(
    *, message, state, session, user, session_id, dream_id, via,
    openrouter_client, context_client, context_model, settings, bot,
    kind=KIND_DREAM, cfg=None, bundle=None,
) -> None:
    imgq = await _load_imgq(session, session_id)
    await _save_imgq(session, session_id, {**imgq, "status": "declined"})
    await emit_append_event(
        session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_CHOICE,
        metadata={"session_id": session_id, "choice": "decline", "via": via,
                  "entry_type": kind},
    )
    await _emit_onb_done_if_onboarding(
        session=session, user=user, session_id=session_id, choice="decline", kind=kind
    )
    await _start_analysis(
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id,
        openrouter_client=openrouter_client, context_client=context_client,
        context_model=context_model, settings=settings, bot=bot,
    )


async def _run_image_extraction_and_ask(
    *, message, state, session, user, session_id, dream_id,
    openrouter_client, context_client, context_model, settings, bot,
    kind=KIND_DREAM, cfg=None, bundle=None,
) -> None:
    """«Я отвечу» принят: платный вызов пре-вопросов → вопросы или молча разбор."""
    from adapters.telegram.handlers.quick import _build_quick_context

    cfg = cfg or entry_config(kind)
    bundle = bundle or prompt_bundle(kind)
    ctx = await _build_quick_context(
        session=session, user_id=user.id, session_id=session_id, settings=settings
    )
    items, fail_reason = [], "llm_error"
    if ctx is not None:
        result = None
        async with thinking_status(
            message, state=state,
            thinking_state=QuickAnalysisStates.GENERATING_INITIAL,
            statuses=list(cfg.preq_thinking_statuses),
        ):
            try:
                result = await generate_quick_response(
                    client=openrouter_client,
                    messages=bundle.build_pre_questions_messages(ctx),
                )
            except Exception:  # noqa: BLE001 — безопасный дефолт: молча разбор
                log.warning("imgq.extraction.call_failed", session_id=session_id,
                            exc_info=True)
        if result is not None and (result.input_tokens or result.output_tokens):
            await UsageRecorder(UsageEventRepository(session)).record_openrouter_generator(
                user_id=user.id, session_id=session_id,
                model=result.model_used or _QUICK_FALLBACK_CHAIN[0],
                input_tokens=result.input_tokens, output_tokens=result.output_tokens,
                prompt_version=bundle.version_pre_questions,
            )
        if result is not None and result.text:
            items, fail_reason = bundle.parse_pre_questions(result.text)

    if not items:
        await _save_imgq(session, session_id, {"status": "no_images"})
        await emit_append_event(
            session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.IMG_Q_NO_IMAGES,
            metadata={"session_id": session_id, "reason": fail_reason or "empty",
                      "entry_type": kind},
        )
        log.info("imgq.no_images", session_id=session_id, reason=fail_reason)
        await _start_analysis(
            message=message, state=state, session=session, user=user,
            session_id=session_id, dream_id=dream_id,
            openrouter_client=openrouter_client, context_client=context_client,
            context_model=context_model, settings=settings, bot=bot,
        )
        return

    imgq = {
        "status": "answering",
        "items": [
            {"name": q.name, "question": q.question, "answer": None, "skipped": False}
            for q in items
        ],
        "idx": 0,
        "total": len(items),
    }
    await _save_imgq(session, session_id, imgq)
    await state.set_state(QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS)
    await _ask_current_question(message=message, state=state, imgq=imgq, cfg=cfg)
    log.info("imgq.questions_started", session_id=session_id, total=len(items))


async def handle_imgq_offer_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openrouter_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    settings: Settings,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    if callback.from_user is None or callback.message is None or callback.data is None:
        return
    await safe_answer(callback)
    chat_id = callback.message.chat.id if callback.message.chat else None
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(callback.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    kind, cfg, bundle = await _entry_setup(
        session, user_id=user.id, session_id=session_id
    )
    choice_label = IMG_Q_ANSWER_BTN if callback.data == IMGQ_CALLBACK_YES else QA_SKIP_BTN
    await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id, choice=choice_label)
    # Гашение клавы при первом тапе + qa_user_lock: double-tap «Я отвечу» не породит
    # второй вызов выделения (краевой случай №14 спеки).
    async with qa_user_lock(callback.from_user.id) as acquired:
        if not acquired:
            await callback.message.answer(THINKING_GUARD_NOTICE)
            return
        # Гард double-tap/гонки: приглашение уже разрешено (выделение прошло
        # между тапами) — не запускаем повторное выделение и не дублируем decline.
        imgq_now = await _load_imgq(session, session_id)
        if imgq_now.get("status") != "offered":
            if imgq_now.get("status") == "answering" and imgq_now.get("items"):
                await state.set_state(QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS)
                await _ask_current_question(
                    message=callback.message, state=state, imgq=imgq_now, cfg=cfg
                )
            return
        common = dict(
            message=callback.message, state=state, session=session, user=user,
            session_id=session_id, dream_id=dream_id,
            openrouter_client=openrouter_client,
            context_client=openrouter_classifier_client,
            context_model=openrouter_context_model, settings=settings, bot=bot,
            kind=kind, cfg=cfg, bundle=bundle,
        )
        if callback.data == IMGQ_CALLBACK_NO:
            await _apply_decline(via="button", **common)
            return
        await emit_append_event(
            session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.IMG_Q_CHOICE,
            metadata={"session_id": session_id, "choice": "answer", "via": "button",
                      "entry_type": kind},
        )
        await _run_image_extraction_and_ask(**common)


async def _offer_act(
    *, text, via, message, state, session, user, session_id, dream_id, bot,
    openrouter_client, intent_client, intent_model, context_client, context_model,
    settings,
) -> None:
    # Гард устаревшего ввода: приглашение могло уже разрешиться, пока сообщение
    # было в пути (гонка с тапом «Я отвечу» во время выделения образов — текст
    # доставлен Telegram уже после релиза qa_lock, busy-guard его не поймал).
    # НЕ переспрашиваем приглашение (иначе затрём answering). Зеркалит _answer_act.
    kind, cfg, bundle = await _entry_setup(
        session, user_id=user.id, session_id=session_id
    )
    imgq_now = await _load_imgq(session, session_id)
    if imgq_now.get("status") != "offered":
        if imgq_now.get("status") == "answering" and imgq_now.get("items"):
            await state.set_state(QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS)
            await _ask_current_question(
                message=message, state=state, imgq=imgq_now, cfg=cfg
            )
        return
    # Сновая ветка — прежний классификатор байт-в-байт; остальные типы — нейтральный
    # (его поле answer показывается юзеру дословно, и «про сон» там было бы враньём).
    async with thinking_status(message, statuses=list(cfg.thinking_statuses)):
        if kind == KIND_DREAM:
            res = await classify_image_offer_intent(
                client=intent_client, model=intent_model, user_text=text
            )
            intent_version = INTENT_IMAGE_OFFER_VERSION
        else:
            res = await classify_entry_offer_intent(
                client=intent_client, model=intent_model, user_text=text
            )
            intent_version = INTENT_ENTRY_OFFER_VERSION
    await record_intent_usage(
        session=session, telegram_user_id=user.telegram_user_id, session_id=session_id,
        model=intent_model, input_tokens=res.input_tokens,
        output_tokens=res.output_tokens, prompt_version=intent_version,
    )
    chat_id = message.chat.id if message.chat else None
    common = dict(
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id,
        openrouter_client=openrouter_client, context_client=context_client,
        context_model=context_model, settings=settings, bot=bot,
        kind=kind, cfg=cfg, bundle=bundle,
    )
    if res.outcome == "answer":
        await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id,
                                choice=IMG_Q_ANSWER_BTN)
        await emit_append_event(
            session, user_id=user.id, telegram_user_id=user.telegram_user_id,
            event_type=AnalyticsEventType.IMG_Q_CHOICE,
            metadata={"session_id": session_id, "choice": "answer", "via": via,
                      "entry_type": kind},
        )
        await _run_image_extraction_and_ask(**common)
        return
    if res.outcome == "decline":
        await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id, choice=QA_SKIP_BTN)
        await _apply_decline(via=via, **common)
        return
    # question → показать answer-поле; unclear → переспрос. Состояние не меняется,
    # приглашение повторяется с кнопками (спека §2.1 шаг 2в, §6.3).
    await extinguish_inline_kb(state=state, bot=bot, chat_id=chat_id,
                               fsm_key=_FSM_KEY_IMGQ_MSG_ID)
    if res.outcome == "question" and res.answer:
        await message.answer(render_quick_markup(normalize_punctuation(res.answer)))
    else:
        await message.answer(cfg.preq_unclear)
    await _send_offer_message(message, state, cfg)


async def handle_imgq_offer_text(
    message: Message, state: FSMContext, session: AsyncSession, bot: Bot,
    openrouter_client: AsyncOpenAI, openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str, settings: Settings,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    if message.from_user is None or message.text is None:
        return
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _offer_act(
            text=message.text.strip(), via="text", message=message, state=state,
            session=session, user=user, session_id=session_id, dream_id=dream_id,
            bot=bot, openrouter_client=openrouter_client,
            intent_client=openrouter_classifier_client,
            intent_model=openrouter_intent_model,
            context_client=openrouter_classifier_client,
            context_model=openrouter_context_model, settings=settings,
        )


async def handle_imgq_offer_voice(
    message: Message, state: FSMContext, session: AsyncSession, bot: Bot,
    openai_client: AsyncOpenAI, openrouter_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI, openrouter_intent_model: str,
    settings: Settings, openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """Голос на приглашении: STT (под stt_lock middleware) → _offer_act(via=voice)."""
    if message.from_user is None or message.voice is None:
        return
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    stt = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client,
        log_prefix="imgq.offer", session=session, settings=settings,
    )
    if stt is None:
        return  # STT-сбой: юзер видел ошибку, состояние не двигается (§7.6)
    await UsageRecorder(UsageEventRepository(session)).record_whisper(
        user_id=user.id, audio_seconds=stt.duration_seconds, session_id=session_id
    )
    await _offer_act(
        text=stt.text.strip(), via="voice", message=message, state=state,
        session=session, user=user, session_id=session_id, dream_id=dream_id,
        bot=bot, openrouter_client=openrouter_client,
        intent_client=openrouter_classifier_client,
        intent_model=openrouter_intent_model,
        context_client=openrouter_classifier_client,
        context_model=openrouter_context_model, settings=settings,
    )


async def _pause_session_safe(
    *, session, session_id, user_id, dream_id, settings, classifier_client,
) -> None:
    """Пауза с экстракцией памяти (lazy-import _pause_session из quick.py)."""
    from adapters.telegram.handlers.quick import _pause_session

    await _pause_session(
        session=session, session_id=session_id, user_id=user_id, dream_id=dream_id,
        settings=settings, openrouter_classifier_client=classifier_client,
    )


async def _advance_or_finish(
    *, message, state, session, user, session_id, dream_id, imgq,
    openrouter_client, context_client, context_model, settings, bot,
    kind=KIND_DREAM, cfg=None, bundle=None,
) -> None:
    cfg = cfg or entry_config(kind)
    if int(imgq["idx"]) < int(imgq["total"]):
        await _ask_current_question(message=message, state=state, imgq=imgq, cfg=cfg)
        return
    imgq["status"] = "done"
    await _save_imgq(session, session_id, imgq)
    answered = sum(1 for i in imgq["items"] if i.get("answer"))
    skipped = sum(1 for i in imgq["items"] if i.get("skipped"))
    await emit_append_event(
        session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_FLOW_DONE,
        metadata={"session_id": session_id, "answered": answered,
                  "skipped": skipped, "total": int(imgq["total"]),
                  "entry_type": kind},
    )
    await _emit_onb_done_if_onboarding(
        session=session, user=user, session_id=session_id, choice="answer", kind=kind
    )
    # Paywall-чек перед финальным разбором (§7.8): лимит мог быть перейдён за время
    # диалога. status="done" сохранён → resume-правило «done без initial» доразберёт.
    if await is_over_limit(session, user, settings):
        await show_paywall(message, session, user, settings)
        await _pause_session_safe(
            session=session, session_id=session_id, user_id=user.id,
            dream_id=dream_id, settings=settings, classifier_client=context_client,
        )
        await state.clear()
        return
    await _start_analysis(
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id,
        openrouter_client=openrouter_client, context_client=context_client,
        context_model=context_model, settings=settings, bot=bot,
    )


async def _apply_skip(
    *, message, state, session, user, session_id, dream_id, via, imgq,
    openrouter_client, context_client, context_model, settings, bot,
    kind=KIND_DREAM, cfg=None, bundle=None,
) -> None:
    idx = int(imgq["idx"])
    imgq["items"][idx]["skipped"] = True
    imgq["idx"] = idx + 1
    await _save_imgq(session, session_id, imgq)
    await emit_append_event(
        session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_SKIPPED,
        metadata={"session_id": session_id, "idx": idx, "total": int(imgq["total"]),
                  "image": imgq["items"][idx]["name"], "via": via,
                  "entry_type": kind},
    )
    await _advance_or_finish(
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id, imgq=imgq,
        openrouter_client=openrouter_client, context_client=context_client,
        context_model=context_model, settings=settings, bot=bot,
        kind=kind, cfg=cfg, bundle=bundle,
    )


async def _answer_act(
    *, text, via, content_type, audio_file_id, message, state, session, user,
    session_id, dream_id, bot, openrouter_client, intent_client, intent_model,
    context_client, context_model, settings,
) -> None:
    imgq = await _load_imgq(session, session_id)
    if imgq.get("status") != "answering" or not imgq.get("items"):
        return  # устаревший ввод — состояние уже ушло
    kind, cfg, bundle = await _entry_setup(
        session, user_id=user.id, session_id=session_id
    )
    idx = int(imgq.get("idx") or 0)
    item = imgq["items"][idx]
    stripped = text.strip()
    common = dict(
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id,
        openrouter_client=openrouter_client, context_client=context_client,
        context_model=context_model, settings=settings, bot=bot,
        kind=kind, cfg=cfg, bundle=bundle,
    )
    chat_id = message.chat.id if message.chat else None
    # Детерминированный гард №1: точный текст кнопки → skip без LLM (§4.3).
    if stripped == QA_SKIP_BTN:
        await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id, choice=QA_SKIP_BTN)
        await _apply_skip(via=via, imgq=imgq, **common)
        return
    # Гард №2 (длина >240 → answer без LLM) — внутри обоих классификаторов.
    # Сновая ветка идёт прежним классификатором байт-в-байт; у остальных типов свой
    # нейтральный: его поле answer при outcome="question" показывается юзеру дословно.
    async with thinking_status(message, statuses=list(cfg.thinking_statuses)):
        if kind == KIND_DREAM:
            res = await classify_image_answer_intent(
                client=intent_client, model=intent_model,
                user_text=stripped, bot_question=item["question"],
            )
            answer_intent_version = INTENT_IMAGE_ANSWER_VERSION
        else:
            res = await classify_entry_answer_intent(
                client=intent_client, model=intent_model,
                user_text=stripped, bot_question=item["question"],
            )
            answer_intent_version = INTENT_ENTRY_ANSWER_VERSION
    await record_intent_usage(
        session=session, telegram_user_id=user.telegram_user_id, session_id=session_id,
        model=intent_model, input_tokens=res.input_tokens,
        output_tokens=res.output_tokens, prompt_version=answer_intent_version,
    )
    if res.outcome == "question":
        # Перефраз БЕЗ интерпретации; тот же вопрос, idx на месте, без лимита (§4.3.3).
        await extinguish_inline_kb(state=state, bot=bot, chat_id=chat_id,
                                   fsm_key=_FSM_KEY_IMGQ_MSG_ID)
        if res.answer:
            await message.answer(render_quick_markup(normalize_punctuation(res.answer)))
        await _ask_current_question(message=message, state=state, imgq=imgq, cfg=cfg)
        return
    await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id,
                            choice=QA_SKIP_BTN if res.outcome == "skip" else "ответ")
    if res.outcome == "skip":
        await _apply_skip(via=via, imgq=imgq, **common)
        return
    # answer (вкл. дефолт при сбое и гард длины): лог ПАРОЙ + прогресс + событие.
    await log_bot_message(
        session, session_id=session_id, content=item["question"],
        phase_label=cfg.phase_pre_q,
    )
    await log_user_message(
        session, session_id=session_id, content=stripped,
        content_type=content_type, audio_file_id=audio_file_id,
    )
    item["answer"] = stripped
    imgq["idx"] = idx + 1
    await _save_imgq(session, session_id, imgq)
    await emit_append_event(
        session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_ANSWERED,
        metadata={"session_id": session_id, "idx": idx, "total": int(imgq["total"]),
                  "image": item["name"], "via": via, "entry_type": kind},
    )
    await _advance_or_finish(imgq=imgq, **common)


async def handle_imgq_question_callback(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession, bot: Bot,
    openrouter_client: AsyncOpenAI, openrouter_classifier_client: AsyncOpenAI,
    settings: Settings, openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    if callback.from_user is None or callback.message is None or callback.data is None:
        return
    await safe_answer(callback)
    chat_id = callback.message.chat.id if callback.message.chat else None
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(callback.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    async with qa_user_lock(callback.from_user.id) as acquired:
        if not acquired:
            await callback.message.answer(THINKING_GUARD_NOTICE)
            return
        imgq = await _load_imgq(session, session_id)
        if imgq.get("status") != "answering":
            await extinguish_inline_kb(state=state, bot=bot, chat_id=chat_id,
                                       fsm_key=_FSM_KEY_IMGQ_MSG_ID)
            return
        kind, cfg, bundle = await _entry_setup(
            session, user_id=user.id, session_id=session_id
        )
        common = dict(
            message=callback.message, state=state, session=session, user=user,
            session_id=session_id, dream_id=dream_id,
            openrouter_client=openrouter_client,
            context_client=openrouter_classifier_client,
            context_model=openrouter_context_model, settings=settings, bot=bot,
            kind=kind, cfg=cfg, bundle=bundle,
        )
        if callback.data == IMGQ_CALLBACK_SKIP:
            await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id,
                                    choice=QA_SKIP_BTN)
            await _apply_skip(via="button", imgq=imgq, **common)
            return
        # skip_all: все оставшиеся skipped=true, событие на каждый (§2.1 шаг 5в).
        from adapters.telegram.texts import IMG_Q_SKIP_ALL_BTN

        await _mark_imgq_choice(state=state, bot=bot, chat_id=chat_id,
                                choice=IMG_Q_SKIP_ALL_BTN)
        total = int(imgq["total"])
        for i in range(int(imgq["idx"]), total):
            imgq["items"][i]["skipped"] = True
            await emit_append_event(
                session, user_id=user.id, telegram_user_id=user.telegram_user_id,
                event_type=AnalyticsEventType.IMG_Q_SKIPPED,
                metadata={"session_id": session_id, "idx": i, "total": total,
                          "image": imgq["items"][i]["name"], "via": "skip_all",
                          "entry_type": kind},
            )
        imgq["idx"] = total
        await _save_imgq(session, session_id, imgq)
        await _advance_or_finish(imgq=imgq, **common)


async def handle_imgq_answer_text(
    message: Message, state: FSMContext, session: AsyncSession, bot: Bot,
    openrouter_client: AsyncOpenAI, openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str, settings: Settings,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    if message.from_user is None or message.text is None:
        return
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _answer_act(
            text=message.text, via="text",
            content_type=SessionMessageContentType.TEXT, audio_file_id=None,
            message=message, state=state, session=session, user=user,
            session_id=session_id, dream_id=dream_id, bot=bot,
            openrouter_client=openrouter_client,
            intent_client=openrouter_classifier_client,
            intent_model=openrouter_intent_model,
            context_client=openrouter_classifier_client,
            context_model=openrouter_context_model, settings=settings,
        )


async def handle_imgq_answer_voice(
    message: Message, state: FSMContext, session: AsyncSession, bot: Bot,
    openai_client: AsyncOpenAI, openrouter_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI, openrouter_intent_model: str,
    settings: Settings, openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """Голос на вопросе: STT (под stt_lock middleware) + record_whisper → _answer_act."""
    if message.from_user is None or message.voice is None:
        return
    data = await state.get_data()
    session_id, dream_id = data.get("session_id"), data.get("dream_id")
    user = await UserRepository(session).get_by_tg_id(message.from_user.id)
    if user is None or session_id is None or dream_id is None:
        return
    stt = await transcribe_with_placeholder(
        message=message, bot=bot, openai_client=openai_client,
        log_prefix="imgq.answer", session=session, settings=settings,
    )
    if stt is None:
        return  # STT-сбой: стандартная ошибка показана, состояние не двигается
    await UsageRecorder(UsageEventRepository(session)).record_whisper(
        user_id=user.id, audio_seconds=stt.duration_seconds, session_id=session_id
    )
    await _answer_act(
        text=stt.text, via="voice",
        content_type=SessionMessageContentType.VOICE,
        audio_file_id=message.voice.file_id,
        message=message, state=state, session=session, user=user,
        session_id=session_id, dream_id=dream_id, bot=bot,
        openrouter_client=openrouter_client,
        intent_client=openrouter_classifier_client,
        intent_model=openrouter_intent_model,
        context_client=openrouter_classifier_client,
        context_model=openrouter_context_model, settings=settings,
    )


async def handle_imgq_stale_callback(callback: CallbackQuery, state: FSMContext) -> None:
    """Тап по устаревшей кнопке фичи (состояние уже ушло) → тихий ответ + снять клаву."""
    await safe_answer(callback)
    if callback.message is not None and callback.bot is not None:
        try:
            await callback.bot.edit_message_reply_markup(
                chat_id=callback.message.chat.id,
                message_id=callback.message.message_id, reply_markup=None,
            )
        except Exception:  # noqa: BLE001
            log.debug("imgq.stale_callback.extinguish_failed", exc_info=True)


def create_router() -> Router:
    router = Router(name="image_questions")
    router.callback_query.register(
        handle_imgq_offer_callback,
        QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS,
        F.data.in_({IMGQ_CALLBACK_YES, IMGQ_CALLBACK_NO}),
    )
    router.message.register(
        handle_imgq_offer_text,
        QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_imgq_offer_voice, QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS, F.voice,
    )
    router.callback_query.register(
        handle_imgq_question_callback,
        QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS,
        F.data.in_({IMGQ_CALLBACK_SKIP, IMGQ_CALLBACK_SKIP_ALL}),
    )
    router.message.register(
        handle_imgq_answer_text,
        QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_imgq_answer_voice, QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS, F.voice,
    )
    # Устаревшие кнопки фичи (спека §7.9): БЕЗ StateFilter, регистрация ПОСЛЕДНЕЙ —
    # сработает только если state-bound хэндлеры выше не подошли.
    router.callback_query.register(handle_imgq_stale_callback, F.data.startswith("imgq:"))
    return router
