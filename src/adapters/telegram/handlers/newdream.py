"""/newdream — опытный юзер пишет новый сон и сразу идёт в Quick/Deep flow."""

from __future__ import annotations

from datetime import UTC, datetime
from html import escape as html_escape

import structlog
from aiogram import Bot, F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

# Shared helper — текст/voice → str (generic, reused from onboarding)
from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    record_intent_usage,
)
from adapters.telegram.handlers._loop_prompt import (
    extinguish_inline_kb,
    extinguish_loop_button,
    send_loop_prompt,
)
from adapters.telegram.handlers._new_entry import start_new_entry
from adapters.telegram.handlers._qa_lock import qa_user_lock
from adapters.telegram.handlers.onboarding import _get_text_from_message
from adapters.telegram.keyboards import (
    CONFIRM_ANALYZE_CALLBACK,
    CONFIRM_SAVE_CALLBACK,
    LOOP_DONE_CALLBACK,
    newdream_confirm_analysis_keyboard,
    remove_keyboard,
)
from adapters.telegram.sender import answer_long
from adapters.telegram.spend_guard import deny_entry
from adapters.telegram.states import NewDreamStates, QuickAnalysisStates
from adapters.telegram.texts import (
    ASK_PER_SESSION_CONTEXT,
    CONFIRM_ANALYZE_BTN,
    DREAM_CARD_TEMPLATE,
    DREAM_CONFIRM_ANALYZE_PROMPT,
    DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE,
    LOOP_ADD_BTN_DONE,
    LOOP_ADD_PROMPT_DREAM,
    NOT_ONBOARDED_PROMPT,
    SAVE_WITHOUT_ANALYSIS_BTN,
    THINKING_GUARD_NOTICE,
)
from adapters.telegram.thinking_status import thinking_status
from adapters.telegram.utils import format_date_ru
from app.config import Settings
from data.models import DreamStatus, SessionType
from data.repositories import (
    DreamRepository,
    SessionRepository,
    UserRepository,
)
from domain.entry_types import KIND_DREAM
from llm.intent import (
    INTENT_CONFIRM_VERSION,
    INTENT_LOOP_VERSION,
    classify_confirm_intent,
    classify_loop_intent,
)

log = structlog.get_logger(__name__)

# Ключ в FSM-data для message_id сообщения-гейта (inline-кнопки), чтобы погасить.
_FSM_KEY_CONFIRM_MSG_ID = "confirm_gate_msg_id"


async def _send_confirm_gate(*, message: Message, state: FSMContext) -> None:
    """Показать вопрос подтверждения с inline-кнопками и запомнить его message_id."""
    sent = await message.answer(
        DREAM_CONFIRM_ANALYZE_PROMPT, reply_markup=newdream_confirm_analysis_keyboard()
    )
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(**{_FSM_KEY_CONFIRM_MSG_ID: msg_id})

_FSM_KEY_NEWDREAM_DREAM_ID = "newdream_dream_id"


async def handle_newdream_command(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    bot: Bot | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    """/newdream: если онбординг пройден, начинаем сбор нового сна.

    Общее тело (гейты, авто-пауза активной сессии, выброс черновиков) живёт в
    `_new_entry.start_new_entry` — там же его берут команды /new_*.
    Здесь остаётся только «юзера нет в БД» → NOT_ONBOARDED.
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
            # Совсем новый юзер: тут /start уместен.
            await message.answer(NOT_ONBOARDED_PROMPT)
            return
    await start_new_entry(
        kind=KIND_DREAM,
        message=message,
        state=state,
        session=session,
        settings=settings,
        bot=bot,
        openrouter_classifier_client=openrouter_classifier_client,
        user=user,
    )


async def handle_newdream_dream_input(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    settings: Settings,
) -> None:
    """ASKING_DREAM: первая реплика юзера. Создаём Dream и переходим к DREAM_LOOP."""
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    repo = UserRepository(session)
    user = await repo.get_by_tg_id(message.from_user.id)
    if user is None:
        return
    dream_repo = DreamRepository(session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription=text
    )
    await state.update_data({_FSM_KEY_NEWDREAM_DREAM_ID: dream.id})
    await state.set_state(NewDreamStates.DREAM_LOOP)
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)


async def _show_confirmation_gate(
    *, message: Message, state: FSMContext, session: AsyncSession, dream_id: int, user_id: int
) -> None:
    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user_id)
    if dream is None:
        return
    card_body = DREAM_CARD_TEMPLATE.format(
        date=format_date_ru(dream.recorded_at),
        transcription=html_escape(dream.transcription),
    )
    log.info(
        "newdream.gate.card",
        dream_id=dream_id,
        user_id=user_id,
        transcription_len=len(dream.transcription),
        card_len=len(card_body),
    )
    # card_body уже html-escaped → answer_long с escape_html=False (иначе двойной escape).
    # Длинный сон (8-мин голосовое) бьётся на чанки ≤4096, иначе Telegram 400.
    await answer_long(message, card_body, escape_html=False)
    await state.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await _send_confirm_gate(message=message, state=state)
    log.info("newdream.confirmation_gate.shown", dream_id=dream_id, user_id=user_id)


async def _newdream_loop_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    """DREAM_LOOP /newdream: done → confirmation gate; continue → append.

    `message` — только для ответов (в callback-пути `callback.message`),
    юзер берётся по явному `telegram_user_id`.
    """
    repo = UserRepository(session)
    user = await repo.get_by_tg_id(telegram_user_id)
    if user is None:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    data = await state.get_data()
    dream_id = data.get(_FSM_KEY_NEWDREAM_DREAM_ID)
    if dream_id is None:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return

    is_done = text == LOOP_ADD_BTN_DONE
    if not is_done:
        async with thinking_status(message):
            res = await classify_loop_intent(client=client, model=model, user_text=text)
        await record_intent_usage(
            session=session, telegram_user_id=telegram_user_id, session_id=None,
            model=model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            prompt_version=INTENT_LOOP_VERSION,
        )
        if res.outcome == "question":
            # Встречный вопрос — отвечаем и повторяем тот же вопрос, не записывая.
            if res.answer:
                await message.answer(res.answer)
            await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
            return
        is_done = res.outcome == "done"

    if is_done:
        await _show_confirmation_gate(
            message=message, state=state, session=session, dream_id=dream_id, user_id=user.id
        )
        return

    # continue — append к Dream.transcription + stay in DREAM_LOOP
    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    if dream is None:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    dream.transcription = f"{dream.transcription}\n\n{text}"
    await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)


async def handle_newdream_dream_loop(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """DREAM_LOOP (текст): новое сообщение гасит старую кнопку; «всё» → confirmation gate; иначе append.

    Классификатор под qa_lock.
    """
    if message.from_user is None or message.text is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = message.text.strip()
    if not text:
        await send_loop_prompt(message=message, state=state, text=LOOP_ADD_PROMPT_DREAM)
        return
    if text == LOOP_ADD_BTN_DONE:
        await _newdream_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _newdream_loop_act(
            text=text, message=message, state=state, session=session,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )


async def handle_newdream_dream_loop_done_cb(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """Тап inline «Это всё» в /newdream DREAM_LOOP → гасим кнопку → confirmation gate."""
    if callback.from_user is None or callback.message is None:
        return
    await safe_answer(callback)
    await extinguish_loop_button(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
    )
    await _newdream_loop_act(
        text=LOOP_ADD_BTN_DONE, message=callback.message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=callback.from_user.id,
    )



async def _confirm_act(
    *,
    text: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    client: AsyncOpenAI,
    model: str,
    telegram_user_id: int,
) -> None:
    """Решить analyze/save/unclear (кнопка → точно, иначе классификатор) и выполнить.

    `message` — только для ответов (в callback-пути `callback.message`),
    юзер берётся по явному `telegram_user_id`.
    """
    repo = UserRepository(session)
    user = await repo.get_by_tg_id(telegram_user_id)
    if user is None:
        return
    data = await state.get_data()
    dream_id = data.get(_FSM_KEY_NEWDREAM_DREAM_ID)
    if dream_id is None:
        return

    # Spend cap: стена стоит ЗДЕСЬ, на решении «разобрать», а не шагом позже.
    # Раньше гейт пропускал дальше, бот просил рассказать о переживаниях и
    # только потом отказывал — человек тратил силы на выброшенное сообщение.
    # Заодно не платим за классификатор
    # свободного текста тому, кого всё равно не пустим.
    # «Сохранить без разбора» не блокируем: он бесплатный, а запись уже в базе.
    if text != SAVE_WITHOUT_ANALYSIS_BTN and await deny_entry(
        message, session, user, settings
    ):
        await state.clear()
        return

    if text == CONFIRM_ANALYZE_BTN:
        outcome = "analyze"
    elif text == SAVE_WITHOUT_ANALYSIS_BTN:
        outcome = "save"
    else:
        async with thinking_status(message):
            res = await classify_confirm_intent(client=client, model=model, user_text=text)
        await record_intent_usage(
            session=session, telegram_user_id=telegram_user_id, session_id=None,
            model=model, input_tokens=res.input_tokens, output_tokens=res.output_tokens,
            prompt_version=INTENT_CONFIRM_VERSION,
        )
        if res.outcome == "question":
            # Встречный вопрос про разбор — отвечаем и снова показываем гейт.
            if res.answer:
                await message.answer(res.answer)
            await _send_confirm_gate(message=message, state=state)
            return
        outcome = res.outcome

    if outcome == "analyze":
        sess_repo = SessionRepository(session)
        new_session = await sess_repo.create(
            dream_id=dream_id,
            user_id=user.id,
            type_=SessionType.QUICK,
            is_onboarding_session=False,
        )
        dream_repo = DreamRepository(session)
        dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
        if dream is not None and dream.status == DreamStatus.RECORDED:
            dream.status = DreamStatus.IN_ANALYSIS
        await state.set_data({"dream_id": dream_id, "session_id": new_session.id})
        await state.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
        await message.answer(ASK_PER_SESSION_CONTEXT, reply_markup=remove_keyboard())
        log.info(
            "newdream.session.created",
            session_id=new_session.id,
            dream_id=dream_id,
            user_id=user.id,
        )
        return

    if outcome == "save":
        await state.clear()
        await message.answer(DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE, reply_markup=remove_keyboard())
        log.info("newdream.saved_without_analysis", dream_id=dream_id, user_id=user.id)
        return

    # unclear — переспрашиваем кнопками (новый гейт, message_id запоминаем)
    await _send_confirm_gate(message=message, state=state)


async def handle_newdream_confirm_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """Тап inline-кнопки гейта → гасим кнопки → разобрать/сохранить.

    Spend-limit отбивает прямо здесь, в `_confirm_act`: тап «Разобрать» — это и
    есть момент решения, и стена обязана стоять на нём.
    """
    if callback.from_user is None or callback.message is None or callback.data is None:
        return
    await safe_answer(callback)
    await extinguish_inline_kb(
        state=state, bot=bot,
        chat_id=callback.message.chat.id if callback.message.chat else None,
        fsm_key=_FSM_KEY_CONFIRM_MSG_ID,
    )
    text = (
        CONFIRM_ANALYZE_BTN
        if callback.data == CONFIRM_ANALYZE_CALLBACK
        else SAVE_WITHOUT_ANALYSIS_BTN
    )
    await _confirm_act(
        text=text, message=callback.message, state=state, session=session,
        settings=settings,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=callback.from_user.id,
    )


async def handle_newdream_confirm_analysis(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    bot: Bot | None = None,
) -> None:
    """CONFIRMING_ANALYSIS (текст): ответ гасит кнопки гейта → разобрать/сохранить.

    Свободный текст классифицируется дешёвой моделью под qa_lock; unclear → повтор.
    """
    if message.from_user is None or message.text is None:
        return
    await extinguish_inline_kb(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None,
        fsm_key=_FSM_KEY_CONFIRM_MSG_ID,
    )
    text = message.text.strip()
    if text in (CONFIRM_ANALYZE_BTN, SAVE_WITHOUT_ANALYSIS_BTN):
        await _confirm_act(
            text=text, message=message, state=state, session=session,
            settings=settings,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )
        return
    async with qa_user_lock(message.from_user.id) as acquired:
        if not acquired:
            await message.answer(THINKING_GUARD_NOTICE)
            return
        await _confirm_act(
            text=text, message=message, state=state, session=session,
            settings=settings,
            client=openrouter_classifier_client, model=openrouter_intent_model,
            telegram_user_id=message.from_user.id,
        )


async def handle_newdream_confirm_analysis_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """CONFIRMING_ANALYSIS + голос: гасим кнопки → STT → analyze/save/unclear (под stt_lock)."""
    if message.from_user is None or message.voice is None:
        return
    await extinguish_inline_kb(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None,
        fsm_key=_FSM_KEY_CONFIRM_MSG_ID,
    )
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    await _confirm_act(
        text=text, message=message, state=state, session=session,
        settings=settings,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )


async def handle_newdream_dream_loop_voice(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    openai_client: AsyncOpenAI,
    openrouter_classifier_client: AsyncOpenAI,
    openrouter_intent_model: str,
    settings: Settings,
) -> None:
    """DREAM_LOOP + голос: гасим старую кнопку → STT → intent → done (gate) / continue (append).

    Весь handler под stt_lock (middleware) — отдельный qa_lock не нужен.
    """
    if message.from_user is None:
        return
    await extinguish_loop_button(
        state=state, bot=bot, chat_id=message.chat.id if message.chat else None
    )
    text = await _get_text_from_message(message, bot, openai_client, session, settings)
    if text is None:
        return
    # settings сюда НЕ передаём: пейвол живёт в _confirm_act (на кнопке
    # «Разобрать»), а шаг «дорассказать» бесплатный. Лишний kwarg здесь ронял
    # хэндлер TypeError'ом — прод-дефект 2026-09-03.
    await _newdream_loop_act(
        text=text, message=message, state=state, session=session,
        client=openrouter_classifier_client, model=openrouter_intent_model,
        telegram_user_id=message.from_user.id,
    )


def create_router() -> Router:
    """Создать роутер с зарегистрированными хэндлерами /newdream."""
    router = Router(name="newdream_handler")
    router.message.register(handle_newdream_command, Command("newdream"))
    # ~F.text.startswith("/"): команды (/dreams, /profile, /start, ...) НЕ должны
    # проглатываться как текст сна — пусть проваливаются к своим command-хэндлерам,
    # которые корректно выйдут из режима рассказа сна (discard_pending_newdream).
    router.message.register(
        handle_newdream_dream_input,
        NewDreamStates.ASKING_DREAM,
        (F.text & ~F.text.startswith("/")) | F.voice,
    )
    router.callback_query.register(
        handle_newdream_dream_loop_done_cb,
        NewDreamStates.DREAM_LOOP,
        F.data == LOOP_DONE_CALLBACK,
    )
    router.message.register(
        handle_newdream_dream_loop,
        NewDreamStates.DREAM_LOOP,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_newdream_dream_loop_voice, NewDreamStates.DREAM_LOOP, F.voice
    )
    router.callback_query.register(
        handle_newdream_confirm_callback,
        NewDreamStates.CONFIRMING_ANALYSIS,
        F.data.in_({CONFIRM_ANALYZE_CALLBACK, CONFIRM_SAVE_CALLBACK}),
    )
    router.message.register(
        handle_newdream_confirm_analysis,
        NewDreamStates.CONFIRMING_ANALYSIS,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(
        handle_newdream_confirm_analysis_voice,
        NewDreamStates.CONFIRMING_ANALYSIS,
        F.voice,
    )
    return router
