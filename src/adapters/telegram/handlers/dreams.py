"""/dreams — список снов юзера. /dream <id> и /dream_<id> — карточка."""

from __future__ import annotations

import re
from html import escape as html_escape

import structlog
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.handlers._common import (
    auto_pause_active_quick,
    discard_pending_entry,
    discard_pending_newdream,
)
from adapters.telegram.keyboards import (
    dream_actions_keyboard,
    dream_delete_confirm_1_keyboard,
    dream_delete_confirm_2_keyboard,
    dream_list_item_inline_keyboard,
    qa_options_keyboard,
    remove_keyboard,
)
from adapters.telegram.quick_markup import render_quick_markup
from adapters.telegram.sender import answer_long, split_for_telegram
from adapters.telegram.session_log import load_dream_qa_for_display
from adapters.telegram.spend_guard import is_over_limit, show_paywall
from adapters.telegram.states import DreamCardStates
from adapters.telegram.texts import (
    DREAM_ACTION_START_ANALYSIS,
    DREAM_CARD_RESUME_NO_REPLY,
    DREAM_ID_USAGE,
    DREAM_LIST_ITEM_TEMPLATE,
    DREAM_NOT_FOUND,
)
from adapters.telegram.utils import format_date_ru
from app.config import Settings
from data.models import Dream, SessionStatus, User
from data.repositories import (
    DreamRepository,
    SessionRepository,
    SessionStateRepository,
    UserRepository,
)
from domain.entry_types import (
    ENTRY_TYPES,
    KIND_CONFLICT,
    KIND_DREAM,
    KIND_SITUATION,
    EntryTypeConfig,
    entry_config,
)

log = structlog.get_logger(__name__)

PREVIEW_LEN = 60
LIST_LIMIT = 20

_DREAM_CMD_PATTERN = re.compile(r"^/dream(?:_|\s+)(\d+)$")


def _preview(text: str, limit: int = PREVIEW_LEN) -> str:
    text = text.strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


async def _render_entry_list(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    cfg: EntryTypeConfig,
    settings: Settings | None,
    openrouter_classifier_client: AsyncOpenAI | None,
) -> None:
    """Список обращений одного типа (фильтр dreams.kind). Механика прежняя."""
    if message.from_user is None:
        return

    # Юзер мог переключиться сюда из режима рассказа — выкидываем недосказанное.
    await discard_pending_newdream(
        state=state, session=session, telegram_user_id=message.from_user.id
    )
    await discard_pending_entry(
        state=state, session=session, telegram_user_id=message.from_user.id
    )
    # …или из открытой карточки — сбрасываем её state (смотрим список заново).
    if (await state.get_state() or "").startswith("DreamCardStates"):
        await state.clear()

    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(message.from_user.id)
    if user is None:
        await message.answer(cfg.list_empty)
        return

    paused = await auto_pause_active_quick(
        session=session, state=state, user_id=user.id,
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )
    if paused is not None:
        await message.answer(
            entry_config(paused.kind).auto_paused_notice, reply_markup=remove_keyboard()
        )

    dream_repo = DreamRepository(session)
    dreams = await dream_repo.list_by_user(
        user_id=user.id, limit=LIST_LIMIT, kind=cfg.kind
    )
    if not dreams:
        await message.answer(cfg.list_empty)
        return

    # Header
    await message.answer(cfg.list_header.format(count=len(dreams)))

    # Per-entry message с inline-keyboard
    for d in dreams:
        body = DREAM_LIST_ITEM_TEMPLATE.format(
            date=format_date_ru(d.recorded_at),
            preview=html_escape(_preview(d.transcription)),
        )
        await message.answer(
            body,
            reply_markup=dream_list_item_inline_keyboard(dream_id=d.id),
        )


async def handle_dreams_list(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    await _render_entry_list(
        message=message, state=state, session=session, cfg=ENTRY_TYPES[KIND_DREAM],
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )


async def handle_situations_list(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    await _render_entry_list(
        message=message, state=state, session=session,
        cfg=ENTRY_TYPES[KIND_SITUATION],
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )


async def handle_conflicts_list(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings | None = None,
    openrouter_classifier_client: AsyncOpenAI | None = None,
) -> None:
    await _render_entry_list(
        message=message, state=state, session=session,
        cfg=ENTRY_TYPES[KIND_CONFLICT],
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )


async def _show_dream_by_id(
    *,
    dream_id: int,
    telegram_user_id: int,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI | None = None,
    openrouter_client: AsyncOpenAI | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """Показать карточку сна по id: auto-pause active quick (если другой сон),
    QUICK_ALREADY_ACTIVE / has_analysis auto-resume / start_analysis по развилке.

    Развилка:
    - Активная IN_PROGRESS сессия по этому сну → карточка + QUICK_ALREADY_ACTIVE
      (без кнопки). Юзер уже в Q&A.
    - Разбор есть (хотя бы одна сессия с initial-сообщением) → карточка +
      отдельным сообщением initial-разбор + qa-пары + кнопка «Задать вопрос».
    - Разбора нет → карточка + кнопка «Разобрать сейчас».

    Auto-pause: если у юзера есть активная Quick-сессия по ДРУГОМУ сну —
    она паузится и шлётся уведомление по типу того обращения. На карточке того же сна,
    по которому идёт Q&A, auto-pause не срабатывает (`excluded_dream_id`).
    """
    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(telegram_user_id)
    if user is None:
        await message.answer(DREAM_NOT_FOUND)
        return

    dream_repo = DreamRepository(session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    if dream is None:
        await message.answer(DREAM_NOT_FOUND)
        return

    paused = await auto_pause_active_quick(
        session=session, state=state, user_id=user.id, excluded_dream_id=dream.id,
        settings=settings, openrouter_classifier_client=openrouter_classifier_client,
    )
    if paused is not None:
        await message.answer(
            entry_config(paused.kind).auto_paused_notice, reply_markup=remove_keyboard()
        )

    sess_repo = SessionRepository(session)
    active = await sess_repo.get_active_for_user_dream(user_id=user.id, dream_id=dream.id)

    cfg = entry_config(dream.kind)
    card_body = cfg.card_title_tmpl.format(
        date=format_date_ru(dream.recorded_at),
        transcription=html_escape(dream.transcription),
    )

    if active is not None and active.status == SessionStatus.IN_PROGRESS:
        await answer_long(
            message, f"{card_body}\n\n{cfg.already_active}", escape_html=False
        )
        return

    has_analysis = await dream_repo.dream_has_analysis(dream_id=dream.id)

    # Сессия застряла на вопросах по образам сна (пауза ДО initial-разбора): initial
    # ещё нет → has_analysis=False, но возобновлять надо В ВОПРОСЫ, а не показывать
    # «Разобрать сейчас» (тап по ней упрётся в QUICK_ALREADY_ACTIVE — тупик, из
    # которого юзер не выйдет). Резюмируем через _resume_qa_for_dream, который для
    # сессии без initial делегирует в resume_image_questions (T15/T17).
    if not has_analysis and settings.image_questions_enabled:
        from adapters.telegram.handlers.image_questions import IMGQ_STATE_KEY
        from adapters.telegram.handlers.quick import _resume_qa_for_dream

        pending = await sess_repo.latest_inactive_quick_for_dream(
            dream_id=dream.id, user_id=user.id
        )
        if pending is not None:
            imgq_state = (
                await SessionStateRepository(session).get_data(session_id=pending.id)
                or {}
            ).get(IMGQ_STATE_KEY)
            if isinstance(imgq_state, dict) and imgq_state.get("status") in (
                "offered", "answering", "done", "no_images", "declined"
            ):
                await answer_long(message, card_body, escape_html=False)
                await _resume_qa_for_dream(
                    callback_message=message, state=state, session=session,
                    user_id=user.id, dream_id=dream.id, auto_pause_other=False,
                    intro_text=None, include_last_reply=False,
                    source="dream_card_imgq_resume", settings=settings,
                    openrouter_client=openrouter_client,
                    openrouter_classifier_client=openrouter_classifier_client,
                    openrouter_context_model=openrouter_context_model,
                )
                return

    if has_analysis:
        # Над лимитом — показываем read-only без inline-клавы Q&A (в Q&A не пускаем).
        over_limit = await is_over_limit(session, user, settings)
        from adapters.telegram.handlers.quick import (  # inline to avoid circular dep
            _FSM_KEY_QA_MSG_ID,
            QA_OPTIONS_KEY,
            _resume_qa_for_dream,
        )

        # Карточка уходит без reply-клавы — «Завершить» теперь живёт inline под
        # последней репликой разбора (как в остальных Q&A-входах, Task 1–8).
        await answer_long(message, card_body, escape_html=False)

        if over_limit:
            # read-only: шлём разбор без inline-клавы и без входа в Q&A.
            loaded = await load_dream_qa_for_display(session, dream_id=dream.id)
            if loaded is not None:
                initial_text, qa_pairs = loaded
                await answer_long(
                    message, render_quick_markup(initial_text), escape_html=False
                )
                for user_q, bot_a in qa_pairs:
                    await answer_long(message, user_q)
                    await answer_long(
                        message, render_quick_markup(bot_a), escape_html=False
                    )
            await show_paywall(message, session, user, settings)
            return

        # Лейблы inline-чипов берём из session_state той сессии, которую
        # резюмирует _resume_qa_for_dream (latest_inactive_quick_for_dream).
        target_session = await sess_repo.latest_inactive_quick_for_dream(
            dream_id=dream.id, user_id=user.id
        )
        labels: list[str] = []
        if target_session is not None:
            saved = (
                await SessionStateRepository(session).get_data(
                    session_id=target_session.id
                )
                or {}
            ).get(QA_OPTIONS_KEY) or {}
            raw_labels = saved.get("labels")
            labels = raw_labels if isinstance(raw_labels, list) else []

        # Рендерим разбор; inline-клаву Q&A вешаем на ПОСЛЕДНЮЮ реплику
        # (последний bot_a, либо initial_text если Q&A-пар нет).
        loaded = await load_dream_qa_for_display(session, dream_id=dream.id)
        last_msg_id: int | None = None
        last_kb_html: str | None = None  # last chunk of the keyboard-bearing message
        if loaded is not None:
            initial_text, qa_pairs = loaded
            if qa_pairs:
                # initial + промежуточные пары без клавы; клава на последнем bot_a.
                await answer_long(
                    message, render_quick_markup(initial_text), escape_html=False
                )
                for idx, (user_q, bot_a) in enumerate(qa_pairs):
                    is_last = idx == len(qa_pairs) - 1
                    await answer_long(message, user_q)
                    last_rendered = render_quick_markup(bot_a)
                    sent = await answer_long(
                        message,
                        last_rendered,
                        escape_html=False,
                        reply_markup=qa_options_keyboard(labels) if is_last else None,
                    )
                    if is_last:
                        last_msg_id = getattr(sent, "message_id", None)
                        last_kb_html = split_for_telegram(last_rendered)[-1]
            else:
                # Q&A-пар нет — клава на initial_text.
                last_rendered = render_quick_markup(initial_text)
                sent = await answer_long(
                    message,
                    last_rendered,
                    escape_html=False,
                    reply_markup=qa_options_keyboard(labels),
                )
                last_msg_id = getattr(sent, "message_id", None)
                last_kb_html = split_for_telegram(last_rendered)[-1]

        # Fallback: dream_has_analysis=True, но load_dream_qa_for_display вернул None
        # (редкое рассогласование данных — initial-сообщение отсутствует). Без этого
        # блока юзер оказался бы в IN_QA без единой inline-клавы (нет чипов, нет
        # «Свой вопрос»/«Завершить») и без _FSM_KEY_QA_MSG_ID — stranded.
        # Решение: шлём одно короткое сообщение с fallback-клавой qa_options_keyboard([]).
        fallback_msg_id: int | None = None
        if loaded is None:
            log.warning("quick.card.display_load_failed", dream_id=dream.id, user_id=user.id)
            sent_fb = await message.answer(
                DREAM_CARD_RESUME_NO_REPLY,
                reply_markup=qa_options_keyboard([]),
            )
            fallback_msg_id = getattr(sent_fb, "message_id", None)

        # Auto-resume Q&A: резюмируем PAUSED-сессию (или последнюю completed).
        # intro_text=None / include_last_reply=False — _resume_qa_for_dream ничего
        # видимого не шлёт и НЕ трогает _FSM_KEY_QA_MSG_ID, НО делает
        # state.set_data({...}) (replace), стирая весь FSM-словарь. Поэтому msg_id
        # сохраняем ПОСЛЕ resume — иначе он будет затёрт.
        await _resume_qa_for_dream(
            callback_message=message,
            state=state,
            session=session,
            user_id=user.id,
            dream_id=dream.id,
            auto_pause_other=False,  # auto_pause_active_quick уже вызывали выше, с excluded_dream_id
            intro_text=None,
            include_last_reply=False,  # initial + Q&A pairs уже отправлены выше
            source="dream_card_open",
            settings=settings,
            openrouter_client=openrouter_client,
            openrouter_classifier_client=openrouter_classifier_client,
            openrouter_context_model=openrouter_context_model,
        )
        # Выбираем: msg_id от нормального рендера (last_msg_id) или от fallback.
        msg_id_to_store = last_msg_id if isinstance(last_msg_id, int) else fallback_msg_id
        if isinstance(msg_id_to_store, int):
            await state.update_data(**{_FSM_KEY_QA_MSG_ID: msg_id_to_store})
        # Persist kb_html into session_state for the resumed session so _mark_qa_choice can use it.
        if last_kb_html is not None and target_session is not None:
            from adapters.telegram.handlers.quick import QA_OPTIONS_KEY as _QA_OPTIONS_KEY
            state_repo_upd = SessionStateRepository(session)
            existing_full = await state_repo_upd.get_data(session_id=target_session.id) or {}
            existing_qa = existing_full.get(_QA_OPTIONS_KEY) or {}
            await state_repo_upd.upsert(
                session_id=target_session.id,
                data={**existing_full, _QA_OPTIONS_KEY: {**existing_qa, "kb_html": last_kb_html}},
            )
        return

    # Spend cap: неразобранный сон над лимитом — показываем карточку (read OK) и
    # пейвол, но БЕЗ кнопки «Разобрать» и БЕЗ входа в DECIDING_ANALYZE. Иначе юзер
    # застрянет в режиме ожидания, и любой его текст будет снова упираться в пейвол.
    if await is_over_limit(session, user, settings):
        await answer_long(message, card_body, escape_html=False)
        await show_paywall(message, session, user, settings)
        await state.clear()
        return

    await answer_long(
        message,
        card_body,
        escape_html=False,
        reply_markup=dream_actions_keyboard(
            dream_id=dream.id,
            text=DREAM_ACTION_START_ANALYSIS,
            callback_action="start_analysis",
        ),
    )
    # Свободный текст/голос на карточке тоже запускает разбор (DreamCardStates) —
    # как «Это всё» в loop'ах. Ключ совпадает с quick._DREAM_CARD_DREAM_ID.
    await state.set_state(DreamCardStates.DECIDING_ANALYZE)
    await state.update_data({"dream_card_dream_id": dream.id})


async def handle_dream_by_id(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI | None = None,
    openrouter_client: AsyncOpenAI | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """Карточка сна по `/dream 42`, `/dream_42` или одинокий `/dream` (usage)."""
    if message.from_user is None or message.text is None:
        return

    text = message.text.strip()
    if text == "/dream":
        await message.answer(DREAM_ID_USAGE)
        return

    match = _DREAM_CMD_PATTERN.match(text)
    if match is None:
        await message.answer(DREAM_ID_USAGE)
        return

    dream_id = int(match.group(1))
    await _show_dream_by_id(
        dream_id=dream_id,
        telegram_user_id=message.from_user.id,
        message=message,
        state=state,
        session=session,
        settings=settings,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_client=openrouter_client,
        openrouter_context_model=openrouter_context_model,
    )


async def handle_dream_open_callback(
    callback: CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI | None = None,
    openrouter_client: AsyncOpenAI | None = None,
    openrouter_context_model: str = "openai/gpt-4o-mini",
) -> None:
    """callback dream_open:<id> → дублирует /dream_<id> deeplink."""
    if callback.data is None or callback.message is None or callback.from_user is None:
        await safe_answer(callback)
        return
    try:
        dream_id = int(callback.data.split(":", 1)[1])
    except (IndexError, ValueError):
        await safe_answer(callback)
        return

    await safe_answer(callback)
    await _show_dream_by_id(
        dream_id=dream_id,
        telegram_user_id=callback.from_user.id,
        message=callback.message,  # type: ignore[arg-type]
        state=state,
        session=session,
        settings=settings,
        openrouter_classifier_client=openrouter_classifier_client,
        openrouter_client=openrouter_client,
        openrouter_context_model=openrouter_context_model,
    )


def _parse_dream_id(callback: CallbackQuery) -> int | None:
    if callback.data is None:
        return None
    try:
        return int(callback.data.split(":", 1)[1])
    except (IndexError, ValueError):
        return None


async def _fetch_dream_for_callback(
    callback: CallbackQuery, session: AsyncSession,
) -> tuple[User, Dream] | None:
    """Парс dream_id из callback.data + проверка принадлежности user."""
    dream_id = _parse_dream_id(callback)
    if dream_id is None or callback.from_user is None:
        return None
    user = await UserRepository(session).get_by_tg_id(callback.from_user.id)
    if user is None:
        return None
    dream = await DreamRepository(session).get_by_id(dream_id=dream_id, user_id=user.id)
    if dream is None:
        return None
    return (user, dream)


async def handle_dream_delete_callback(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession,
) -> None:
    """Step 1: dream_delete:<id> → показать первое подтверждение."""
    await safe_answer(callback)  # ack первым: «часики» гаснут до любой работы
    fetched = await _fetch_dream_for_callback(callback, session)
    if fetched is None or callback.message is None:
        return
    _, dream = fetched
    cfg = entry_config(dream.kind)
    await callback.message.edit_text(
        cfg.delete_confirm_1.format(date=format_date_ru(dream.recorded_at)),
        reply_markup=dream_delete_confirm_1_keyboard(dream_id=dream.id),
    )


async def handle_dream_delete_confirm_callback(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession,
) -> None:
    """Step 2: dream_delete_confirm:<id> → показать второе подтверждение «навсегда»."""
    await safe_answer(callback)
    fetched = await _fetch_dream_for_callback(callback, session)
    if fetched is None or callback.message is None:
        return
    _, dream = fetched
    cfg = entry_config(dream.kind)
    await callback.message.edit_text(
        cfg.delete_confirm_2,
        reply_markup=dream_delete_confirm_2_keyboard(dream_id=dream.id),
    )


async def handle_dream_delete_final_callback(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession,
) -> None:
    """Step 3: dream_delete_final:<id> → hard_delete_with_cascade."""
    await safe_answer(callback)
    fetched = await _fetch_dream_for_callback(callback, session)
    if fetched is None or callback.message is None:
        return
    user, dream = fetched
    dream_id = dream.id
    cfg = entry_config(dream.kind)  # ДО удаления — после него объект отвязан
    dream_repo = DreamRepository(session)
    deleted = await dream_repo.hard_delete_with_cascade(dream_id=dream_id, user_id=user.id)
    await session.commit()
    if not deleted:
        return
    await callback.message.edit_text(cfg.deleted_notice)
    log.info("dream.hard_deleted", dream_id=dream_id, user_id=user.id)


async def handle_dream_delete_cancel_callback(
    callback: CallbackQuery, state: FSMContext, session: AsyncSession,
) -> None:
    """Cancel на любом из 2 шагов — вернуть карточку с [↩|🗑]."""
    await safe_answer(callback)
    fetched = await _fetch_dream_for_callback(callback, session)
    if fetched is None or callback.message is None:
        return
    _, dream = fetched
    body = DREAM_LIST_ITEM_TEMPLATE.format(
        date=format_date_ru(dream.recorded_at),
        preview=html_escape(_preview(dream.transcription)),
    )
    await callback.message.edit_text(
        body,
        reply_markup=dream_list_item_inline_keyboard(dream_id=dream.id),
    )


def create_router() -> Router:
    """Создать новый экземпляр Router с зарегистрированными хэндлерами."""
    router = Router(name="dreams")
    router.message.register(handle_dreams_list, Command("dreams"))
    router.message.register(handle_situations_list, Command("external_situations"))
    router.message.register(handle_conflicts_list, Command("internal_conflicts"))
    router.message.register(
        handle_dream_by_id,
        F.text.regexp(r"^/dream(?:_\d+|\s+\d+|\b).*$"),
    )
    router.callback_query.register(
        handle_dream_open_callback,
        F.data.startswith("dream_open:"),
    )
    router.callback_query.register(
        handle_dream_delete_callback,
        F.data.startswith("dream_delete:"),
    )
    router.callback_query.register(
        handle_dream_delete_confirm_callback,
        F.data.startswith("dream_delete_confirm:"),
    )
    router.callback_query.register(
        handle_dream_delete_final_callback,
        F.data.startswith("dream_delete_final:"),
    )
    router.callback_query.register(
        handle_dream_delete_cancel_callback,
        F.data.startswith("dream_delete_cancel:"),
    )
    return router
