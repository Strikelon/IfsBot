"""Общий вход в новое обращение: /newdream и /new_*.

Гейты онбординга, авто-пауза активной сессии и выброс черновиков обязаны вести себя
одинаково во всех точках входа, поэтому логика собрана в одну функцию.
"""

from __future__ import annotations

import structlog
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers._common import (
    FSM_KEY_ENTRY_KIND,
    discard_pending_entry,
    discard_pending_newdream,
    entry_kind_for_dream,
)
from adapters.telegram.keyboards import remove_keyboard
from adapters.telegram.menu import sync_user_commands
from adapters.telegram.spend_guard import deny_entry
from adapters.telegram.states import EntryStates, NewDreamStates
from adapters.telegram.texts import (
    NEWDREAM_PROMPT,
    ONBOARDING_IN_PROGRESS_NOTICE,
)
from app.config import Settings
from data.models import SessionStatus, User
from data.repositories import SessionRepository, UserRepository
from domain.entry_types import KIND_DREAM, entry_config

log = structlog.get_logger(__name__)


async def _pause_active_session(
    *,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    settings: Settings,
    openrouter_classifier_client: AsyncOpenAI | None,
    current_state: str | None,
) -> None:
    """Закрыть активную Quick-сессию перед новым обращением.

    Два пути, как было в /newdream: по FSM (юзер прямо сейчас в разборе) и по БД
    (сессия висит IN_PROGRESS без FSM — stale или начата из /dreams). Локальный
    импорт _pause_session — против цикла импортов с quick.py.
    """
    from adapters.telegram.handlers.quick import _pause_session  # noqa: PLC0415

    sess_repo = SessionRepository(session)
    if current_state and current_state.startswith("QuickAnalysisStates"):
        data = await state.get_data()
        active_session_id = data.get("session_id")
        active_dream_id = data.get("dream_id")
        if active_session_id is not None and active_dream_id is not None:
            active = await sess_repo.get_by_id(active_session_id, user_id=user.id)
            if active is not None and active.status == SessionStatus.IN_PROGRESS:
                await _pause_session(
                    session=session,
                    session_id=active_session_id,
                    user_id=user.id,
                    dream_id=active_dream_id,
                    settings=settings,
                    openrouter_classifier_client=openrouter_classifier_client,
                )
                paused_kind = await entry_kind_for_dream(
                    session, dream_id=active_dream_id, user_id=user.id
                )
                await message.answer(entry_config(paused_kind).auto_paused_notice)
                log.info(
                    "new_entry.autoclose_quick",
                    quick_session_id=active_session_id,
                    user_id=user.id,
                )
        # DB-path пропускаем: иначе второе уведомление о паузе на ту же сессию.
        return

    active = await sess_repo.latest_in_progress_for_user(user_id=user.id)
    if active is not None:
        await _pause_session(
            session=session,
            session_id=active.id,
            user_id=user.id,
            dream_id=active.dream_id,
            settings=settings,
            openrouter_classifier_client=openrouter_classifier_client,
        )
        paused_kind = await entry_kind_for_dream(
            session, dream_id=active.dream_id, user_id=user.id
        )
        await message.answer(entry_config(paused_kind).auto_paused_notice)
        log.info(
            "new_entry.auto_paused_previous", paused_session_id=active.id, user_id=user.id
        )


async def start_new_entry(
    *,
    kind: str,
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    settings: Settings,
    bot: Bot | None,
    openrouter_classifier_client: AsyncOpenAI | None,
    user: User,
) -> bool:
    """Перевести юзера в рассказ нового обращения типа `kind`.

    Возвращает True, если юзер переведён в состояние рассказа; False — если сработал
    гард (пейвол / незакрытый онбординг): вызывающему остаётся только выйти.
    """
    telegram_user_id = message.chat.id
    # Повторный вход во время недосказанного рассказа — выкидываем черновик.
    await discard_pending_newdream(
        state=state, session=session, telegram_user_id=telegram_user_id
    )
    await discard_pending_entry(
        state=state, session=session, telegram_user_id=telegram_user_id
    )
    # Spend cap: над лимитом не пускаем в запись нового обращения.
    if await deny_entry(message, session, user, settings):
        return False

    repo = UserRepository(session)
    current_state = await state.get_state()
    if not UserRepository.is_onboarded(user):
        if current_state and current_state.startswith("QuickAnalysisStates"):
            # Онбординг-разбор уже сделан → завершаем онбординг и продолжаем ниже.
            await repo.mark_onboarded(user_id=user.id)
            await session.flush()
            if bot is not None:
                await sync_user_commands(
                    bot=bot, session=session, user=user, onboarded=True,
                    settings=settings,
                )
            log.info("new_entry.onboarding_completed_via_entry", user_id=user.id)
        else:
            # Ранний шаг знакомства — не сбрасываем прогресс советом /start.
            await message.answer(ONBOARDING_IN_PROGRESS_NOTICE)
            return False

    await _pause_active_session(
        message=message,
        state=state,
        session=session,
        user=user,
        settings=settings,
        openrouter_classifier_client=openrouter_classifier_client,
        current_state=current_state,
    )
    # Чистим любую stale FSM-data (например, newdream_dream_id из CONFIRMING_ANALYSIS,
    # которое юзер бросил вместо тапа кнопки).
    await state.clear()
    if kind == KIND_DREAM:
        await state.set_state(NewDreamStates.ASKING_DREAM)
        await message.answer(NEWDREAM_PROMPT, reply_markup=remove_keyboard())
    else:
        await state.set_state(EntryStates.ASKING_ENTRY)
        await state.update_data(**{FSM_KEY_ENTRY_KIND: kind})
        await message.answer(entry_config(kind).repeat_invite, reply_markup=remove_keyboard())
    return True
