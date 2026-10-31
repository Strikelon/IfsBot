"""/start — точка входа. Auto-sync display_name + онбординг или приветствие возврата."""

from __future__ import annotations

from html import escape as html_escape

from aiogram import Bot, Router
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.types import Message
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers._common import (
    discard_pending_entry,
    discard_pending_newdream,
)
from adapters.telegram.keyboards import entry_type_keyboard, remove_keyboard
from adapters.telegram.menu import sync_user_commands
from adapters.telegram.states import OnboardingStates
from adapters.telegram.texts import (
    ENTRY_CHOICE_INTRO,
    WELCOME_BACK,
)
from app.config import Settings
from data.analytics_events import AnalyticsEventType, emit_funnel_step
from data.models import DISPLAY_NAME_MAX_LEN
from data.repositories import UserRepository

_FALLBACK_NAME = "друг"


def _extract_display_name(tg_user) -> str:
    """Извлечь display_name из Telegram user (для auto-sync).

    Fallback chain: first_name → username → "друг". Truncate to column length.
    """
    raw = (tg_user.first_name or tg_user.username or _FALLBACK_NAME).strip()
    return (raw[:DISPLAY_NAME_MAX_LEN] if raw else _FALLBACK_NAME)


async def handle_start(
    message: Message,
    state: FSMContext,
    session: AsyncSession,
    bot: Bot,
    settings: Settings | None = None,
) -> None:
    if message.from_user is None:
        return  # системные апдейты — игнор

    # Если юзер был в процессе рассказа сна/обращения — выкидываем недосказанное.
    await discard_pending_newdream(
        state=state, session=session, telegram_user_id=message.from_user.id
    )
    await discard_pending_entry(
        state=state, session=session, telegram_user_id=message.from_user.id
    )

    tg_user = message.from_user
    display_name = _extract_display_name(tg_user)

    user_repo = UserRepository(session)
    user = await user_repo.get_or_create_by_tg_id(
        telegram_user_id=tg_user.id, display_name=display_name
    )
    # Auto-sync: если first_name в Telegram сменился — обновляем БД.
    # No flush() needed: ORM mutation is visible in the identity map within the same session.
    await user_repo.update_display_name_if_changed(
        user_id=user.id, display_name=display_name
    )

    if UserRepository.is_onboarded(user):
        await state.clear()
        await sync_user_commands(
            bot=bot, session=session, user=user, onboarded=True, settings=settings
        )
        # remove_keyboard: если юзер дал /start посреди Q&A — убираем «Завершить».
        await message.answer(
            WELCOME_BACK.format(name=html_escape(user.display_name)),
            reply_markup=remove_keyboard(),
        )
        return

    # Новый юзер — экран выбора типа обращения (спека B3). Прод-юзеры, уже
    # стоящие в ASKING_DREAM, сюда не попадают — /start для них либо WELCOME_BACK
    # (onboarded), либо новый выбор (не onboarded, что законно: рассказ они ещё
    # не начали, дефект «/start сбрасывает» не добавляется — state и так менялся).
    await sync_user_commands(
        bot=bot, session=session, user=user, onboarded=False, settings=settings
    )
    await state.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    await emit_funnel_step(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.FIRST_LAUNCH,
    )
    sent = await message.answer(
        ENTRY_CHOICE_INTRO.format(name=html_escape(user.display_name)),
        reply_markup=entry_type_keyboard(),
    )
    msg_id = getattr(sent, "message_id", None)
    if isinstance(msg_id, int):
        await state.update_data(entry_choice_msg_id=msg_id)


def create_router() -> Router:
    """Создать новый экземпляр Router с зарегистрированными хэндлерами."""
    router = Router(name="start_handler")
    router.message.register(handle_start, CommandStart())
    return router
