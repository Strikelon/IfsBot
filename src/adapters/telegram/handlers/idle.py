"""Idle catch-all: text/voice without command + no active session → IDLE_HINT.

Defensive: если на момент handler'а у юзера в памяти держится `qa_user_lock`
(значит Quick LLM-вызов в полёте), отвечаем THINKING_GUARD_NOTICE и НЕ
делаем state.clear() — иначе мы (a) сбросим валидный GENERATING_* state,
(b) упрёмся в SQLite lock из-за параллельной LLM-транзакции.

DB-based state read бесполезен здесь — параллельная Quick-сессия держит
open transaction с pending UPDATE, и idle через свою сессию видит старый
snapshot. In-memory lock `_qa_lock.is_locked()` — точный сигнал.
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import Message, ReplyKeyboardRemove
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers._qa_lock import (
    is_locked as qa_is_locked,
)
from adapters.telegram.handlers._qa_lock import (
    is_stt_locked,
)
from adapters.telegram.texts import IDLE_HINT, STT_GUARD_NOTICE, THINKING_GUARD_NOTICE
from data.repositories import SessionRepository, UserRepository

log = structlog.get_logger(__name__)


async def _user_has_active_session(
    *, session: AsyncSession, telegram_user_id: int
) -> bool:
    user_repo = UserRepository(session)
    user = await user_repo.get_by_tg_id(telegram_user_id)
    if user is None:
        return False
    sess_repo = SessionRepository(session)
    return await sess_repo.exists_in_progress_for_user(user_id=user.id)


async def _intercept_if_qa_locked(
    *, message: Message, telegram_user_id: int, kind: str
) -> bool:
    """Если у юзера активный stt/qa lock — guard notice (соответствующего тона) + True."""
    if is_stt_locked(telegram_user_id):
        log.info(
            "idle.routed_during_stt_lock",
            kind=kind,
            telegram_user_id=telegram_user_id,
        )
        await message.answer(STT_GUARD_NOTICE)
        return True
    if qa_is_locked(telegram_user_id):
        log.info(
            "idle.routed_during_qa_lock",
            kind=kind,
            telegram_user_id=telegram_user_id,
        )
        await message.answer(THINKING_GUARD_NOTICE)
        return True
    return False


async def handle_idle_text(
    message: Message, state: FSMContext, session: AsyncSession
) -> None:
    if message.from_user is None or message.text is None:
        return
    if await _intercept_if_qa_locked(
        message=message, telegram_user_id=message.from_user.id, kind="text"
    ):
        return
    if await _user_has_active_session(
        session=session, telegram_user_id=message.from_user.id
    ):
        # Active session in DB but no specific FSM-state handler matched — likely
        # stale/leaked FSM state. Clear state and fall through to IDLE_HINT as
        # safe default so the user gets a legible response.
        log.warning(
            "idle.text_unexpected_active_session",
            telegram_user_id=message.from_user.id,
        )
    try:
        await state.clear()
    except Exception as exc:  # noqa: BLE001
        # SQLite может быть locked параллельной LLM-транзакцией — это не блокер,
        # IDLE_HINT всё равно шлём. Юзер хотя бы получит понятный ответ.
        log.warning(
            "idle.state_clear_failed",
            telegram_user_id=message.from_user.id,
            exc_type=type(exc).__name__,
            exc=str(exc),
        )
    await message.answer(IDLE_HINT, reply_markup=ReplyKeyboardRemove())
    log.info("idle.text_hint_sent", telegram_user_id=message.from_user.id)


async def handle_idle_voice(
    message: Message, state: FSMContext, session: AsyncSession
) -> None:
    if message.from_user is None or message.voice is None:
        return
    if await _intercept_if_qa_locked(
        message=message, telegram_user_id=message.from_user.id, kind="voice"
    ):
        return
    # Не транскрибируем — экономим STT.
    if await _user_has_active_session(
        session=session, telegram_user_id=message.from_user.id
    ):
        log.warning(
            "idle.voice_unexpected_active_session",
            telegram_user_id=message.from_user.id,
        )
    try:
        await state.clear()
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "idle.state_clear_failed",
            telegram_user_id=message.from_user.id,
            exc_type=type(exc).__name__,
            exc=str(exc),
        )
    await message.answer(IDLE_HINT, reply_markup=ReplyKeyboardRemove())
    log.info("idle.voice_hint_sent", telegram_user_id=message.from_user.id)


def create_router() -> Router:
    router = Router(name="idle")
    router.message.register(
        handle_idle_text,
        F.text & ~F.text.startswith("/"),
    )
    router.message.register(handle_idle_voice, F.voice)
    return router
