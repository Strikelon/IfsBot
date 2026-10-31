"""Оповещение юзера о выданном интервью-безлимите.

Одна точка на два вызывающих: кнопка «Интервью» в админке и команда /interview
в боте. До 2026-08-24 повышение тира было молчаливым — человек, упёршийся в
пейвол, уходил и не узнавал, что доступ открыли (см. разбор рекламной когорты).

Отправка никогда не роняет вызывающего: админка обязана применить смену тира
даже если юзер заблокировал бота.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.texts import INTERVIEW_GRANTED_NOTICE
from adapters.telegram.utils import format_date_ru
from data.repositories import UserRepository

log = structlog.get_logger(__name__)


async def notify_interview_granted(
    *,
    bot: Bot,
    session: AsyncSession,
    telegram_user_id: int,
    until: datetime | None,
) -> bool:
    """Написать юзеру, что доступ открыт. True — сообщение доставлено.

    `until is None` (безлимит без срока, ставится только руками через SQL) —
    оповещение не шлём: обещать «до такого-то числа» нечего.
    """
    if until is None:
        return False
    text = INTERVIEW_GRANTED_NOTICE.format(until=format_date_ru(until))
    try:
        await bot.send_message(chat_id=telegram_user_id, text=text)
    except TelegramForbiddenError:
        # Юзер заблокировал бота — помечаем его неактивным.
        repo = UserRepository(session)
        user = await repo.get_by_tg_id(telegram_user_id)
        if user is not None:
            await repo.set_blocked(
                user_id=user.id, blocked=True, now=datetime.now(UTC)
            )
        log.warning("interview.notice.blocked", telegram_user_id=telegram_user_id)
        return False
    except Exception as exc:  # noqa: BLE001 — смена тира важнее доставки письма
        log.warning(
            "interview.notice.failed",
            telegram_user_id=telegram_user_id,
            error=str(exc),
        )
        return False
    log.info("interview.notice.sent", telegram_user_id=telegram_user_id)
    return True
