"""Команда /interview — выдать человеку интервью-безлимит прямо из бота.

Зачем при существующей админке: админка живёт за SSH-туннелем и с телефона
недоступна. Нужно уметь открыть доступ «даже из отпуска, когда есть
только телефон» (2026-08-24).

Доступна ТОЛЬКО владельцу (`ADMIN_ID`). Для всех остальных команды не существует:
на чужой вызов бот молчит, а не отвечает «нельзя» — так она не светится.

Поиск идёт по имени, нику и telegram id сразу: у части юзеров ника нет вовсе,
и как раз им чаще всего и нужен ручной доступ. Найденные показываются кнопками,
выдача — одним тапом.
"""

from __future__ import annotations

from datetime import UTC, datetime

import structlog
from aiogram import Bot, F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.interview_notice import notify_interview_granted
from adapters.telegram.utils import format_date_ru
from app.config import Settings
from data.models import User
from data.repositories import UserRepository
from domain.access import apply_tier_change

log = structlog.get_logger(__name__)

INTERVIEW_CALLBACK_PREFIX = "iv:"

USAGE_TEXT = (
    "Кому выдать безлимит? Пришлите имя, ник или id:\n"
    "<code>/interview Мария</code>"
)
NOT_FOUND_TMPL = "Никого не нашёл по запросу «{query}»."
PICK_TMPL = "Нашёл {count}. Кому выдать безлимит на 2 месяца?"
GRANTED_TMPL = (
    "Готово: {who} – безлимит до {until}.\n"
    "Было: {before}.\n"
    "{delivery}"
)
DELIVERY_OK = "Оповещение отправлено."
DELIVERY_FAILED = "⚠️ Оповещение НЕ доставлено (бот заблокирован или недоступен)."
GONE_TEXT = "Этого пользователя больше нет в базе."


def _is_admin(telegram_user_id: int | None, settings: Settings) -> bool:
    return (
        settings.admin_id is not None
        and telegram_user_id is not None
        and telegram_user_id == settings.admin_id
    )


def _label(user: User) -> str:
    """Строка на кнопке: имя, ник и id — чтобы различить тёзок."""
    parts = [user.display_name or "без имени"]
    if user.username:
        parts.append(f"@{user.username}")
    parts.append(str(user.telegram_user_id))
    return " · ".join(parts)


def _keyboard(users: list[User]) -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(
                    text=_label(u),
                    callback_data=f"{INTERVIEW_CALLBACK_PREFIX}{u.telegram_user_id}",
                )
            ]
            for u in users
        ]
    )


async def handle_interview_command(
    message: Message,
    command: CommandObject,
    session: AsyncSession,
    settings: Settings,
) -> None:
    """/interview <имя|ник|id> — показать кандидатов кнопками."""
    if not _is_admin(message.from_user.id if message.from_user else None, settings):
        return  # команды для этого юзера не существует
    query = (command.args or "").strip()
    if not query:
        await message.answer(USAGE_TEXT)
        return
    found = await UserRepository(session).search(query)
    if not found:
        await message.answer(NOT_FOUND_TMPL.format(query=query))
        return
    await message.answer(
        PICK_TMPL.format(count=len(found)), reply_markup=_keyboard(found)
    )
    log.info("interview.search", query=query, found=len(found))


async def handle_interview_callback(
    callback: CallbackQuery,
    session: AsyncSession,
    settings: Settings,
    bot: Bot | None = None,
) -> None:
    """Тап по кандидату: выдать безлимит и оповестить его."""
    if not _is_admin(callback.from_user.id if callback.from_user else None, settings):
        await safe_answer(callback)
        return
    if callback.data is None or callback.message is None:
        return
    await safe_answer(callback)
    try:
        target_id = int(callback.data.removeprefix(INTERVIEW_CALLBACK_PREFIX))
    except ValueError:
        return

    repo = UserRepository(session)
    target = await repo.get_by_tg_id(target_id)
    if target is None:
        await callback.message.answer(GONE_TEXT)
        return

    before = target.access_tier
    apply_tier_change(target, "interview", now=datetime.now(UTC))
    until = target.access_tier_until
    # Коммит ДО отправки: выданный доступ не должен зависеть от того, дошло ли
    # письмо. Иначе обрыв сети на следующей строке откатил бы тир человеку,
    # которому бот уже пообещал безлимит.
    await session.commit()

    # Клавиатуру гасим, чтобы повторный тап по тому же списку не уходил вслепую.
    try:
        await callback.message.edit_reply_markup(reply_markup=None)
    except Exception:  # noqa: BLE001 — сообщение могли удалить
        log.debug("interview.markup_not_extinguished", target_id=target_id)

    delivered = False
    if bot is not None:
        delivered = await notify_interview_granted(
            bot=bot, session=session, telegram_user_id=target_id, until=until
        )

    await callback.message.answer(
        GRANTED_TMPL.format(
            who=_label(target),
            until=format_date_ru(until) if until else "без срока",
            before=before,
            delivery=DELIVERY_OK if delivered else DELIVERY_FAILED,
        )
    )
    log.info(
        "interview.granted",
        target_id=target_id,
        tier_from=before,
        until=until.isoformat() if until else None,
        notified=delivered,
    )


def create_router() -> Router:
    """Роутер команды владельца. Включается сразу после thinking_guard: команда
    обязана работать в любом состоянии FSM."""
    router = Router(name="interview")
    router.message.register(handle_interview_command, Command("interview"))
    router.callback_query.register(
        handle_interview_callback, F.data.startswith(INTERVIEW_CALLBACK_PREFIX)
    )
    return router
