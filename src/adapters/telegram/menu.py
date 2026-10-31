"""Telegram bot commands menu — per-user scope.

Меню (кнопка «☰» рядом с полем ввода) появляется только после
завершения онбординга. До онбординга — пустой список (Telegram не
показывает кнопку).
"""
from __future__ import annotations

import structlog
from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BotCommand, BotCommandScopeChat
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import User

log = structlog.get_logger(__name__)

# Бампать при ЛЮБОМ изменении ONBOARDED_COMMANDS. Telegram хранит per-chat
# список вечно, а выставляется он лишь в паре точек (/start и конец онбординга),
# поэтому без версии старые юзеры не увидят новых команд никогда — ровно так и
# вышло с /new_external_situation и /new_internal_conflict (дефект от 2026-08-24).
COMMANDS_VERSION = 2

ONBOARDED_COMMANDS: list[BotCommand] = [
    BotCommand(command="start", description="Начать с начала"),
    BotCommand(command="newdream", description="Записать сон и разобрать"),
    BotCommand(
        command="new_external_situation", description="Разобрать волнующую ситуацию"
    ),
    BotCommand(
        command="new_internal_conflict", description="Разобрать внутренний конфликт"
    ),
    BotCommand(command="dreams", description="Мои сны"),
    BotCommand(command="external_situations", description="Мои волнующие ситуации"),
    BotCommand(command="internal_conflicts", description="Мои внутренние конфликты"),
    BotCommand(command="profile", description="Профиль и настройки"),
]

# Команда владельца. Меню выставляется per-chat, поэтому её видит только он —
# остальным её в списке просто нет (а хэндлер чужим и не отвечает).
ADMIN_EXTRA_COMMANDS: list[BotCommand] = [
    BotCommand(command="interview", description="Выдать человеку безлимит"),
]

EMPTY_COMMANDS: list[BotCommand] = []


def commands_for(*, onboarded: bool, is_admin: bool) -> list[BotCommand]:
    """Набор команд для конкретного человека."""
    if not onboarded:
        return EMPTY_COMMANDS
    if is_admin:
        return [*ONBOARDED_COMMANDS, *ADMIN_EXTRA_COMMANDS]
    return list(ONBOARDED_COMMANDS)


def _is_admin(user: User, settings) -> bool:
    admin_id = getattr(settings, "admin_id", None) if settings is not None else None
    return admin_id is not None and user.telegram_user_id == admin_id


async def update_user_commands(
    *, bot: Bot, telegram_user_id: int, onboarded: bool, is_admin: bool = False
) -> None:
    """Выставить меню команд для конкретного пользователя.

    Args:
        bot: aiogram Bot instance.
        telegram_user_id: Telegram user ID (= chat_id для private chat).
        onboarded: True → полный набор команд; False → пустой список (скрыто).
        is_admin: владельцу дополнительно показываем /interview.

    Failures (юзер заблокировал бота и т.п.) — логируем warning и не падаем.
    """
    commands = commands_for(onboarded=onboarded, is_admin=is_admin)
    try:
        await bot.set_my_commands(
            commands=commands,
            scope=BotCommandScopeChat(chat_id=telegram_user_id),
        )
    except TelegramBadRequest as exc:
        log.warning(
            "telegram.set_my_commands.failed",
            telegram_user_id=telegram_user_id,
            onboarded=onboarded,
            error=str(exc),
        )


async def sync_user_commands(
    *,
    bot: Bot,
    session: AsyncSession,
    user: User,
    onboarded: bool,
    settings=None,
) -> None:
    """Выставить меню и запомнить выставленную версию в БД.

    Версия — 0 для скрытого меню и COMMANDS_VERSION для полного набора: так
    юзер, вернувшийся в онбординг, получит полный список заново.
    """
    await update_user_commands(
        bot=bot,
        telegram_user_id=user.telegram_user_id,
        onboarded=onboarded,
        is_admin=_is_admin(user, settings),
    )
    version = COMMANDS_VERSION if onboarded else 0
    if user.menu_version == version:
        return
    user.menu_version = version
    # SAVEPOINT: зовётся из middleware, где исключения глотаются. Упавший flush
    # без вложенной транзакции отравил бы всю PG-транзакцию апдейта, и дальше
    # посыпался бы сам хендлер — ровно как было с daily_active.
    async with session.begin_nested():
        await session.flush()


async def ensure_menu_fresh(
    *, bot: Bot, session: AsyncSession, user: User, settings=None
) -> None:
    """Догнать меню до актуальной версии. Зовётся из middleware на каждом апдейте.

    Дешёвая проверка целого числа: вызов в Telegram уходит только когда версия
    действительно отстала, то есть один раз на юзера на бамп COMMANDS_VERSION.
    """
    if user.onboarding_completed_at is None:
        return
    if user.menu_version == COMMANDS_VERSION:
        return
    await sync_user_commands(
        bot=bot, session=session, user=user, onboarded=True, settings=settings
    )
    log.info(
        "telegram.menu.refreshed",
        telegram_user_id=user.telegram_user_id,
        version=COMMANDS_VERSION,
    )
