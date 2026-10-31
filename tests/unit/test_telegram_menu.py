from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramBadRequest
from aiogram.types import BotCommandScopeChat

from adapters.telegram.menu import (
    COMMANDS_VERSION,
    EMPTY_COMMANDS,
    ONBOARDED_COMMANDS,
    commands_for,
    ensure_menu_fresh,
    sync_user_commands,
    update_user_commands,
)
from data.models import User


@pytest.mark.asyncio
async def test_update_user_commands_onboarded_true_sends_full_list() -> None:
    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await update_user_commands(bot=bot, telegram_user_id=42, onboarded=True)
    bot.set_my_commands.assert_awaited_once()
    kwargs = bot.set_my_commands.await_args.kwargs
    assert kwargs["commands"] == ONBOARDED_COMMANDS
    assert isinstance(kwargs["scope"], BotCommandScopeChat)
    assert kwargs["scope"].chat_id == 42


@pytest.mark.asyncio
async def test_update_user_commands_onboarded_false_sends_empty_list() -> None:
    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await update_user_commands(bot=bot, telegram_user_id=99, onboarded=False)
    bot.set_my_commands.assert_awaited_once()
    kwargs = bot.set_my_commands.await_args.kwargs
    assert kwargs["commands"] == EMPTY_COMMANDS


@pytest.mark.asyncio
async def test_update_user_commands_swallows_telegram_bad_request() -> None:
    """TelegramBadRequest (юзер заблокировал бота и т.п.) — не должны падать."""
    bot = MagicMock()
    # TelegramBadRequest requires `method` (Method instance) and `message` — pass minimum.
    bot.set_my_commands = AsyncMock(
        side_effect=TelegramBadRequest(method=MagicMock(), message="user not found")
    )
    # Не должно бросать
    await update_user_commands(bot=bot, telegram_user_id=42, onboarded=True)
    bot.set_my_commands.assert_awaited_once()


def test_onboarded_commands_cover_all_entry_types() -> None:
    """Вход И список на каждый из трёх типов, плюс старт и профиль.

    Списки добавлены 2026-08-24: без них человек видел
    в меню только «записать новое» и не знал, как вернуться к разобранному.
    """
    names = [c.command for c in ONBOARDED_COMMANDS]
    assert names == [
        "start",
        "newdream",
        "new_external_situation",
        "new_internal_conflict",
        "dreams",
        "external_situations",
        "internal_conflicts",
        "profile",
    ]


def test_every_entry_type_has_both_commands_in_menu() -> None:
    """Ни один тип не должен остаться без входа или без списка."""
    from domain.entry_types import ENTRY_TYPES

    names = {c.command for c in ONBOARDED_COMMANDS}
    for cfg in ENTRY_TYPES.values():
        assert cfg.new_command.lstrip("/") in names
        assert cfg.list_command.lstrip("/") in names


def test_interview_shown_to_owner_only() -> None:
    """/interview — в меню только у владельца, у остальных её нет."""
    plain = [c.command for c in commands_for(onboarded=True, is_admin=False)]
    owner = [c.command for c in commands_for(onboarded=True, is_admin=True)]
    assert "interview" not in plain
    assert owner == [*plain, "interview"]


def test_hidden_menu_ignores_admin_flag() -> None:
    """До конца онбординга меню пустое даже у владельца."""
    assert commands_for(onboarded=False, is_admin=True) == []


def test_menu_command_names_match_handlers() -> None:
    """Имена в меню совпадают с зарегистрированными командами (иначе меню шлёт в пустоту)."""
    from domain.entry_types import ENTRY_TYPES

    names = {c.command for c in ONBOARDED_COMMANDS}
    for cfg in ENTRY_TYPES.values():
        assert cfg.new_command.lstrip("/") in names


def test_empty_commands_is_empty_list() -> None:
    assert EMPTY_COMMANDS == []


# --- Версионирование меню (дефект 2026-08-24) ---


@pytest.mark.asyncio
async def test_ensure_menu_fresh_updates_stale_onboarded_user(test_session) -> None:
    """Юзер, прошедший онбординг до появления новых команд, догоняет меню сам."""
    user = User(
        telegram_user_id=1001,
        display_name="Старый",
        onboarding_completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        menu_version=0,
    )
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await ensure_menu_fresh(bot=bot, session=test_session, user=user)

    bot.set_my_commands.assert_awaited_once()
    assert bot.set_my_commands.await_args.kwargs["commands"] == ONBOARDED_COMMANDS
    assert user.menu_version == COMMANDS_VERSION


@pytest.mark.asyncio
async def test_ensure_menu_fresh_is_silent_when_version_matches(test_session) -> None:
    """Актуальная версия — ни одного вызова в Telegram (проверка это целое число)."""
    user = User(
        telegram_user_id=1002,
        display_name="Свежий",
        onboarding_completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        menu_version=COMMANDS_VERSION,
    )
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await ensure_menu_fresh(bot=bot, session=test_session, user=user)
    bot.set_my_commands.assert_not_awaited()


@pytest.mark.asyncio
async def test_ensure_menu_fresh_skips_not_onboarded(test_session) -> None:
    """До конца онбординга меню скрыто — не показываем его досрочно."""
    user = User(telegram_user_id=1003, display_name="Новый", menu_version=0)
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await ensure_menu_fresh(bot=bot, session=test_session, user=user)
    bot.set_my_commands.assert_not_awaited()
    assert user.menu_version == 0


@pytest.mark.asyncio
async def test_sync_user_commands_resets_version_when_menu_hidden(test_session) -> None:
    """Скрытое меню — версия 0: вернувшийся в онбординг получит полный список заново."""
    user = User(
        telegram_user_id=1004, display_name="Сброс", menu_version=COMMANDS_VERSION
    )
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await sync_user_commands(bot=bot, session=test_session, user=user, onboarded=False)
    assert user.menu_version == 0


def test_commands_version_is_bumped_when_list_changes() -> None:
    """Напоминание: меняя ONBOARDED_COMMANDS, бампни COMMANDS_VERSION.

    Слепок длины и имён — если список поехал, а версия нет, тест упадёт и
    заставит принять решение осознанно.
    """
    names = tuple(c.command for c in ONBOARDED_COMMANDS)
    snapshots = {
        1: (
            "start",
            "newdream",
            "new_external_situation",
            "new_internal_conflict",
            "dreams",
            "profile",
        ),
        2: (
            "start",
            "newdream",
            "new_external_situation",
            "new_internal_conflict",
            "dreams",
            "external_situations",
            "internal_conflicts",
            "profile",
        ),
    }
    assert COMMANDS_VERSION in snapshots, (
        f"COMMANDS_VERSION={COMMANDS_VERSION} без слепка — допиши его в snapshots."
    )
    assert names == snapshots[COMMANDS_VERSION], (
        "ONBOARDED_COMMANDS изменился — бампни COMMANDS_VERSION и добавь слепок, "
        "иначе старые юзеры останутся со старым меню навсегда."
    )


@pytest.mark.asyncio
async def test_sync_gives_owner_the_interview_command(test_session) -> None:
    """Владелец получает свой набор — по admin_id из конфига."""

    class _Settings:
        admin_id = 777

    user = User(
        telegram_user_id=777,
        display_name="Владелец",
        onboarding_completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        menu_version=0,
    )
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await ensure_menu_fresh(
        bot=bot, session=test_session, user=user, settings=_Settings()
    )
    sent = [c.command for c in bot.set_my_commands.await_args.kwargs["commands"]]
    assert sent[-1] == "interview"


@pytest.mark.asyncio
async def test_sync_without_settings_never_leaks_admin_command(test_session) -> None:
    """Без конфига (старые вызовы) команда владельца не протекает."""
    user = User(
        telegram_user_id=888,
        display_name="Обычный",
        onboarding_completed_at=datetime(2026, 8, 1, tzinfo=UTC),
        menu_version=0,
    )
    test_session.add(user)
    await test_session.flush()

    bot = MagicMock()
    bot.set_my_commands = AsyncMock()
    await ensure_menu_fresh(bot=bot, session=test_session, user=user)
    sent = [c.command for c in bot.set_my_commands.await_args.kwargs["commands"]]
    assert "interview" not in sent
