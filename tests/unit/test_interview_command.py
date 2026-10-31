"""Команда /interview и оповещение о выданном безлимите (2026-08-24)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiogram.exceptions import TelegramForbiddenError
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.interview import (
    GONE_TEXT,
    NOT_FOUND_TMPL,
    USAGE_TEXT,
    handle_interview_callback,
    handle_interview_command,
)
from adapters.telegram.interview_notice import notify_interview_granted
from adapters.telegram.texts import INTERVIEW_GRANTED_NOTICE
from data.models import User
from domain.access import TIER_EARLY_TESTER, TIER_FREE_INTERVIEW, TIER_NEW

ADMIN_ID = 777


class _Settings:
    admin_id = ADMIN_ID


def _message(uid: int = ADMIN_ID) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Админ")
    m.chat = Chat(id=uid, type="private")
    m.answer = AsyncMock()
    return m


def _command(args: str | None):
    obj = MagicMock()
    obj.args = args
    return obj


def _callback(data: str, uid: int = ADMIN_ID):
    cb = MagicMock()
    cb.from_user = TgUser(id=uid, is_bot=False, first_name="Админ")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.answer = AsyncMock()
    cb.message.edit_reply_markup = AsyncMock()
    cb.answer = AsyncMock()
    return cb


async def _user(session, tg_id: int, *, name: str, username: str | None = None) -> User:
    u = User(telegram_user_id=tg_id, display_name=name, username=username)
    session.add(u)
    await session.flush()
    return u


# --- доступ ---


@pytest.mark.asyncio
async def test_command_is_silent_for_non_admin(test_session):
    """Чужому команда не отвечает вовсе — она не должна светиться."""
    msg = _message(uid=12345)
    await handle_interview_command(msg, _command("Мария"), test_session, _Settings())
    msg.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_command_is_silent_when_admin_id_not_configured(test_session):
    """Без ADMIN_ID в конфиге команда мертва — а не открыта всем."""

    class _NoAdmin:
        admin_id = None

    msg = _message()
    await handle_interview_command(msg, _command("Мария"), test_session, _NoAdmin())
    msg.answer.assert_not_awaited()


@pytest.mark.asyncio
async def test_callback_from_non_admin_changes_nothing(test_session):
    target = await _user(test_session, 900, name="Жертва")
    cb = _callback("iv:900", uid=12345)
    await handle_interview_callback(cb, test_session, _Settings(), bot=None)
    assert target.access_tier == TIER_NEW
    cb.message.answer.assert_not_awaited()


# --- поиск ---


@pytest.mark.asyncio
async def test_no_args_shows_usage(test_session):
    msg = _message()
    await handle_interview_command(msg, _command(None), test_session, _Settings())
    assert msg.answer.await_args.args[0] == USAGE_TEXT


@pytest.mark.asyncio
async def test_search_finds_user_without_username_by_name(test_session):
    """Главный сценарий: у Марии ника нет, ищем по имени."""
    await _user(test_session, 700000001, name="Мария", username=None)
    msg = _message()
    await handle_interview_command(msg, _command("мария"), test_session, _Settings())
    kwargs = msg.answer.await_args.kwargs
    buttons = kwargs["reply_markup"].inline_keyboard
    assert len(buttons) == 1
    assert buttons[0][0].callback_data == "iv:700000001"


@pytest.mark.asyncio
async def test_search_finds_by_telegram_id(test_session):
    await _user(test_session, 700000001, name="Мария")
    msg = _message()
    await handle_interview_command(
        msg, _command("700000001"), test_session, _Settings()
    )
    buttons = msg.answer.await_args.kwargs["reply_markup"].inline_keyboard
    assert buttons[0][0].callback_data == "iv:700000001"


@pytest.mark.asyncio
async def test_search_finds_by_username_with_at(test_session):
    await _user(test_session, 42, name="Алина", username="AlinaMax")
    msg = _message()
    await handle_interview_command(msg, _command("@alinamax"), test_session, _Settings())
    buttons = msg.answer.await_args.kwargs["reply_markup"].inline_keyboard
    assert buttons[0][0].callback_data == "iv:42"


@pytest.mark.asyncio
async def test_search_reports_nothing_found(test_session):
    msg = _message()
    await handle_interview_command(msg, _command("Хтоня"), test_session, _Settings())
    assert msg.answer.await_args.args[0] == NOT_FOUND_TMPL.format(query="Хтоня")
    assert msg.answer.await_args.kwargs.get("reply_markup") is None


# --- выдача ---


@pytest.mark.asyncio
async def test_grant_sets_tier_and_notifies(test_session):
    target = await _user(test_session, 901, name="Мария")
    bot = MagicMock()
    bot.send_message = AsyncMock()
    cb = _callback("iv:901")

    await handle_interview_callback(cb, test_session, _Settings(), bot=bot)

    assert target.access_tier == TIER_FREE_INTERVIEW
    assert target.access_tier_until is not None
    assert target.interview_passed_at is not None
    bot.send_message.assert_awaited_once()
    assert bot.send_message.await_args.kwargs["chat_id"] == 901
    assert "/new_internal_conflict" in bot.send_message.await_args.kwargs["text"]


@pytest.mark.asyncio
async def test_grant_remembers_previous_tier(test_session):
    """Первый тестер после снятия интервью должен вернуться первым тестером."""
    target = await _user(test_session, 902, name="Ксения")
    target.access_tier = TIER_EARLY_TESTER
    await test_session.flush()

    bot = MagicMock()
    bot.send_message = AsyncMock()
    await handle_interview_callback(_callback("iv:902"), test_session, _Settings(), bot=bot)
    assert target.tier_before_interview == TIER_EARLY_TESTER


@pytest.mark.asyncio
async def test_grant_reports_undelivered_notice(test_session):
    """Заблокировавший бота получает тир, но админ видит предупреждение."""
    target = await _user(test_session, 903, name="Молчун")
    bot = MagicMock()
    bot.send_message = AsyncMock(
        side_effect=TelegramForbiddenError(method=MagicMock(), message="blocked")
    )
    cb = _callback("iv:903")

    await handle_interview_callback(cb, test_session, _Settings(), bot=bot)

    assert target.access_tier == TIER_FREE_INTERVIEW
    assert target.is_active is False
    assert "НЕ доставлено" in cb.message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_grant_on_deleted_user_says_so(test_session):
    cb = _callback("iv:999999")
    await handle_interview_callback(cb, test_session, _Settings(), bot=None)
    assert cb.message.answer.await_args.args[0] == GONE_TEXT


# --- текст оповещения ---


@pytest.mark.asyncio
async def test_notice_carries_date_and_menu(test_session):
    await _user(test_session, 904, name="Кто-то")
    bot = MagicMock()
    bot.send_message = AsyncMock()
    ok = await notify_interview_granted(
        bot=bot,
        session=test_session,
        telegram_user_id=904,
        until=datetime(2026, 10, 24, tzinfo=UTC),
    )
    assert ok is True
    text = bot.send_message.await_args.kwargs["text"]
    assert "24 октября" in text
    assert "сохранено" in text
    assert "/dreams" in text and "/newdream" in text


@pytest.mark.asyncio
async def test_notice_skipped_without_deadline(test_session):
    """Безлимит без срока (ручной SQL) — обещать «до числа» нечего, молчим."""
    bot = MagicMock()
    bot.send_message = AsyncMock()
    ok = await notify_interview_granted(
        bot=bot, session=test_session, telegram_user_id=905, until=None
    )
    assert ok is False
    bot.send_message.assert_not_awaited()


def test_notice_promises_the_same_two_months_as_the_paywall():
    """Обещание в пейволе и в оповещении обязаны совпадать."""
    from adapters.telegram.texts import LIMIT_REACHED

    assert "2 месяца" in LIMIT_REACHED
    assert "2 месяца" in INTERVIEW_GRANTED_NOTICE
