"""Показ пейвола фиксируется: дата первого показа + событие на каждый показ."""

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select

from adapters.telegram.spend_guard import deny_entry, exit_with_paywall, show_paywall
from adapters.telegram.texts import LIMIT_REACHED
from data.analytics_events import AnalyticsEventType
from data.models import AnalyticsEvent, LlmService, UsageMetric, User
from data.repositories import UsageEventRepository


class _Settings:
    support_contact_url = "https://t.me/test_support"
    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0
    spend_limit_exempt_set = frozenset()


def _message(uid: int = 55) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Катя")
    m.chat = Chat(id=uid, type="private")
    m.answer = AsyncMock()
    return m


async def _over_limit_user(session, tg_id: int) -> User:
    user = User(telegram_user_id=tg_id, display_name="A")
    session.add(user)
    await session.flush()
    await UsageEventRepository(session).record(
        user_id=user.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("1.50"),
    )
    await session.flush()
    return user


async def _events(session, user_id: int) -> list[AnalyticsEvent]:
    rows = await session.execute(
        select(AnalyticsEvent).where(
            AnalyticsEvent.user_id == user_id,
            AnalyticsEvent.event_type == AnalyticsEventType.PAYWALL_SHOWN,
        )
    )
    return list(rows.scalars().all())


@pytest.mark.asyncio
async def test_deny_entry_marks_and_emits(test_session):
    user = await _over_limit_user(test_session, 55)
    msg = _message()
    assert await deny_entry(msg, test_session, user, _Settings()) is True
    assert user.paywall_shown_at is not None
    assert msg.answer.await_args.args[0] == LIMIT_REACHED.format(contact="https://t.me/test_support")
    assert len(await _events(test_session, user.id)) == 1


@pytest.mark.asyncio
async def test_first_show_date_is_not_overwritten(test_session):
    user = await _over_limit_user(test_session, 56)
    first = datetime(2026, 8, 1, tzinfo=UTC)
    user.paywall_shown_at = first
    await test_session.flush()
    await deny_entry(_message(56), test_session, user, _Settings())
    assert user.paywall_shown_at == first
    # событие пишется на КАЖДЫЙ показ, дата — только первая
    assert len(await _events(test_session, user.id)) == 1


@pytest.mark.asyncio
async def test_exit_with_paywall_marks_too(test_session):
    user = await _over_limit_user(test_session, 57)
    fsm = FSMContext(storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=57, user_id=57))
    assert await exit_with_paywall(
        _message(57), fsm, test_session, user, _Settings()
    ) is True
    assert user.paywall_shown_at is not None
    assert len(await _events(test_session, user.id)) == 1


@pytest.mark.asyncio
async def test_under_limit_marks_nothing(test_session):
    user = User(telegram_user_id=58, display_name="B")
    test_session.add(user)
    await test_session.flush()
    assert await deny_entry(_message(58), test_session, user, _Settings()) is False
    assert user.paywall_shown_at is None
    assert await _events(test_session, user.id) == []


def test_paywall_text_invites_to_interview():
    assert "2 месяца" in LIMIT_REACHED
    assert "{contact}" in LIMIT_REACHED
    assert "30–45 минут" in LIMIT_REACHED


@pytest.mark.asyncio
async def test_show_paywall_marks_and_sends(test_session):
    """Прямой показ (карточка сна, голосовое над лимитом) тоже учитывается."""
    user = await _over_limit_user(test_session, 59)
    msg = _message(59)
    await show_paywall(msg, test_session, user, _Settings())
    assert user.paywall_shown_at is not None
    assert msg.answer.await_args.args[0] == LIMIT_REACHED.format(contact="https://t.me/test_support")
    assert len(await _events(test_session, user.id)) == 1


def test_paywall_single_send_point():
    """LIMIT_REACHED уходит юзеру ТОЛЬКО через show_paywall.

    До 2026-08-24 четыре хендлера слали текст напрямую и не оставляли следа:
    колонка «Пейвол» в админке показывала прочерк людям, которые пейвол видели.
    Тест структурный, потому что дефект структурный — новый вызов
    `message.answer(LIMIT_REACHED)` в любом хендлере воспроизвёл бы его снова.
    """
    src = Path(__file__).resolve().parents[2] / "src"
    allowed = {"texts.py", "spend_guard.py"}
    offenders = [
        str(f.relative_to(src))
        for f in src.rglob("*.py")
        if f.name not in allowed and "LIMIT_REACHED" in f.read_text(encoding="utf-8")
    ]
    assert offenders == [], (
        "LIMIT_REACHED упоминается вне spend_guard.py: "
        f"{offenders}. Слать пейвол можно только через show_paywall()."
    )
