from datetime import UTC, datetime

import pytest
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.middlewares import UserActivityMiddleware
from data.repositories import UserRepository


def _make_message(tg_id: int, username: str | None) -> Message:
    return Message.model_construct(
        message_id=1,
        date=datetime(2026, 6, 8, tzinfo=UTC),
        chat=Chat(id=tg_id, type="private"),
        from_user=TgUser(id=tg_id, is_bot=False, first_name="A", username=username),
    )


@pytest.mark.asyncio
async def test_middleware_touches_activity(test_session):
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="A")
    await test_session.commit()

    mw = UserActivityMiddleware()
    called = {}

    async def handler(event, data):
        called["ok"] = True
        return "done"

    event = _make_message(42, "ann")
    result = await mw(handler, event, {"session": test_session})
    assert result == "done"
    user = await repo.get_by_tg_id(42)
    assert user.username == "ann"
    assert user.last_seen_at is not None


@pytest.mark.asyncio
async def test_middleware_passes_through_without_session(test_session):
    mw = UserActivityMiddleware()

    async def handler(event, data):
        return "ok"

    # no session in data -> must not crash, just pass through
    result = await mw(handler, _make_message(99, "x"), {})
    assert result == "ok"
