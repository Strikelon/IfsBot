from datetime import UTC, datetime

import pytest
from aiogram.types import Chat, ChatMemberBanned, ChatMemberMember, ChatMemberUpdated
from aiogram.types import User as TgUser

from adapters.telegram.handlers.chat_member import handle_my_chat_member
from data.repositories import UserRepository


def _update(tg_id: int, new_status_member: bool) -> ChatMemberUpdated:
    tg = TgUser(id=tg_id, is_bot=False, first_name="A")
    new = (
        ChatMemberMember(user=tg) if new_status_member else ChatMemberBanned(user=tg, until_date=datetime(2099, 1, 1, tzinfo=UTC))
    )
    old = ChatMemberMember(user=tg)
    return ChatMemberUpdated.model_construct(
        chat=Chat(id=tg_id, type="private"),
        from_user=tg,
        date=datetime(2026, 6, 8, tzinfo=UTC),
        old_chat_member=old,
        new_chat_member=new,
    )


@pytest.mark.asyncio
async def test_block_marks_inactive(test_session):
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=11, display_name="A")
    await handle_my_chat_member(_update(11, new_status_member=False), test_session)
    user = await repo.get_by_tg_id(11)
    assert user.is_active is False
    assert user.blocked_at is not None


@pytest.mark.asyncio
async def test_unblock_marks_active(test_session):
    repo = UserRepository(test_session)
    u = await repo.get_or_create_by_tg_id(telegram_user_id=12, display_name="A")
    await repo.set_blocked(user_id=u.id, blocked=True, now=datetime(2026, 6, 7, tzinfo=UTC))
    await handle_my_chat_member(_update(12, new_status_member=True), test_session)
    user = await repo.get_by_tg_id(12)
    assert user.is_active is True
