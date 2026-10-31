from __future__ import annotations

from datetime import UTC, datetime

from aiogram import Router
from aiogram.types import ChatMemberUpdated
from sqlalchemy.ext.asyncio import AsyncSession

from data.repositories import UserRepository

_BLOCKED_STATUSES = {"kicked", "banned", "left"}


async def handle_my_chat_member(event: ChatMemberUpdated, session: AsyncSession) -> None:
    tg_user = event.from_user
    if tg_user is None or tg_user.is_bot:
        return
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(
        telegram_user_id=tg_user.id, display_name=tg_user.first_name
    )
    new_status = event.new_chat_member.status
    blocked = new_status in _BLOCKED_STATUSES
    await repo.set_blocked(user_id=user.id, blocked=blocked, now=datetime.now(UTC))


def create_router() -> Router:
    router = Router(name="chat_member")
    router.my_chat_member.register(handle_my_chat_member)
    return router
