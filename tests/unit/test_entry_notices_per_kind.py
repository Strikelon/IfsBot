"""Служебные уведомления не должны говорить «сон», когда обращение другого типа.

Дефект найден по жалобе на «Перевожу с языка снов…»
(см. test_thinking_statuses_per_kind). Самый вредный из оставшихся —
AUTO_PAUSED_NOTICE: он не просто говорит «сон», а отправляет в /dreams, где
обращения другого типа НЕТ (списки фильтруются по kind, dreams.py:116).
"""

import re
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._common import auto_pause_active_quick
from adapters.telegram.handlers.newdream import handle_newdream_command
from adapters.telegram.texts import (
    AUTO_PAUSED_NOTICE,
    DREAM_CARD_RESUME_NO_REPLY,
    DREAM_NOT_FOUND,
    IDLE_HINT,
    ONBOARDING_IN_PROGRESS_NOTICE,
    WELCOME_BACK,
)
from data.models import User
from domain.entry_types import ENTRY_TYPES, KIND_CONFLICT, KIND_DREAM, KIND_SITUATION

NEW_KINDS = (KIND_SITUATION, KIND_CONFLICT)
# «сон» во всех падежах + «сновидение». «Записанные» и «снова» не ловим.
DREAM_WORD = re.compile(r"\bсн[аоыуе]\b|\bсон\b|\bснов\b|сновид", re.IGNORECASE)


class _NoLimitSettings:
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    openrouter_context_model = "openai/gpt-4o-mini"


def _fsm(uid: int) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=uid, user_id=uid)
    )


def _message(uid: int) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Катя")
    m.chat = Chat(id=uid, type="private")
    m.text = "/newdream"
    m.answer = AsyncMock()
    return m


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", NEW_KINDS)
async def test_auto_pause_helper_reports_kind(test_session, make_quick_session, kind):
    """Хелпер обязан отдать тип запаузенного обращения — иначе текст не собрать."""
    sess = await make_quick_session(kind=kind)
    user = await test_session.get(User, sess.user_id)
    paused = await auto_pause_active_quick(
        session=test_session, state=_fsm(user.telegram_user_id), user_id=user.id,
    )
    assert paused is not None
    assert paused.session_id == sess.id
    assert paused.kind == kind


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", NEW_KINDS)
async def test_auto_paused_notice_points_at_right_list(
    test_session, make_quick_session, kind
):
    """/newdream поверх активной ситуации/конфликта: адрес возврата — свой список."""
    sess = await make_quick_session(kind=kind)
    user = await test_session.get(User, sess.user_id)
    msg = _message(user.telegram_user_id)
    await handle_newdream_command(
        msg, _fsm(user.telegram_user_id), test_session, _NoLimitSettings(),
        bot=None, openrouter_classifier_client=None,
    )
    notice = msg.answer.await_args_list[0].args[0]
    cfg = ENTRY_TYPES[kind]
    assert notice == cfg.auto_paused_notice
    assert f"/{cfg.list_command}" in notice
    assert "/dreams" not in notice
    assert not DREAM_WORD.search(notice)


@pytest.mark.asyncio
async def test_auto_paused_notice_for_dream_unchanged(test_session, make_quick_session):
    """Ветка сна — байт-в-байт как до правки."""
    sess = await make_quick_session()
    user = await test_session.get(User, sess.user_id)
    msg = _message(user.telegram_user_id)
    await handle_newdream_command(
        msg, _fsm(user.telegram_user_id), test_session, _NoLimitSettings(),
        bot=None, openrouter_classifier_client=None,
    )
    assert msg.answer.await_args_list[0].args[0] == AUTO_PAUSED_NOTICE


def test_shared_notices_are_kind_neutral():
    """Тексты одни на все типы — значит, слова «сон» в них быть не должно.

    ONBOARDING_IN_PROGRESS_NOTICE слышит человек, пришедший с конфликтом;
    DREAM_NOT_FOUND отвечает на /dream_<id>, общий для всех типов;
    DREAM_CARD_RESUME_NO_REPLY — fallback карточки любого типа.
    """
    for text in (
        ONBOARDING_IN_PROGRESS_NOTICE, DREAM_NOT_FOUND, DREAM_CARD_RESUME_NO_REPLY
    ):
        assert not DREAM_WORD.search(text), f"сновое слово в общем тексте: {text!r}"


def test_auto_paused_notices_per_kind_are_wired():
    """Каждый тип ведёт в СВОЙ список — гард от копипасты между конфигами."""
    for kind, cfg in ENTRY_TYPES.items():
        assert f"/{cfg.list_command}" in cfg.auto_paused_notice
        assert "—" not in cfg.auto_paused_notice
        if kind != KIND_DREAM:
            assert not DREAM_WORD.search(cfg.auto_paused_notice)


def test_welcome_back_offers_every_entry_type():
    """Приветствие вернувшегося – самый читаемый текст; звать только в сны нечестно.

    Сверяемся с IDLE_HINT: он перечисляет все шесть команд и служит эталоном
    полноты (появится четвёртый тип – оба текста должны поехать вместе).
    """
    for cfg in ENTRY_TYPES.values():
        assert f"/{cfg.new_command}" in WELCOME_BACK
        assert f"/{cfg.list_command}" in WELCOME_BACK
        assert f"/{cfg.new_command}" in IDLE_HINT
        assert f"/{cfg.list_command}" in IDLE_HINT
    assert "—" not in WELCOME_BACK
