"""Auto-sync of `User.display_name` from Telegram `first_name` on /start.

Covers spec §10.3 + §11.1-§11.3 edge cases. Uses `SimpleNamespace` as a
TG-user stand-in so we can exercise empty/None first_name and username
combinations that aiogram's pydantic model rejects at construction time.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message

from adapters.telegram.handlers.start import handle_start
from adapters.telegram.states import OnboardingStates
from data.repositories import UserRepository


@pytest.fixture
def fsm_context() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _make_message(
    *,
    user_id: int = 42,
    first_name: str | None = "Александр",
    username: str | None = None,
) -> Message:
    """Construct a Message whose `from_user` is a SimpleNamespace.

    SimpleNamespace lets us pass `first_name=None` (aiogram's pydantic
    TgUser would reject that at construction time) while still exposing
    the same attribute interface `_extract_display_name` reads.
    """
    message = AsyncMock(spec=Message)
    message.from_user = SimpleNamespace(
        id=user_id, is_bot=False, first_name=first_name, username=username
    )
    message.chat = Chat(id=user_id, type="private")
    message.text = "/start"
    message.answer = AsyncMock()
    return message


@pytest.mark.asyncio
async def test_start_new_user_extracts_first_name(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """first_name → display_name; FSM → CHOOSING_ENTRY_TYPE; menu→onboarded=False."""
    message = _make_message(first_name="Александр")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ) as mock_update_cmds:
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    repo = UserRepository(test_session)
    user = await repo.get_by_tg_id(42)
    assert user is not None
    assert user.display_name == "Александр"

    state = await fsm_context.get_state()
    # Спека B3: новый юзер попадает на экран выбора типа обращения.
    assert state == OnboardingStates.CHOOSING_ENTRY_TYPE.state

    mock_update_cmds.assert_awaited_once()
    kwargs = mock_update_cmds.await_args.kwargs
    assert kwargs["user"].telegram_user_id == 42
    assert kwargs["onboarded"] is False


@pytest.mark.asyncio
async def test_start_existing_user_first_name_changed_updates_db(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """If first_name in Telegram differs from DB — value is updated in-memory.

    `update_display_name_if_changed` mutates the ORM object on the same session;
    a subsequent read via `get_by_tg_id` (same session, same identity map) sees
    the new value. We don't flush here — handler tests run against an open
    session that the middleware would commit in prod.
    """
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Старое")
    await test_session.commit()

    message = _make_message(first_name="Новое")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ):
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    # Re-read via repo (same session) — identity-map cached object reflects mutation.
    reread = await repo.get_by_tg_id(42)
    assert reread is not None
    assert reread.display_name == "Новое"


@pytest.mark.asyncio
async def test_start_existing_user_same_first_name_no_op(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """If first_name == display_name → no DB UPDATE (handler-level smoke).

    Confirms the handler doesn't churn the row when nothing changed.
    update_display_name_if_changed itself returns False; we just verify
    the value stays the same.
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Александр")
    await test_session.commit()
    original_id = user.id

    message = _make_message(first_name="Александр")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ):
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    await test_session.refresh(user)
    assert user.display_name == "Александр"
    assert user.id == original_id


@pytest.mark.asyncio
async def test_start_no_first_name_falls_back_to_username(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """first_name пустой → fallback на username."""
    message = _make_message(first_name="", username="alex42")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ):
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    repo = UserRepository(test_session)
    user = await repo.get_by_tg_id(42)
    assert user is not None
    assert user.display_name == "alex42"


@pytest.mark.asyncio
async def test_start_no_first_name_no_username_falls_back_to_drug(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """first_name=None, username=None → fallback на 'друг'."""
    message = _make_message(first_name=None, username=None)
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ):
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    repo = UserRepository(test_session)
    user = await repo.get_by_tg_id(42)
    assert user is not None
    assert user.display_name == "друг"


@pytest.mark.asyncio
async def test_start_first_name_with_emoji_accepted_as_is(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """Эмодзи в first_name сохраняются как есть (без санитизации)."""
    message = _make_message(first_name="🦁 Лев")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ):
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    repo = UserRepository(test_session)
    user = await repo.get_by_tg_id(42)
    assert user is not None
    assert user.display_name == "🦁 Лев"


@pytest.mark.asyncio
async def test_start_onboarded_user_calls_sync_user_commands_with_onboarded_true(
    test_session, schema, fsm_context: FSMContext
) -> None:
    """Returning onboarded user → WELCOME_BACK + menu set with onboarded=True."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()

    message = _make_message(first_name="Катя")
    bot = MagicMock(spec=Bot)

    with patch(
        "adapters.telegram.handlers.start.sync_user_commands",
        new=AsyncMock(),
    ) as mock_update_cmds:
        await handle_start(message, fsm_context, session=test_session, bot=bot)

    # WELCOME_BACK sent
    message.answer.assert_awaited_once()
    sent = message.answer.await_args.args[0]
    assert "с возвращением" in sent.lower()
    assert "Катя" in sent

    # sync_user_commands called with onboarded=True
    mock_update_cmds.assert_awaited_once()
    kwargs = mock_update_cmds.await_args.kwargs
    assert kwargs["user"].telegram_user_id == 42
    assert kwargs["onboarded"] is True
