"""thinking_status context manager: на входе ставит thinking_state, на выходе — prev."""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.states import QuickAnalysisStates
from adapters.telegram.thinking_status import thinking_status


@pytest_asyncio.fixture
async def fsm_context() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )


def _make_message() -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=42, is_bot=False, first_name="T")
    msg.chat = Chat(id=42, type="private")
    msg.text = "dummy"
    msg.answer = AsyncMock(return_value=AsyncMock())  # placeholder reply
    return msg


@pytest.mark.asyncio
async def test_thinking_status_uses_custom_statuses() -> None:
    """Если передан кастомный список статусов — плейсхолдер берётся из него."""
    msg = _make_message()
    custom = ["Знакомлюсь с вами поближе…", "Беру на заметку важное…"]
    async with thinking_status(msg, statuses=custom):
        pass
    sent = msg.answer.await_args.args[0]
    assert sent in custom


@pytest.mark.asyncio
async def test_thinking_status_defaults_to_dream_statuses() -> None:
    """Без statuses — общий (dream-themed) набор THINKING_STATUSES."""
    from adapters.telegram.texts import THINKING_STATUSES

    msg = _make_message()
    async with thinking_status(msg):
        pass
    assert msg.answer.await_args.args[0] in THINKING_STATUSES


@pytest.mark.asyncio
async def test_thinking_status_sets_thinking_state_on_enter_and_restores(
    fsm_context: FSMContext,
):
    await fsm_context.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    msg = _make_message()
    captured = {}

    async with thinking_status(
        msg, state=fsm_context, thinking_state=QuickAnalysisStates.GENERATING_INITIAL
    ):
        captured["inside"] = await fsm_context.get_state()

    assert captured["inside"] == QuickAnalysisStates.GENERATING_INITIAL.state
    assert (await fsm_context.get_state()) == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state


@pytest.mark.asyncio
async def test_thinking_status_restores_state_even_on_exception(
    fsm_context: FSMContext,
):
    await fsm_context.set_state(QuickAnalysisStates.IN_QA)
    msg = _make_message()

    with pytest.raises(RuntimeError):
        async with thinking_status(
            msg, state=fsm_context, thinking_state=QuickAnalysisStates.GENERATING_FOLLOWUP
        ):
            raise RuntimeError("LLM upstream timeout")

    assert (await fsm_context.get_state()) == QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_thinking_status_backwards_compatible_without_state_args(
    fsm_context: FSMContext,
):
    """Если state/thinking_state не передан — поведение как было: ничего с FSM не делаем."""
    msg = _make_message()
    async with thinking_status(msg):
        pass  # просто не должно падать


@pytest.mark.asyncio
async def test_thinking_status_noop_if_already_in_thinking_state(
    fsm_context: FSMContext,
):
    """Если state уже == thinking_state — не пишем (idempotent), не restore'им."""
    msg = _make_message()
    await fsm_context.set_state(QuickAnalysisStates.GENERATING_FOLLOWUP)

    async with thinking_status(
        msg, state=fsm_context,
        thinking_state=QuickAnalysisStates.GENERATING_FOLLOWUP,
    ):
        # Внутри: state остался GENERATING_FOLLOWUP
        assert (await fsm_context.get_state()) == QuickAnalysisStates.GENERATING_FOLLOWUP.state

    # После: state по-прежнему GENERATING_FOLLOWUP (не restore'или потому что swap не делался)
    assert (await fsm_context.get_state()) == QuickAnalysisStates.GENERATING_FOLLOWUP.state
