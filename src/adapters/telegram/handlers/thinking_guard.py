"""Guard handler: перехватывает text/voice во время LLM-вызовов.

Срабатывает на FSM-state'ах, выставленных thinking_status context manager'ом:
- QuickAnalysisStates.GENERATING_INITIAL  — initial разбор после branch tap
- QuickAnalysisStates.GENERATING_FOLLOWUP — followup в Q&A
- OnboardingStates.GENERATING_INITIAL_ONB — initial в онбординге

Voice НЕ транскрибируется (экономия STT). Handler намеренно не принимает
AsyncSession — guard должен быть максимально дешёвым, без DB-touch.
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.types import Message

from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import THINKING_GUARD_NOTICE

log = structlog.get_logger(__name__)

_GENERATING_STATES = StateFilter(
    OnboardingStates.GENERATING_INITIAL_ONB,
    QuickAnalysisStates.GENERATING_INITIAL,
    QuickAnalysisStates.GENERATING_FOLLOWUP,
)


async def handle_thinking_text(message: Message, state: FSMContext) -> None:
    if message.from_user is None:
        return
    log.info(
        "thinking_guard.intercepted",
        kind="text",
        telegram_user_id=message.from_user.id,
        current_state=await state.get_state(),
    )
    await message.answer(THINKING_GUARD_NOTICE)


async def handle_thinking_voice(message: Message, state: FSMContext) -> None:
    if message.from_user is None:
        return
    log.info(
        "thinking_guard.intercepted",
        kind="voice",
        telegram_user_id=message.from_user.id,
        current_state=await state.get_state(),
    )
    await message.answer(THINKING_GUARD_NOTICE)


def create_router() -> Router:
    router = Router(name="thinking_guard")
    router.message.register(handle_thinking_text, _GENERATING_STATES, F.text)
    router.message.register(handle_thinking_voice, _GENERATING_STATES, F.voice)
    return router
