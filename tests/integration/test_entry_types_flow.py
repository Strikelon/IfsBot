"""Сквозные флоу «ситуация»/«конфликт»: онбординг, повтор, пейвол, resume."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.entry_flow import (
    handle_entry_input,
    handle_entry_loop,
    handle_new_conflict_command,
)
from adapters.telegram.handlers.onboarding import (
    handle_life_context_input,
    handle_life_context_loop,
    handle_per_session_context_onb,
)
from adapters.telegram.handlers.quick import handle_per_session_context_text
from adapters.telegram.states import EntryStates, OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import ANALYSIS_START_NOTICE, LIMIT_REACHED, LOOP_ADD_BTN_DONE
from data.models import LlmService, UsageMetric
from data.repositories import (
    UsageEventRepository,
    UserRepository,
)
from domain.entry_types import ENTRY_TYPES, KIND_CONFLICT, KIND_SITUATION

_GEN_OK = SimpleNamespace(
    text=(
        "Разбор по существу.\n\n**Резюме**\n\nСуть.\n\n**Какую сторону исследуем?**\n"
        '===QA_OPTIONS===\n{"kind":"choice","options":["Сторона А","Сторона Б"],'
        '"map":{"focus":"x","type":"manager","stage":"вступление","covered":[]}}'
    ),
    error_category=None, model_used="google/gemini-3.5-flash",
    input_tokens=100, output_tokens=200,
)


class _Settings:
    support_contact_url = "https://t.me/test_support"
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    memory_inject_enabled = False
    image_questions_enabled = False  # шаг 1–3 вопросов покрыт юнитами Task 12
    openrouter_context_model = "openai/gpt-4o-mini"


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _msg(text: str, uid: int = 42) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=uid, is_bot=False, first_name="Катя")
    m.chat = Chat(id=uid, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock(return_value=AsyncMock(message_id=11))
    return m


def _sent(msg: Message) -> list[str]:
    return [c.args[0] for c in msg.answer.await_args_list]


@pytest.mark.asyncio
async def test_full_situation_onboarding_to_qa(test_session):
    """Выбор сделан (EntryStates) → рассказ → loop done → знакомство → done →
    переживания → разбор (мок) → IN_QA. ANALYSIS_START_NOTICE показан (онбординг)."""
    await UserRepository(test_session).get_or_create_by_tg_id(
        telegram_user_id=42, display_name="Катя"
    )
    fsm = _fsm()
    await fsm.set_state(EntryStates.ASKING_ENTRY)
    await fsm.update_data(**{FSM_KEY_ENTRY_KIND: KIND_SITUATION})

    m1 = _msg("поссорилась с руководителем")
    await handle_entry_input(
        m1, fsm, session=test_session, bot=AsyncMock(),
        openai_client=AsyncMock(), settings=_Settings(),
    )
    assert await fsm.get_state() == EntryStates.ENTRY_LOOP.state

    m2 = _msg(LOOP_ADD_BTN_DONE)
    await handle_entry_loop(
        m2, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini", bot=None,
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_LIFE_CONTEXT.state

    m3 = _msg("мне 30, работаю в найме")
    await handle_life_context_input(
        m3, fsm, session=test_session, bot=AsyncMock(),
        openai_client=AsyncMock(), settings=_Settings(),
    )
    m4 = _msg(LOOP_ADD_BTN_DONE)
    await handle_life_context_loop(
        m4, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini", bot=None,
    )
    assert await fsm.get_state() == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state
    assert m4.answer.await_args.args[0] == (
        ENTRY_TYPES[KIND_SITUATION].per_session_question
    )

    m5 = _msg("боялась подвести команду")
    with patch(
        "adapters.telegram.handlers.quick.generate_quick_response",
        new=AsyncMock(return_value=_GEN_OK),
    ):
        await handle_per_session_context_onb(
            m5, fsm, session=test_session, bot=AsyncMock(),
            openai_client=AsyncMock(), openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(), settings=_Settings(),
        )
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    sent = _sent(m5)
    assert ANALYSIS_START_NOTICE in sent          # онбординг: дисклеймер показан
    assert any("Разбор по существу." in t for t in sent)
    # usage записан с версией situation.v1
    from sqlalchemy import select

    from data.models import LlmUsageEvent

    versions = (
        await test_session.execute(select(LlmUsageEvent.prompt_version))
    ).scalars().all()
    assert "situation.v1" in versions


@pytest.mark.asyncio
async def test_repeat_conflict_no_intro_and_no_notice(test_session, make_quick_session):
    """Повтор (onboarded): /new_internal_conflict → рассказ → done →
    per-session → разбор БЕЗ ANALYSIS_START_NOTICE (B6)."""
    sess = await make_quick_session(status="paused")  # onboarded юзер с прошлым сном
    from data.models import User

    user = await test_session.get(User, sess.user_id)
    fsm = _fsm()

    m0 = _msg("/new_internal_conflict", uid=user.telegram_user_id)
    await handle_new_conflict_command(
        m0, fsm, session=test_session, settings=_Settings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert await fsm.get_state() == EntryStates.ASKING_ENTRY.state

    m1 = _msg("не могу сесть за диплом", uid=user.telegram_user_id)
    await handle_entry_input(
        m1, fsm, session=test_session, bot=AsyncMock(),
        openai_client=AsyncMock(), settings=_Settings(),
    )
    m2 = _msg(LOOP_ADD_BTN_DONE, uid=user.telegram_user_id)
    await handle_entry_loop(
        m2, fsm, session=test_session,
        openrouter_classifier_client=AsyncMock(),
        openrouter_intent_model="openai/gpt-4o-mini", bot=None,
    )
    assert await fsm.get_state() == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    assert m2.answer.await_args.args[0] == ENTRY_TYPES[KIND_CONFLICT].per_session_question

    m3 = _msg("злюсь на себя", uid=user.telegram_user_id)
    with patch(
        "adapters.telegram.handlers.quick.generate_quick_response",
        new=AsyncMock(return_value=_GEN_OK),
    ):
        await handle_per_session_context_text(
            m3, fsm, session=test_session, openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(), settings=_Settings(), bot=None,
        )
    assert await fsm.get_state() == QuickAnalysisStates.IN_QA.state
    assert ANALYSIS_START_NOTICE not in _sent(m3)  # повтор: без дисклеймера
    from sqlalchemy import select

    from data.models import LlmUsageEvent

    versions = (
        await test_session.execute(select(LlmUsageEvent.prompt_version))
    ).scalars().all()
    assert "conflict.v1" in versions


@pytest.mark.asyncio
async def test_paywall_blocks_repeat_entry_command(test_session, make_quick_session):
    """Над лимитом /new_external_situation не пускает (deny_entry, B10)."""
    from adapters.telegram.handlers.entry_flow import handle_new_situation_command
    from data.models import User

    sess = await make_quick_session(status="paused")
    user = await test_session.get(User, sess.user_id)
    # Нагоняем траты выше лимита.
    await UsageEventRepository(test_session).record(
        user_id=user.id, service=LlmService.OPENROUTER, model="x",
        usage_metric=UsageMetric.OUTPUT_TOKENS, usage_amount=1,
        cost_usd=999, prompt_version="t",
    )
    await test_session.flush()

    class _TightSettings(_Settings):
        user_spend_limit_usd = 0.01
        new_user_spend_limit_usd = 0.01

    fsm = _fsm()
    m = _msg("/new_external_situation", uid=user.telegram_user_id)
    await handle_new_situation_command(
        m, fsm, session=test_session, settings=_TightSettings(), bot=None,
        openrouter_classifier_client=None,
    )
    assert m.answer.await_args.args[0] == LIMIT_REACHED.format(contact="https://t.me/test_support")
    assert await fsm.get_state() is None
