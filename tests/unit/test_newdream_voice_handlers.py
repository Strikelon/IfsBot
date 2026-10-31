"""Голосовые хэндлеры /newdream: DREAM_LOOP и CONFIRMING_ANALYSIS.

Оба были без покрытия — из-за этого прод-дефект 2026-09-03 (`TypeError:
_newdream_loop_act() got an unexpected keyword argument 'settings'`) прошёл
1155 зелёных тестов и молча ронял ответ голосом на «Хотите рассказать ещё?».
Юзер оставался без разбора: гейт подтверждения не показывался, сессия не
создавалась, транзакция откатывалась целиком.
"""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest import mock
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message, Voice
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers.newdream import (
    handle_newdream_confirm_analysis_voice,
    handle_newdream_dream_loop_voice,
)
from adapters.telegram.states import NewDreamStates, QuickAnalysisStates
from adapters.telegram.texts import (
    DREAM_CONFIRM_ANALYZE_PROMPT,
    LOOP_ADD_PROMPT_DREAM,
)
from data.models import DreamStatus, SessionType
from data.models import Session as SessionModel
from data.repositories import DreamRepository, UserRepository

_INTENT_MODEL = "openai/gpt-4o-mini"


class _NoLimitSettings:
    """Spend-limit stub: лимит недостижимо высокий, гейт трат не срабатывает."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()


@pytest_asyncio.fixture
async def fsm_context() -> FSMContext:
    storage = MemoryStorage()
    return FSMContext(storage=storage, key=StorageKey(bot_id=1, chat_id=42, user_id=42))


def _make_voice_message(user_id: int = 42) -> Message:
    """Голосовое сообщение: text=None, voice заполнен — как приходит из Telegram."""
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    message.chat = Chat(id=user_id, type="private")
    message.text = None
    message.voice = Voice(file_id="v1", file_unique_id="u1", duration=6)
    message.answer = AsyncMock()
    return message


def _intent_client(outcome: str, answer: str = "") -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            f'{{"outcome": "{outcome}", "answer": "{answer}", '
                            f'"reason": "x"}}'
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=5),
        )
    )
    return client


def _transcribed_as(text: str):
    """Патч STT: голосовое распознаётся в заданный текст."""
    return mock.patch(
        "adapters.telegram.handlers.onboarding._transcribe_message_voice",
        new_callable=AsyncMock,
        return_value=text,
    )


async def _seed_dream(session: AsyncSession, transcription: str):
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription=transcription
    )
    await session.commit()
    return user, dream


# ---------------------------------------------------------------------------
# DREAM_LOOP + голос
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_loop_voice_continue_appends_and_stays(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Голосом дорассказали сон → текст дописан, состояние DREAM_LOOP, снова вопрос."""
    _, dream = await _seed_dream(test_session, "первая часть")

    await fsm_context.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_voice_message()
    with _transcribed_as("вторая часть"):
        await handle_newdream_dream_loop_voice(
            message,
            fsm_context,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_classifier_client=_intent_client("continue"),
            openrouter_intent_model=_INTENT_MODEL,
            settings=_NoLimitSettings(),
        )

    assert dream.transcription == "первая часть\n\nвторая часть"
    assert await fsm_context.get_state() == NewDreamStates.DREAM_LOOP.state
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_DREAM


@pytest.mark.asyncio
async def test_dream_loop_voice_done_shows_confirmation_gate(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Голосом сказали «это всё» → гейт подтверждения, состояние CONFIRMING_ANALYSIS."""
    _, dream = await _seed_dream(test_session, "сон про синий лес")

    await fsm_context.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_voice_message()
    with _transcribed_as("нет, это всё"):
        await handle_newdream_dream_loop_voice(
            message,
            fsm_context,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_classifier_client=_intent_client("done"),
            openrouter_intent_model=_INTENT_MODEL,
            settings=_NoLimitSettings(),
        )

    assert await fsm_context.get_state() == NewDreamStates.CONFIRMING_ANALYSIS.state
    texts_sent = [call.args[0] for call in message.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_sent)


# ---------------------------------------------------------------------------
# CONFIRMING_ANALYSIS + голос
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_analysis_voice_analyze_creates_session(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Голосом согласились на разбор → Quick-сессия создана, сон в IN_ANALYSIS."""
    _, dream = await _seed_dream(test_session, "сон про море")

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_voice_message()
    with _transcribed_as("да, давай разберём"):
        await handle_newdream_confirm_analysis_voice(
            message,
            fsm_context,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_classifier_client=_intent_client("analyze"),
            openrouter_intent_model=_INTENT_MODEL,
            settings=_NoLimitSettings(),
        )

    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 1
    assert sessions[0].type == SessionType.QUICK
    assert sessions[0].dream_id == dream.id

    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS
    assert (
        await fsm_context.get_state()
        == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    )
