"""Экран выбора типа: кнопка, текст, question/unclear, идемпотентный onb_type_chosen."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import func, select

from adapters.telegram.handlers._common import FSM_KEY_ENTRY_KIND
from adapters.telegram.handlers.entry_choice import (
    _choice_text_act,
    handle_entry_choice_callback,
    handle_entry_choice_text,
)
from adapters.telegram.handlers.onboarding import _FSM_KEY_ONBOARDING_DREAM_ID
from adapters.telegram.states import EntryStates, OnboardingStates
from adapters.telegram.texts import ENTRY_CHOICE_UNCLEAR, LOOP_ADD_BTN_DONE
from data.analytics_events import AnalyticsEventType
from data.models import AnalyticsEvent, Dream
from data.repositories import UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_DREAM, KIND_SITUATION


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _callback(kind: str):
    cb = AsyncMock()
    cb.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    cb.data = f"entry_type:{kind}"
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=42, type="private")
    cb.message.answer = AsyncMock()
    return cb


def _message(text: str) -> Message:
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=42, is_bot=False, first_name="Катя")
    m.chat = Chat(id=42, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock(return_value=AsyncMock(message_id=5))
    return m


def _msg() -> Message:
    return _message("")


def _patch_classifier(monkeypatch, *, outcome: str, told: bool) -> None:
    """Подменить classify_entry_type_intent в модуле entry_choice."""
    from adapters.telegram.handlers import entry_choice
    from llm.intent import IntentResult

    async def _fake(**_kwargs):
        return IntentResult(
            outcome=outcome, told=told, answer="", input_tokens=10, output_tokens=2
        )

    monkeypatch.setattr(entry_choice, "classify_entry_type_intent", _fake)


@pytest.mark.asyncio
async def test_callback_situation_moves_to_entry_states(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(
        telegram_user_id=42, display_name="Катя"
    )
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    cb = _callback(KIND_SITUATION)

    await handle_entry_choice_callback(cb, fsm, session=test_session, bot=AsyncMock())

    assert await fsm.get_state() == EntryStates.ASKING_ENTRY.state
    assert (await fsm.get_data())[FSM_KEY_ENTRY_KIND] == KIND_SITUATION
    sent = cb.message.answer.await_args.args[0]
    assert sent == ENTRY_TYPES[KIND_SITUATION].onb_invite.format(name="Катя")
    # onb_type_chosen записан с типом
    ev = (
        await test_session.execute(
            select(AnalyticsEvent).where(AnalyticsEvent.event_type == "onb_type_chosen")
        )
    ).scalar_one()
    assert ev.event_metadata["entry_type"] == KIND_SITUATION


@pytest.mark.asyncio
async def test_callback_dream_moves_to_asking_dream(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(
        telegram_user_id=42, display_name="Катя"
    )
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    cb = _callback(KIND_DREAM)
    await handle_entry_choice_callback(cb, fsm, session=test_session, bot=AsyncMock())
    assert await fsm.get_state() == OnboardingStates.ASKING_DREAM.state
    assert "первый сон" in cb.message.answer.await_args.args[0]


@pytest.mark.asyncio
async def test_type_chosen_is_idempotent(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    await handle_entry_choice_callback(
        _callback(KIND_SITUATION), fsm, session=test_session, bot=AsyncMock()
    )
    # Юзер вернулся и тапнул ещё раз (двойной тап/стейл) — второй строки нет.
    await fsm.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    await handle_entry_choice_callback(
        _callback(KIND_SITUATION), fsm, session=test_session, bot=AsyncMock()
    )
    n = (
        await test_session.execute(
            select(func.count()).select_from(AnalyticsEvent).where(
                AnalyticsEvent.event_type == "onb_type_chosen"
            )
        )
    ).scalar_one()
    assert n == 1


@pytest.mark.asyncio
async def test_free_text_unclear_reasks_with_buttons(test_session):
    await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    fsm = _fsm()
    await fsm.set_state(OnboardingStates.CHOOSING_ENTRY_TYPE)
    msg = _message("ммм")

    intent = SimpleNamespace(
        outcome="unclear", answer="", told=False, input_tokens=1, output_tokens=1
    )
    with patch(
        "adapters.telegram.handlers.entry_choice.classify_entry_type_intent",
        new=AsyncMock(return_value=intent),
    ):
        await handle_entry_choice_text(
            msg, fsm, session=test_session, bot=None,
            openrouter_classifier_client=AsyncMock(),
            openrouter_intent_model="openai/gpt-4o-mini",
        )
    assert await fsm.get_state() == OnboardingStates.CHOOSING_ENTRY_TYPE.state
    sent_texts = [c.args[0] for c in msg.answer.await_args_list]
    assert ENTRY_CHOICE_UNCLEAR in sent_texts
    # переспрос — с клавиатурой из трёх кнопок
    kb = msg.answer.await_args.kwargs.get("reply_markup")
    assert kb is not None and len(kb.inline_keyboard) == 3


@pytest.mark.asyncio
async def test_told_situation_is_saved_and_skips_asking(test_session, monkeypatch):
    """Рассказал ситуацию сразу → запись создана, показана юзеру, состояние = loop."""
    story = "Вчера поругалась с мамой из-за денег и до сих пор трясёт."
    _patch_classifier(monkeypatch, outcome=KIND_SITUATION, told=True)
    message, fsm = _msg(), _fsm()

    await _choice_text_act(
        text=story, message=message, state=fsm, session=test_session, bot=None,
        client=AsyncMock(), model="openai/gpt-4o-mini", telegram_user_id=42,
    )

    dream = (await test_session.execute(select(Dream))).scalars().one()
    assert dream.kind == KIND_SITUATION
    assert dream.transcription == story
    assert await fsm.get_state() == EntryStates.ENTRY_LOOP.state
    assert (await fsm.get_data())[_FSM_KEY_ONBOARDING_DREAM_ID] == dream.id
    texts_sent = [c.args[0] for c in message.answer.await_args_list]
    assert any(story in t for t in texts_sent), "юзеру не показали, что записано"
    assert not any("Опишите её подробно" in t for t in texts_sent), "просим рассказать заново"
    kb = message.answer.await_args.kwargs["reply_markup"]
    assert LOOP_ADD_BTN_DONE in str(kb)


@pytest.mark.asyncio
async def test_told_dream_goes_to_existing_dream_loop(test_session, monkeypatch):
    """Для сна перепрыгивание ведёт в СУЩЕСТВУЮЩЕЕ состояние DREAM_LOOP (B7: имена не трогаем)."""
    _patch_classifier(monkeypatch, outcome=KIND_DREAM, told=True)
    message, fsm = _msg(), _fsm()

    await _choice_text_act(
        text="Мне снилось, что я падаю с крыши.", message=message, state=fsm,
        session=test_session, bot=None, client=AsyncMock(),
        model="openai/gpt-4o-mini", telegram_user_id=42,
    )

    dream = (await test_session.execute(select(Dream))).scalars().one()
    assert dream.kind == KIND_DREAM
    assert await fsm.get_state() == OnboardingStates.DREAM_LOOP.state


@pytest.mark.asyncio
async def test_told_emits_onb_dream_told_once(test_session, monkeypatch):
    """Воронка не должна ослепнуть: шаг onb_dream_told эмитится и на этом пути."""
    _patch_classifier(monkeypatch, outcome=KIND_SITUATION, told=True)
    message, fsm = _msg(), _fsm()
    await _choice_text_act(
        text="Меня сокращают, не сплю третью ночь.", message=message, state=fsm,
        session=test_session, bot=None, client=AsyncMock(),
        model="openai/gpt-4o-mini", telegram_user_id=42,
    )
    rows = (
        await test_session.execute(
            select(AnalyticsEvent).where(
                AnalyticsEvent.event_type == AnalyticsEventType.ONB_DREAM_TOLD
            )
        )
    ).scalars().all()
    assert len(rows) == 1
    assert rows[0].event_metadata.get("entry_type") == KIND_SITUATION


@pytest.mark.asyncio
async def test_not_told_still_asks_to_tell(test_session, monkeypatch):
    """told=False — поведение прежнее: приглашение рассказать, записи нет."""
    _patch_classifier(monkeypatch, outcome=KIND_SITUATION, told=False)
    message, fsm = _msg(), _fsm()
    await _choice_text_act(
        text="хочу разобрать ситуацию", message=message, state=fsm,
        session=test_session, bot=None, client=AsyncMock(),
        model="openai/gpt-4o-mini", telegram_user_id=42,
    )
    assert (await test_session.execute(select(Dream))).scalars().all() == []
    assert await fsm.get_state() == EntryStates.ASKING_ENTRY.state
