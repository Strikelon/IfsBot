"""Оркестратор пре-вопросов: per-kind offer-текст, шаблон вопроса, фаза лога."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from adapters.telegram.handlers.image_questions import (
    _ask_current_question,
    _entry_setup,
    start_image_questions_flow,
)
from data.models import SessionMessageContentType, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_DREAM, KIND_SITUATION


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _msg() -> AsyncMock:
    m = AsyncMock()
    m.answer = AsyncMock(return_value=AsyncMock(message_id=101))
    return m


async def _make_entry(test_session, kind: str):
    user = await UserRepository(test_session).get_or_create_by_tg_id(telegram_user_id=42)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="x", kind=kind
    )
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    return user, dream, sess


@pytest.mark.asyncio
async def test_entry_setup_resolves_kind_cfg_bundle(test_session):
    user, _, sess = await _make_entry(test_session, KIND_SITUATION)
    kind, cfg, bundle = await _entry_setup(
        test_session, user_id=user.id, session_id=sess.id
    )
    assert kind == KIND_SITUATION
    assert cfg is ENTRY_TYPES[KIND_SITUATION]
    assert bundle.version_pre_questions == "situation_questions.v1"


@pytest.mark.asyncio
async def test_offer_text_per_kind(test_session):
    user, dream, sess = await _make_entry(test_session, KIND_SITUATION)
    message, fsm = _msg(), _fsm()
    await start_image_questions_flow(
        message=message, state=fsm, session=test_session, user=user,
        session_id=sess.id, dream_id=dream.id, is_onboarding=False,
        settings=AsyncMock(), bot=None,
    )
    sent_text = message.answer.await_args.args[0]
    assert sent_text == ENTRY_TYPES[KIND_SITUATION].preq_offer_text
    assert "по вашему сну" not in sent_text


@pytest.mark.asyncio
async def test_question_template_without_image_word_for_situation():
    fsm = _fsm()
    message = _msg()
    imgq = {
        "status": "answering", "idx": 0, "total": 2,
        "items": [
            {"name": "Чувства", "question": "Что вы почувствовали?", "answer": None,
             "skipped": False},
            {"name": "Опасения", "question": "Чего опасаетесь?", "answer": None,
             "skipped": False},
        ],
    }
    await _ask_current_question(
        message=message, state=fsm, imgq=imgq, cfg=ENTRY_TYPES[KIND_SITUATION]
    )
    text = message.answer.await_args.args[0]
    assert "Вопрос 1 из 2" in text
    assert "Что вы почувствовали?" in text
    assert "Образ" not in text


@pytest.mark.asyncio
async def test_question_template_dream_unchanged():
    fsm = _fsm()
    message = _msg()
    imgq = {
        "status": "answering", "idx": 0, "total": 1,
        "items": [{"name": "Тигр", "question": "Как вы относитесь к тиграм?",
                   "answer": None, "skipped": False}],
    }
    await _ask_current_question(
        message=message, state=fsm, imgq=imgq, cfg=ENTRY_TYPES[KIND_DREAM]
    )
    text = message.answer.await_args.args[0]
    # render_quick_markup html-экранирует кавычки (поведение до фичи не менялось).
    assert "Образ &quot;Тигр&quot;" in text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "dream_calls", "entry_calls"),
    [(KIND_DREAM, 1, 0), (KIND_SITUATION, 0, 1)],
)
async def test_answer_classifier_dispatched_by_kind(
    test_session, monkeypatch, kind, dream_calls, entry_calls
):
    """Сон идёт прежним классификатором, остальные типы — нейтральным: его ответ
    на встречный вопрос показывается юзеру дословно."""
    from adapters.telegram.handlers import image_questions as mod
    from llm.intent import IntentResult

    calls = {"dream": 0, "entry": 0}

    async def _fake_dream(**_kw):
        calls["dream"] += 1
        return IntentResult(outcome="answer", input_tokens=1, output_tokens=1)

    async def _fake_entry(**_kw):
        calls["entry"] += 1
        return IntentResult(outcome="answer", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(mod, "classify_image_answer_intent", _fake_dream)
    monkeypatch.setattr(mod, "classify_entry_answer_intent", _fake_entry)

    user, dream, sess = await _make_entry(test_session, kind)
    await mod._save_imgq(test_session, sess.id, {
        "status": "answering", "idx": 0, "total": 1,
        "items": [{"name": "Тема", "question": "Что почувствовали?",
                   "answer": None, "skipped": False}],
    })
    class _Settings:
        user_spend_limit_usd = 1_000_000.0
        new_user_spend_limit_usd = 1_000_000.0
        spend_limit_exempt_set = frozenset()
        memory_write_enabled = False
        image_questions_enabled = True
        memory_inject_enabled = False
        openrouter_context_model = "openai/gpt-4o-mini"

    fsm, message = _fsm(), _msg()
    message.chat = None
    await mod._answer_act(
        text="злость", via="text",
        content_type=SessionMessageContentType.TEXT, audio_file_id=None,
        message=message, state=fsm, session=test_session, user=user,
        session_id=sess.id, dream_id=dream.id, bot=None,
        openrouter_client=AsyncMock(), intent_client=AsyncMock(),
        intent_model="openai/gpt-4o-mini", context_client=AsyncMock(),
        context_model="openai/gpt-4o-mini", settings=_Settings(),
    )
    assert (calls["dream"], calls["entry"]) == (dream_calls, entry_calls)
