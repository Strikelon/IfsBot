"""Плейсхолдер «думаю…» обязан быть по типу обращения во ВСЁМ общем конвейере.

Дефект с прода: в разборе волнующей ситуации при ответах на
пре-вопросы всплывала фраза «Перевожу с языка снов…». Причина — голый вызов
`thinking_status(message)` без `statuses=`: дефолт в модуле сновый. Тесты держат
все три точки, где классификатор крутится под плейсхолдером, а тип уже известен.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage

from adapters.telegram.texts import THINKING_STATUSES
from data.models import SessionMessageContentType, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository
from domain.entry_types import ENTRY_TYPES, KIND_CONFLICT, KIND_DREAM, KIND_SITUATION

NEW_KINDS = (KIND_SITUATION, KIND_CONFLICT)


def _dream_only() -> set[str]:
    """Сновые фразы, которых у новых типов быть не должно.

    Просто «не из THINKING_STATUSES» проверять нельзя: нейтральные «Завариваю
    мысли…» и «Подбираю слова бережно…» намеренно общие для всех типов, и
    стартовая фраза ротации выбирается случайно.
    """
    shared = set(ENTRY_TYPES[KIND_SITUATION].thinking_statuses) | set(
        ENTRY_TYPES[KIND_CONFLICT].thinking_statuses
    )
    return set(THINKING_STATUSES) - shared


class _Settings:
    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    memory_write_enabled = False
    image_questions_enabled = True
    memory_inject_enabled = False
    openrouter_context_model = "openai/gpt-4o-mini"


def _fsm() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(), key=StorageKey(bot_id=1, chat_id=42, user_id=42)
    )


def _msg() -> AsyncMock:
    m = AsyncMock()
    m.answer = AsyncMock(return_value=AsyncMock(message_id=101))
    m.chat = None
    return m


def _placeholder_text(message: AsyncMock) -> str:
    """Первое, что уходит юзеру, — сам плейсхолдер."""
    return message.answer.await_args_list[0].args[0]


async def _make_entry(test_session, kind: str, tg_id: int = 42):
    user = await UserRepository(test_session).get_or_create_by_tg_id(
        telegram_user_id=tg_id
    )
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="x", kind=kind
    )
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    return user, dream, sess


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", NEW_KINDS)
async def test_preq_answer_placeholder_is_per_kind(test_session, monkeypatch, kind):
    """Ответ на пре-вопрос — ровно тот экран, где всплывал «язык снов»."""
    from adapters.telegram.handlers import image_questions as mod
    from llm.intent import IntentResult

    async def _fake(**_kw):
        return IntentResult(outcome="answer", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(mod, "classify_entry_answer_intent", _fake)

    user, dream, sess = await _make_entry(test_session, kind)
    await mod._save_imgq(test_session, sess.id, {
        "status": "answering", "idx": 0, "total": 1,
        "items": [{"name": "Тема", "question": "Что почувствовали?",
                   "answer": None, "skipped": False}],
    })
    message = _msg()
    await mod._answer_act(
        text="злость", via="text",
        content_type=SessionMessageContentType.TEXT, audio_file_id=None,
        message=message, state=_fsm(), session=test_session, user=user,
        session_id=sess.id, dream_id=dream.id, bot=None,
        openrouter_client=AsyncMock(), intent_client=AsyncMock(),
        intent_model="openai/gpt-4o-mini", context_client=AsyncMock(),
        context_model="openai/gpt-4o-mini", settings=_Settings(),
    )
    text = _placeholder_text(message)
    assert text in ENTRY_TYPES[kind].thinking_statuses
    assert text not in _dream_only()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", NEW_KINDS)
async def test_preq_offer_placeholder_is_per_kind(test_session, monkeypatch, kind):
    """Свободная реплика на приглашении «можно задать 2–3 вопроса?»."""
    from adapters.telegram.handlers import image_questions as mod
    from llm.intent import IntentResult

    async def _fake(**_kw):
        return IntentResult(outcome="skip", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(mod, "classify_entry_offer_intent", _fake)

    user, dream, sess = await _make_entry(test_session, kind)
    await mod._save_imgq(test_session, sess.id, {"status": "offered"})
    message = _msg()
    await mod._offer_act(
        text="не хочу", via="text",
        message=message, state=_fsm(), session=test_session, user=user,
        session_id=sess.id, dream_id=dream.id, bot=None,
        openrouter_client=AsyncMock(), intent_client=AsyncMock(),
        intent_model="openai/gpt-4o-mini", context_client=AsyncMock(),
        context_model="openai/gpt-4o-mini", settings=_Settings(),
    )
    text = _placeholder_text(message)
    assert text in ENTRY_TYPES[kind].thinking_statuses
    assert text not in _dream_only()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", NEW_KINDS)
async def test_card_decide_placeholder_is_per_kind(test_session, monkeypatch, kind):
    """Свободная реплика на гейте «Разобрать сейчас?» у карточки обращения."""
    from adapters.telegram.handlers import quick as mod
    from llm.intent import IntentResult

    async def _fake(**_kw):
        return IntentResult(outcome="unclear", input_tokens=1, output_tokens=1)

    monkeypatch.setattr(mod, "classify_confirm_intent", _fake)

    user, dream, _ = await _make_entry(test_session, kind, tg_id=43)
    fsm = _fsm()
    await fsm.update_data(**{mod._DREAM_CARD_DREAM_ID: dream.id})
    message = _msg()
    message.from_user = AsyncMock(id=user.telegram_user_id)
    await mod._dream_card_decide_act(
        text="что это значит?", message=message, state=fsm, session=test_session,
        client=AsyncMock(), model="openai/gpt-4o-mini", settings=_Settings(),
    )
    text = _placeholder_text(message)
    assert text in ENTRY_TYPES[kind].thinking_statuses
    assert text not in _dream_only()


def test_dream_statuses_stay_dream_themed():
    """Гард против «починим общим текстом»: у сна фразы остаются сновыми."""
    assert "Перевожу с языка снов…" in ENTRY_TYPES[KIND_DREAM].thinking_statuses
    assert "Перевожу с языка снов…" in _dream_only()
    for kind in NEW_KINDS:
        for phrase in ENTRY_TYPES[kind].thinking_statuses:
            assert phrase not in _dream_only()
