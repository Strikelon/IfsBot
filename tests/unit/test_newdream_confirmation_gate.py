"""Unit tests for /newdream confirmation gate (Task #4).

После DREAM_LOOP «Это всё» → CONFIRMING_ANALYSIS (карточка + reply-клава).
Тап «Разобрать сон» → QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.
Тап «Сохранить без разбора» → FSM clear, dream остаётся RECORDED.
"""

from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.handlers.newdream import (
    handle_newdream_confirm_analysis,
    handle_newdream_confirm_callback,
    handle_newdream_dream_loop,
)
from adapters.telegram.states import NewDreamStates, QuickAnalysisStates
from adapters.telegram.texts import (
    CONFIRM_ANALYZE_BTN,
    DREAM_CONFIRM_ANALYZE_PROMPT,
    DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE,
    LIMIT_REACHED,
    LOOP_ADD_BTN_DONE,
    SAVE_WITHOUT_ANALYSIS_BTN,
)
from data.models import DreamStatus, LlmService, UsageMetric
from data.models import Session as SessionModel
from data.repositories import DreamRepository, UsageEventRepository, UserRepository


@pytest_asyncio.fixture
async def fsm_context() -> FSMContext:
    storage = MemoryStorage()
    ctx = FSMContext(
        storage=storage,
        key=StorageKey(bot_id=1, chat_id=42, user_id=42),
    )
    return ctx


def _make_message(text: str, user_id: int = 42) -> Message:
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    message.chat = Chat(id=user_id, type="private")
    message.text = text
    message.voice = None
    message.answer = AsyncMock()
    return message


def _make_callback(data: str, user_id: int = 42) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=user_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


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


_INTENT_MODEL = "openai/gpt-4o-mini"
class _Settings:
    support_contact_url = "https://t.me/test_support"
    """Минимум для spend_guard: лимит есть, но траты по умолчанию нулевые."""

    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0
    spend_limit_exempt_set = frozenset()


# Хэндлеры loop-шага живут без spend-guard'а, гейт подтверждения — с ним.
_LOOP_KW = {
    "openrouter_classifier_client": _intent_client("unclear"),
    "openrouter_intent_model": _INTENT_MODEL,
}
_INTENT_KW = {**_LOOP_KW, "settings": _Settings()}


# ---------------------------------------------------------------------------
# 1. DREAM_LOOP «Это всё» → CONFIRMING_ANALYSIS (не Quick!)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_done_tap_in_dream_loop_shows_confirmation_gate(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """После DREAM_LOOP «Это всё»:
    - state переходит в CONFIRMING_ANALYSIS (не QuickAnalysisStates!)
    - юзер видит DREAM_CONFIRM_ANALYZE_PROMPT
    - Quick-сессия НЕ создана.
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про лес и туман",
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message(LOOP_ADD_BTN_DONE)
    await handle_newdream_dream_loop(message, fsm_context, session=test_session, **_LOOP_KW)

    # Состояние — CONFIRMING_ANALYSIS, не Quick
    assert await fsm_context.get_state() == NewDreamStates.CONFIRMING_ANALYSIS.state

    # Юзер видит prompt с вопросом
    texts_sent = [call.args[0] for call in message.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_sent)

    # Quick-сессия НЕ создана
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0


# ---------------------------------------------------------------------------
# 2. CONFIRMING_ANALYSIS + тап «Разобрать сон» → Quick
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tap_confirm_analyze_creates_quick_session(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Тап CONFIRM_ANALYZE_BTN в CONFIRMING_ANALYSIS:
    - создаётся Quick-сессия
    - dream.status = IN_ANALYSIS
    - state = QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT
    """
    from data.models import SessionType

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про море",
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message(CONFIRM_ANALYZE_BTN)
    await handle_newdream_confirm_analysis(message, fsm_context, session=test_session, **_INTENT_KW)

    # Quick-сессия создана
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 1
    new_sess = sessions[0]
    assert new_sess.type == SessionType.QUICK
    assert new_sess.is_onboarding_session is False
    assert new_sess.dream_id == dream.id

    # Dream статус повышен
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS

    # FSM data содержит dream_id и session_id
    fsm_data = await fsm_context.get_data()
    assert fsm_data["dream_id"] == dream.id
    assert fsm_data["session_id"] == new_sess.id

    # State → Quick
    assert (
        await fsm_context.get_state()
        == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    )


# ---------------------------------------------------------------------------
# 3. CONFIRMING_ANALYSIS + тап «Сохранить без разбора» → FSM clear, RECORDED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tap_save_without_analysis_clears_state(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Тап SAVE_WITHOUT_ANALYSIS_BTN в CONFIRMING_ANALYSIS:
    - FSM очищен (state=None)
    - Quick-сессия не создана
    - dream.status остаётся RECORDED
    - юзер получает DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про горы",
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message(SAVE_WITHOUT_ANALYSIS_BTN)
    await handle_newdream_confirm_analysis(message, fsm_context, session=test_session, **_INTENT_KW)

    # FSM очищен
    assert await fsm_context.get_state() is None

    # Quick-сессия НЕ создана
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0

    # Dream статус — RECORDED (не изменён)
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.RECORDED

    # Юзер получил notice
    texts_sent = [call.args[0] for call in message.answer.call_args_list]
    assert any(DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE in t for t in texts_sent)


# ---------------------------------------------------------------------------
# 4. Fallback: посторонний текст в CONFIRMING_ANALYSIS → повторить prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_confirm_freeform_analyze_creates_quick_session(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Свободный текст «давай разберём» → классификатор analyze → Quick-сессия."""
    from data.models import SessionType

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про реку"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message("да, давай разберём")
    await handle_newdream_confirm_analysis(
        message,
        fsm_context,
        session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=_intent_client("analyze"),
        openrouter_intent_model=_INTENT_MODEL,
    )

    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 1
    assert sessions[0].type == SessionType.QUICK
    assert (
        await fsm_context.get_state()
        == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    )


@pytest.mark.asyncio
async def test_confirm_freeform_save_clears_state(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Свободный текст «нет, потом» → классификатор save → FSM clear, RECORDED."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про поле"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message("нет, сохрани на потом")
    await handle_newdream_confirm_analysis(
        message,
        fsm_context,
        session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=_intent_client("save"),
        openrouter_intent_model=_INTENT_MODEL,
    )

    assert await fsm_context.get_state() is None
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.RECORDED


@pytest.mark.asyncio
async def test_confirm_callback_analyze_creates_session_and_extinguishes(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Тап inline «Разобрать сон» → Quick-сессия + кнопки гейта погашены."""
    from data.models import SessionType

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про лес"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id, confirm_gate_msg_id=555)

    cb = _make_callback("confirm_analyze")
    bot = AsyncMock()
    client = _intent_client("save")  # классификатор не должен вызываться
    await handle_newdream_confirm_callback(
        cb, fsm_context, session=test_session, settings=_Settings(),
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL, bot=bot,
    )

    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 1
    assert sessions[0].type == SessionType.QUICK
    assert (
        await fsm_context.get_state() == QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT.state
    )
    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=555, reply_markup=None
    )
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_confirm_callback_save_clears_and_extinguishes(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Тап inline «Сохранить без разбора» → FSM clear, RECORDED, кнопки погашены."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про горы"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id, confirm_gate_msg_id=555)

    cb = _make_callback("confirm_save")
    bot = AsyncMock()
    await handle_newdream_confirm_callback(
        cb, fsm_context, session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=_intent_client("analyze"),
        openrouter_intent_model=_INTENT_MODEL, bot=bot,
    )

    assert await fsm_context.get_state() is None
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.RECORDED
    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=555, reply_markup=None
    )


@pytest.mark.asyncio
async def test_confirm_text_extinguishes_previous_gate(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Ответ текстом в CONFIRMING_ANALYSIS гасит кнопки предыдущего гейта."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id, confirm_gate_msg_id=444)

    message = _make_message("хм, не уверен")
    bot = AsyncMock()
    await handle_newdream_confirm_analysis(
        message, fsm_context, session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=_intent_client("unclear"),
        openrouter_intent_model=_INTENT_MODEL, bot=bot,
    )

    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=444, reply_markup=None
    )


@pytest.mark.asyncio
async def test_confirmation_gate_long_dream_split_into_chunks(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Длинный сон (8-мин голосовое) → карточка-расшифровка дробится на чанки ≤4096,
    бот не падает на Telegram-лимите."""
    from adapters.telegram.handlers.newdream import _show_confirmation_gate

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    long_text = "во сне я шёл по бесконечному коридору и видел двери. " * 200  # ~10k симв.
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription=long_text
    )
    await test_session.commit()
    await fsm_context.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message("x")
    await _show_confirmation_gate(
        message=message, state=fsm_context, session=test_session,
        dream_id=dream.id, user_id=user.id,
    )

    texts = [c.args[0] for c in message.answer.await_args_list]
    # Ни одно отправленное сообщение не превышает Telegram-лимит
    assert texts and all(len(t) <= 4096 for t in texts)
    # Текст сна доставлен (целиком, по частям) + показан вопрос гейта
    assert "бесконечному коридору" in "".join(texts)
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts)


@pytest.mark.asyncio
async def test_confirm_counter_question_answers_and_keeps_gate(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Встречный вопрос на гейте → бот отвечает, гейт остаётся, разбор не запускается."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон"
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message("а что это даст?")
    await handle_newdream_confirm_analysis(
        message, fsm_context, session=test_session,
        settings=_Settings(),
        openrouter_classifier_client=_intent_client("question", answer="Поможет понять сон."),
        openrouter_intent_model=_INTENT_MODEL,
    )

    # Сессия не создана, state остался на гейте
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0
    assert await fsm_context.get_state() == NewDreamStates.CONFIRMING_ANALYSIS.state
    # Ответ отправлен + гейт показан снова
    texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("Поможет понять" in t for t in texts)
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts)


@pytest.mark.asyncio
async def test_tap_unknown_text_repeats_prompt(
    test_session: AsyncSession, schema, fsm_context: FSMContext
) -> None:
    """Любой текст кроме CONFIRM_ANALYZE_BTN / SAVE_WITHOUT_ANALYSIS_BTN:
    - state остаётся CONFIRMING_ANALYSIS
    - юзер видит prompt ещё раз
    - Quick-сессия НЕ создана, dream остаётся RECORDED.
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="произвольный сон",
    )
    await test_session.commit()

    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    message = _make_message("какой-то произвольный текст")
    await handle_newdream_confirm_analysis(message, fsm_context, session=test_session, **_INTENT_KW)

    # State не изменился
    assert await fsm_context.get_state() == NewDreamStates.CONFIRMING_ANALYSIS.state

    # Юзер увидел prompt ещё раз
    texts_sent = [call.args[0] for call in message.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_sent)

    # Quick-сессия НЕ создана, dream не изменён
    sessions = (await test_session.execute(select(SessionModel))).scalars().all()
    assert len(sessions) == 0
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.RECORDED


# ---------------------------------------------------------------------------
# Пейвол стоит НА кнопке «Разобрать», а не шагом позже (замечание 2026-08-24)
# ---------------------------------------------------------------------------


class _OverLimitSettings(_Settings):
    """Тот же конфиг — за лимит юзера выводят реальные траты в БД."""


async def _user_over_limit(session, tg_id: int = 42):
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=tg_id, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    await UsageEventRepository(session).record(
        user_id=user.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("2.00"),
    )
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про лес"
    )
    await session.commit()
    return user, dream


@pytest.mark.asyncio
async def test_analyze_button_over_limit_shows_paywall_and_starts_nothing(
    test_session: AsyncSession, fsm_context: FSMContext
):
    """Тап «Разобрать сон» над лимитом: стена сразу, сессии нет, FSM чист.

    Раньше гейт пропускал в ASKING_PER_SESSION_CONTEXT, и человек узнавал об
    отказе только после того, как напишет о своих переживаниях.
    """
    user, dream = await _user_over_limit(test_session)
    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id, confirm_gate_msg_id=555)

    cb = _make_callback("confirm_analyze")
    client = _intent_client("analyze")
    await handle_newdream_confirm_callback(
        cb, fsm_context, session=test_session, settings=_OverLimitSettings(),
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
        bot=AsyncMock(),
    )

    assert (await test_session.execute(select(SessionModel))).scalars().all() == []
    assert await fsm_context.get_state() is None
    assert cb.message.answer.await_args.args[0] == LIMIT_REACHED.format(contact="https://t.me/test_support")
    # След показа обязан остаться — иначе админка врёт (см. show_paywall).
    await test_session.refresh(user)
    assert user.paywall_shown_at is not None
    # Сон никуда не делся: он уже записан и лежит в /dreams.
    saved = await DreamRepository(test_session).get_by_id(dream.id, user_id=user.id)
    assert saved is not None and saved.status == DreamStatus.RECORDED


@pytest.mark.asyncio
async def test_freeform_over_limit_does_not_pay_for_classifier(
    test_session: AsyncSession, fsm_context: FSMContext
):
    """Свободный текст над лимитом: стена без платного классификатора."""
    user, dream = await _user_over_limit(test_session, tg_id=42)
    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id)

    client = _intent_client("analyze")
    message = _make_message("ну давай разбирай")
    await handle_newdream_confirm_analysis(
        message, fsm_context, session=test_session, settings=_OverLimitSettings(),
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
    )

    client.chat.completions.create.assert_not_awaited()
    assert message.answer.await_args.args[0] == LIMIT_REACHED.format(contact="https://t.me/test_support")


@pytest.mark.asyncio
async def test_save_without_analysis_still_works_over_limit(
    test_session: AsyncSession, fsm_context: FSMContext
):
    """«Сохранить без разбора» бесплатен — стеной его закрывать нечестно."""
    user, dream = await _user_over_limit(test_session)
    await fsm_context.set_state(NewDreamStates.CONFIRMING_ANALYSIS)
    await fsm_context.update_data(newdream_dream_id=dream.id, confirm_gate_msg_id=555)

    cb = _make_callback("confirm_save")
    await handle_newdream_confirm_callback(
        cb, fsm_context, session=test_session, settings=_OverLimitSettings(),
        openrouter_classifier_client=_intent_client("save"),
        openrouter_intent_model=_INTENT_MODEL, bot=AsyncMock(),
    )

    assert cb.message.answer.await_args.args[0] == DREAM_SAVED_WITHOUT_ANALYSIS_NOTICE
    assert await fsm_context.get_state() is None
    await test_session.refresh(user)
    assert user.paywall_shown_at is None
