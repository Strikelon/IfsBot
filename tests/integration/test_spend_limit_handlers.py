from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import AsyncMock, patch

import pytest
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message, Voice
from aiogram.types import User as TgUser
from sqlalchemy import select

from adapters.telegram.states import (
    DreamCardStates,
    NewDreamStates,
    OnboardingStates,
    QuickAnalysisStates,
)
from adapters.telegram.voice_helper import transcribe_with_placeholder
from data.models import LlmService, Session, UsageMetric, User
from data.repositories import DreamRepository, UsageEventRepository, UserRepository


class _Settings:
    support_contact_url = "https://t.me/test_support"
    user_spend_limit_usd = 1.0
    new_user_spend_limit_usd = 1.0
    spend_limit_exempt_set = frozenset()
    image_questions_enabled = False  # регресс без фичи: карточка как до image-questions


def _fsm(tg_id=20) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=tg_id, user_id=tg_id),
    )


def _cb(data: str, tg_id=20):
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=tg_id, is_bot=False, first_name="A")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=tg_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _text_msg(text: str, tg_id=20):
    m = AsyncMock(spec=Message)
    m.from_user = TgUser(id=tg_id, is_bot=False, first_name="A")
    m.chat = Chat(id=tg_id, type="private")
    m.text = text
    m.voice = None
    m.answer = AsyncMock()
    return m


async def _make_over_limit_user_with_dream(session, tg_id=20):
    """Onboarded user whose cumulative neural spend has reached the limit, + a saved dream."""
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=tg_id, display_name="A")
    await repo.update_profile(user_id=user.id, profile_life_context="x")
    await repo.mark_onboarded(user_id=user.id)
    await UsageEventRepository(session).record(
        user_id=user.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("1.00"),
    )
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="тёмный лес",
    )
    await session.flush()
    return user, dream


def _sent_texts(mock_answer):
    return [c.args[0] for c in mock_answer.call_args_list if c.args]


def _voice_msg(tg_id):
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=tg_id, is_bot=False, first_name="A")
    msg.chat = Chat(id=tg_id, type="private")
    msg.voice = Voice(file_id="F", file_unique_id="U", duration=3)
    msg.answer = AsyncMock()
    return msg


@pytest.mark.asyncio
async def test_voice_blocked_over_limit_no_whisper(test_session):
    u = User(telegram_user_id=10, display_name="A")
    test_session.add(u)
    await test_session.flush()
    await UsageEventRepository(test_session).record(
        user_id=u.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("1.00"),
    )
    await test_session.flush()

    msg = _voice_msg(10)
    bot = AsyncMock()
    with patch(
        "adapters.telegram.voice_helper.transcribe_voice", new=AsyncMock()
    ) as stt:
        result = await transcribe_with_placeholder(
            message=msg, bot=bot, openai_client=AsyncMock(), log_prefix="t",
            session=test_session, settings=_Settings(),
        )
    assert result is None
    stt.assert_not_awaited()  # Whisper never called
    # paywall message sent, NOT the "Слушаю…" placeholder
    sent = [c.args[0] for c in msg.answer.call_args_list]
    assert any("лимит" in s.lower() for s in sent)


# ---------------------------------------------------------------------------
# Task 5: entry guards (over-limit blocks paid actions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_analysis_blocked_over_limit(test_session):
    from adapters.telegram.handlers.quick import handle_start_analysis_callback

    user, dream = await _make_over_limit_user_with_dream(test_session)
    cb = _cb(f"start_analysis:{dream.id}")

    await handle_start_analysis_callback(cb, _fsm(), session=test_session, settings=_Settings())

    # no Quick session created; paywall shown
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert sessions == []
    assert any("лимит" in s.lower() for s in _sent_texts(cb.message.answer))


@pytest.mark.asyncio
async def test_per_session_blocked_over_limit_no_generator(test_session):
    """Ветка убрана: per-session над лимитом → разбор не запускается, пейвол."""
    from adapters.telegram.handlers.quick import handle_per_session_context_text
    from data.models import SessionType
    from data.repositories import SessionRepository

    user, dream = await _make_over_limit_user_with_dream(test_session)
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await test_session.flush()
    fsm = _fsm()
    await fsm.set_state(QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _text_msg("вчера было сложно")
    with patch(
        "adapters.telegram.handlers.quick._run_initial_quick", new=AsyncMock()
    ) as gen:
        await handle_per_session_context_text(
            msg, fsm, session=test_session,
            openrouter_client=AsyncMock(), openrouter_classifier_client=AsyncMock(),
            settings=_Settings(),
        )

    gen.assert_not_awaited()
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


@pytest.mark.asyncio
async def test_newdream_command_blocked_over_limit(test_session):
    from adapters.telegram.handlers.newdream import handle_newdream_command

    await _make_over_limit_user_with_dream(test_session)
    msg = _text_msg("/newdream")
    fsm = _fsm()

    await handle_newdream_command(msg, fsm, session=test_session, settings=_Settings())

    # did not enter the new-dream flow; paywall shown
    assert await fsm.get_state() != NewDreamStates.ASKING_DREAM.state
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


@pytest.mark.asyncio
async def test_dream_card_text_blocked_over_limit_no_classifier(test_session):
    from adapters.telegram.handlers.quick import _DREAM_CARD_DREAM_ID, handle_dream_card_text

    user, dream = await _make_over_limit_user_with_dream(test_session)
    fsm = _fsm()
    await fsm.set_state(DreamCardStates.DECIDING_ANALYZE)
    await fsm.set_data({_DREAM_CARD_DREAM_ID: dream.id})

    msg = _text_msg("да, разбери")
    with patch(
        "adapters.telegram.handlers.quick.classify_confirm_intent", new=AsyncMock()
    ) as clf:
        await handle_dream_card_text(
            msg, fsm, session=test_session,
            openrouter_classifier_client=AsyncMock(),
            openrouter_intent_model="m",
            settings=_Settings(),
        )

    clf.assert_not_awaited()
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


@pytest.mark.asyncio
async def test_start_analysis_allowed_under_limit(test_session):
    """Under limit the same flow proceeds (sanity that the guard is conditional)."""
    from adapters.telegram.handlers.quick import handle_start_analysis_callback

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=21, display_name="B")
    await repo.update_profile(user_id=user.id, profile_life_context="x")
    await repo.mark_onboarded(user_id=user.id)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 10, tzinfo=UTC), transcription="лес",
    )
    await test_session.flush()

    cb = _cb(f"start_analysis:{dream.id}", tg_id=21)
    await handle_start_analysis_callback(cb, _fsm(21), session=test_session, settings=_Settings())

    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1  # session created — not blocked


# ---------------------------------------------------------------------------
# Task 6: saved-dream open shows the card but blocks Q&A resume over limit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_open_over_limit_shows_card_blocks_resume(test_session):
    from unittest.mock import AsyncMock as _AM

    from adapters.telegram.handlers.dreams import handle_dream_by_id
    from data.repositories import DreamRepository as _DR

    user, dream = await _make_over_limit_user_with_dream(test_session)
    msg = _text_msg(f"/dream_{dream.id}")
    fsm = _fsm()

    with (
        patch.object(_DR, "dream_has_analysis", new=_AM(return_value=True)),
        patch(
            "adapters.telegram.handlers.quick._resume_qa_for_dream", new=_AM()
        ) as resume,
    ):
        await handle_dream_by_id(msg, state=fsm, session=test_session, settings=_Settings())

    # card was rendered (read OK) ...
    assert any("тёмный лес" in s for s in _sent_texts(msg.answer))
    # ... but Q&A resume blocked + paywall shown, FSM not in IN_QA
    resume.assert_not_awaited()
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))
    assert await fsm.get_state() != QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_dream_open_under_limit_resumes(test_session):
    from unittest.mock import AsyncMock as _AM

    from adapters.telegram.handlers.dreams import handle_dream_by_id
    from data.repositories import DreamRepository as _DR

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=22, display_name="C")
    await repo.update_profile(user_id=user.id, profile_life_context="x")
    await repo.mark_onboarded(user_id=user.id)
    dream = await DreamRepository(test_session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 10, tzinfo=UTC), transcription="река",
    )
    await test_session.flush()
    msg = _text_msg(f"/dream_{dream.id}", tg_id=22)

    with (
        patch.object(_DR, "dream_has_analysis", new=_AM(return_value=True)),
        patch(
            "adapters.telegram.handlers.quick._resume_qa_for_dream", new=_AM()
        ) as resume,
    ):
        await handle_dream_by_id(msg, state=_fsm(22), session=test_session, settings=_Settings())

    resume.assert_awaited_once()
    assert not any("лимит" in s.lower() for s in _sent_texts(msg.answer))


# ---------------------------------------------------------------------------
# Task 7: Q&A in-flow guard — crossing the cap mid-Q&A saves the dream and exits
# ---------------------------------------------------------------------------


async def _active_qa(session, tg_id=20):
    from data.models import SessionType
    from data.repositories import SessionRepository

    user, dream = await _make_over_limit_user_with_dream(session, tg_id=tg_id)
    sess = await SessionRepository(session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    await session.flush()
    fsm = _fsm(tg_id)
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})
    return user, dream, sess, fsm


@pytest.mark.asyncio
async def test_qa_text_over_limit_saves_and_exits(test_session):
    from adapters.telegram.handlers.quick import handle_quick_qa_text
    from data.models import SessionStatus

    user, dream, sess, fsm = await _active_qa(test_session)
    msg = _text_msg("ещё один вопрос")

    with patch(
        "adapters.telegram.handlers.quick._handle_qa_turn", new=AsyncMock()
    ) as turn:
        await handle_quick_qa_text(
            msg, fsm, session=test_session, bot=AsyncMock(),
            openrouter_client=AsyncMock(), openrouter_generator_model="m",
            openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="m",
            openrouter_intent_model="m", settings=_Settings(),
        )

    turn.assert_not_awaited()  # no LLM
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.COMPLETED  # session closed
    assert await fsm.get_state() is None  # FSM cleared
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))
    # dream row preserved
    assert await DreamRepository(test_session).get_by_id(dream.id, user_id=user.id) is not None


@pytest.mark.asyncio
async def test_qa_voice_over_limit_saves_and_exits_before_whisper(test_session):
    from adapters.telegram.handlers.quick import handle_quick_qa_voice
    from data.models import SessionStatus

    user, dream, sess, fsm = await _active_qa(test_session)
    msg = _voice_msg(20)

    with patch(
        "adapters.telegram.handlers.quick.transcribe_with_placeholder", new=AsyncMock()
    ) as stt:
        await handle_quick_qa_voice(
            msg, fsm, session=test_session, bot=AsyncMock(), openai_client=AsyncMock(),
            openrouter_client=AsyncMock(), openrouter_generator_model="m",
            openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="m",
            openrouter_intent_model="m", settings=_Settings(),
        )

    stt.assert_not_awaited()  # Whisper never reached
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.COMPLETED
    assert await fsm.get_state() is None
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


@pytest.mark.asyncio
async def test_continue_quick_over_limit_blocked(test_session):
    from adapters.telegram.handlers.quick import handle_continue_quick_callback

    user, dream = await _make_over_limit_user_with_dream(test_session)
    cb = _cb(f"continue_quick:{dream.id}")

    with patch(
        "adapters.telegram.handlers.quick._resume_qa_for_dream", new=AsyncMock()
    ) as resume:
        await handle_continue_quick_callback(
            cb, _fsm(), session=test_session, settings=_Settings()
        )

    resume.assert_not_awaited()
    assert any("лимит" in s.lower() for s in _sent_texts(cb.message.answer))


# ---------------------------------------------------------------------------
# Task 8: onboarding over-limit saves+exits AND marks onboarded
# ---------------------------------------------------------------------------


async def _onboarding_not_yet_done(session, tg_id=30):
    """Onboarding user (NOT yet onboarded) over the limit, with a dream."""
    repo = UserRepository(session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=tg_id, display_name="O")
    await repo.update_profile(user_id=user.id, profile_life_context="x")
    # intentionally NOT mark_onboarded → onboarding_completed_at is None
    await UsageEventRepository(session).record(
        user_id=user.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("1.00"),
    )
    dream = await DreamRepository(session).create(
        user_id=user.id, recorded_at=datetime(2026, 5, 10, tzinfo=UTC), transcription="онб-сон",
    )
    await session.flush()
    return user, dream


@pytest.mark.asyncio
async def test_onboarding_qa_over_limit_saves_exits_and_marks_onboarded(test_session):
    from adapters.telegram.handlers.quick import handle_quick_qa_text
    from data.models import SessionStatus, SessionType
    from data.repositories import SessionRepository

    user, dream = await _onboarding_not_yet_done(test_session)
    sess = await SessionRepository(test_session).create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK, is_onboarding_session=True,
    )
    await test_session.flush()
    fsm = _fsm(30)
    await fsm.set_state(QuickAnalysisStates.IN_QA)
    await fsm.set_data({"dream_id": dream.id, "session_id": sess.id})

    msg = _text_msg("ещё вопрос", tg_id=30)
    with patch(
        "adapters.telegram.handlers.quick._handle_qa_turn", new=AsyncMock()
    ) as turn:
        await handle_quick_qa_text(
            msg, fsm, session=test_session, bot=AsyncMock(),
            openrouter_client=AsyncMock(), openrouter_generator_model="m",
            openrouter_classifier_client=AsyncMock(), openrouter_classifier_model="m",
            openrouter_intent_model="m", settings=_Settings(),
        )

    turn.assert_not_awaited()
    await test_session.refresh(sess)
    await test_session.refresh(user)
    assert sess.status == SessionStatus.COMPLETED  # saved+exited
    assert user.onboarding_completed_at is not None  # marked onboarded
    assert await fsm.get_state() is None
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


@pytest.mark.asyncio
async def test_onboarding_analysis_entry_over_limit_marks_onboarded(test_session):
    """Ветка убрана: per-session в онбординге над лимитом → онбординг помечен, разбор не запущен."""
    from adapters.telegram.handlers.onboarding import (
        _FSM_KEY_ONBOARDING_DREAM_ID,
        handle_per_session_context_onb,
    )

    user, dream = await _onboarding_not_yet_done(test_session)
    fsm = _fsm(30)
    await fsm.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm.set_data({_FSM_KEY_ONBOARDING_DREAM_ID: dream.id})
    msg = _text_msg("контекст", tg_id=30)

    with patch(
        "adapters.telegram.handlers.onboarding._run_quick_for_onboarding", new=AsyncMock()
    ) as gen:
        await handle_per_session_context_onb(
            msg, fsm, session=test_session, bot=AsyncMock(),
            openai_client=AsyncMock(), openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(), settings=_Settings(),
        )

    gen.assert_not_awaited()  # paid analysis never ran
    await test_session.flush()
    await test_session.refresh(user)
    assert user.onboarding_completed_at is not None  # marked onboarded (no loop)
    assert any("лимит" in s.lower() for s in _sent_texts(msg.answer))


# ---------------------------------------------------------------------------
# Fix: unanalyzed dream over limit — card + paywall, NO analyze button, NO DECIDING state
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unanalyzed_dream_over_limit_no_button_no_state(test_session):
    from adapters.telegram.handlers.dreams import handle_dream_by_id

    user, dream = await _make_over_limit_user_with_dream(test_session)  # dream has NO analysis
    msg = _text_msg(f"/dream_{dream.id}")
    fsm = _fsm()

    await handle_dream_by_id(msg, state=fsm, session=test_session, settings=_Settings())

    sent = _sent_texts(msg.answer)
    assert any("тёмный лес" in s for s in sent)  # card shown (read OK)
    assert any("лимит" in s.lower() for s in sent)  # paywall shown
    # NOT parked in DECIDING_ANALYZE → next free text won't loop the paywall
    assert await fsm.get_state() != DreamCardStates.DECIDING_ANALYZE.state
    # the card message carried no inline keyboard ("Разобрать" button absent)
    for c in msg.answer.call_args_list:
        if c.args and "тёмный лес" in c.args[0]:
            assert c.kwargs.get("reply_markup") is None
