from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    CallbackQuery,
    Chat,
    InlineKeyboardMarkup,
    Message,
    ReplyKeyboardMarkup,
)
from aiogram.types import User as TgUser
from sqlalchemy.ext.asyncio import AsyncEngine

from adapters.telegram.handlers.dreams import (
    handle_dream_by_id,
    handle_dream_open_callback,
    handle_dreams_list,
)
from adapters.telegram.handlers.quick import handle_continue_quick_callback
from adapters.telegram.texts import (
    AUTO_PAUSED_NOTICE,
    DREAM_ACTION_START_ANALYSIS,
    QUICK_ALREADY_ACTIVE,
)
from data.models import (
    Base,
    ExperienceLevel,
    LlmRole,
    Session,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
)
from data.repositories import (
    DreamRepository,
    SessionMessageRepository,
    SessionRepository,
    SessionStateRepository,
    UserRepository,
)


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Task 15: _resume_qa_for_dream теперь читает флаг фичи image questions.
    image_questions_enabled = True


@pytest_asyncio.fixture
async def schema(test_engine: AsyncEngine):
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def onboarded_user(test_session, schema):
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(
        user_id=user.id,
        profile_life_context="x",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    await test_session.commit()
    return user


def _message(text: str, user_id: int = 42) -> Message:
    msg = AsyncMock(spec=Message)
    msg.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    msg.chat = Chat(id=user_id, type="private")
    msg.text = text
    msg.answer = AsyncMock()
    return msg


def _make_state(user_id: int = 42) -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=user_id, user_id=user_id),
    )


@pytest.mark.asyncio
async def test_dreams_list_empty(test_session, onboarded_user) -> None:
    msg = _message("/dreams")
    await handle_dreams_list(msg, session=test_session, state=_make_state())
    msg.answer.assert_awaited_once()
    assert "пока нет" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_dreams_list_shows_recent_in_reverse_chronological(
    test_session, onboarded_user
) -> None:
    repo = DreamRepository(test_session)
    await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="первый сон, длинный текст про лес и фигуру в нём, всё это снится",
    )
    await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 12, tzinfo=UTC),
        transcription="второй сон короткий",
    )
    await test_session.commit()

    msg = _message("/dreams")
    await handle_dreams_list(msg, session=test_session, state=_make_state())

    # 1 header + 2 per-dream messages = 3 calls total
    assert msg.answer.await_count == 3
    all_calls = msg.answer.await_args_list
    # Первое сообщение — header
    header_text = all_calls[0].args[0]
    assert "2" in header_text  # count=2

    # Собираем тексты per-dream сообщений
    per_dream_texts = [c.args[0] for c in all_calls[1:]]
    combined = "\n".join(per_dream_texts)
    # Превью присутствуют
    assert "второй сон короткий" in combined
    assert "первый сон" in combined
    # Даты есть
    assert "12 мая" in combined
    assert "10 мая" in combined
    # Более свежий сон идёт первым (индекс 0 в per-dream)
    assert "12 мая" in per_dream_texts[0]
    assert "10 мая" in per_dream_texts[1]
    # Нет deeplink /dream_<id>
    for t in per_dream_texts:
        assert "/dream_" not in t


@pytest.mark.asyncio
async def test_dreams_list_does_not_leak_other_users_dreams(test_session, schema) -> None:
    repo = UserRepository(test_session)
    a = await repo.get_or_create_by_tg_id(telegram_user_id=1)
    b = await repo.get_or_create_by_tg_id(telegram_user_id=2)
    await repo.update_profile(
        user_id=a.id, profile_life_context="x", profile_experience_level=ExperienceLevel.NEWBIE
    )
    await repo.update_profile(
        user_id=b.id, profile_life_context="x", profile_experience_level=ExperienceLevel.NEWBIE
    )
    await test_session.commit()

    dr = DreamRepository(test_session)
    await dr.create(user_id=a.id, recorded_at=datetime.now(UTC), transcription="чужой сон")
    await test_session.commit()

    msg = _message("/dreams", user_id=2)
    await handle_dreams_list(msg, session=test_session, state=_make_state(user_id=2))
    sent = msg.answer.await_args.args[0]
    assert "чужой сон" not in sent
    assert "пока нет" in sent.lower()


@pytest.mark.asyncio
async def test_dream_card_recorded_no_analysis_shows_start_button(
    test_session, onboarded_user
) -> None:
    """Сон без разбора — кнопка «Разобрать сейчас»."""
    repo = DreamRepository(test_session)
    dream = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст",
    )
    await test_session.commit()

    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    msg.answer.assert_awaited()
    last_call = msg.answer.await_args_list[-1]
    kb = last_call.kwargs.get("reply_markup")
    assert kb is not None
    btns = [b for row in kb.inline_keyboard for b in row]
    assert btns[0].text == DREAM_ACTION_START_ANALYSIS
    assert btns[0].callback_data == f"start_analysis:{dream.id}"


@pytest.mark.asyncio
async def test_dream_card_resumes_stuck_image_questions_not_start_button(
    test_session, onboarded_user
) -> None:
    """Сон с паузой ПОСРЕДИ вопросов по образам (initial ещё нет) — карточка
    возобновляет вопросы, а НЕ предлагает «Разобрать сейчас» (иначе тап по ней
    упирается в QUICK_ALREADY_ACTIVE — тупик, из которого не выйти)."""
    from adapters.telegram.handlers.image_questions import IMGQ_STATE_KEY
    from adapters.telegram.states import QuickAnalysisStates

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон про голумов и балкон",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    await SessionStateRepository(test_session).upsert(session_id=sess.id, data={
        IMGQ_STATE_KEY: {"status": "answering", "idx": 0, "total": 2, "items": [
            {"name": "Голумы", "question": "Как вы относитесь к голумам в реальности?",
             "answer": None, "skipped": False},
            {"name": "Балкон", "question": "q2", "answer": None, "skipped": False},
        ]}})
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    state = _make_state()
    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(
        msg, session=test_session, state=state, settings=_NoLimitSettings()
    )

    # Возобновились прямо в вопросы по образам, а не в «Разобрать сейчас».
    assert (await state.get_state()) == \
        QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS.state
    all_sent = " || ".join(str(c.args[0]) for c in msg.answer.await_args_list if c.args)
    assert "голумам" in all_sent  # переспрошен текущий вопрос
    for c in msg.answer.await_args_list:  # нигде нет кнопки «Разобрать сейчас»
        kb = c.kwargs.get("reply_markup")
        if kb is not None and hasattr(kb, "inline_keyboard"):
            for row in kb.inline_keyboard:
                for b in row:
                    assert getattr(b, "callback_data", "") != f"start_analysis:{dream.id}"


@pytest.mark.asyncio
async def test_dream_with_analysis_auto_resumes_qa(
    test_session, onboarded_user
) -> None:
    """Сон с разбором — карточка + отдельным сообщением initial + FSM автоматически в IN_QA."""
    from data.constants import QUICK_PHASE_INITIAL

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст сна",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="МОЙ-IFS-РАЗБОР",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    state = _make_state()
    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=state, settings=_NoLimitSettings())

    # At least 2 calls: dream card + initial-разбор message
    assert msg.answer.await_count >= 2
    all_sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert "10 мая" in all_sent
    assert "МОЙ-IFS-РАЗБОР" in all_sent

    # FSM автоматически выставлен в IN_QA (auto-resume, без тапа кнопки)
    from adapters.telegram.states import QuickAnalysisStates

    assert (await state.get_state()) == QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_dream_card_with_analysis_includes_qa_pairs(
    test_session, onboarded_user
) -> None:
    """Если у сна были Q&A — они тоже шлются (после initial)."""
    from data.constants import QUICK_PHASE_INITIAL, QUICK_PHASE_QA

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="initial-разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=SessionMessageContentType.TEXT,
        content="USER-Q-1",
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="BOT-A-1",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_QA,
    )
    await sess_repo.pause(session_id=sess.id)
    await test_session.commit()

    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    all_sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert "initial-разбор" in all_sent
    assert "USER-Q-1" in all_sent
    assert "BOT-A-1" in all_sent


@pytest.mark.asyncio
async def test_dream_card_shows_inline_qa_not_reply_finish(
    test_session, onboarded_user
) -> None:
    """Карточка сна с разбором: последняя реплика несёт inline qa_options_keyboard
    (чипы из session_state + «Свой вопрос»/«Завершить»), reply-«Завершить» нет,
    FSM qa_options_msg_id выставлен."""
    from adapters.telegram.handlers.quick import _FSM_KEY_QA_MSG_ID, QA_OPTIONS_KEY
    from data.constants import QUICK_PHASE_INITIAL, QUICK_PHASE_QA

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст",
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    msg_repo = SessionMessageRepository(test_session)
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="initial-разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.USER_TO_BOT,
        content_type=SessionMessageContentType.TEXT,
        content="USER-Q-1",
    )
    await msg_repo.append(
        session_id=sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="BOT-A-1",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_QA,
    )
    await sess_repo.pause(session_id=sess.id)
    labels = ["Чип А", "Чип Б"]
    await SessionStateRepository(test_session).upsert(
        session_id=sess.id, data={QA_OPTIONS_KEY: {"labels": labels, "kind": "choice"}}
    )
    await test_session.commit()

    # answer() должен возвращать Message с реальным int message_id, чтобы код
    # мог сохранить qa_options_msg_id в FSM.
    msg = _message(f"/dream {dream.id}")
    counter = {"n": 0}

    async def _answer(*args, **kwargs):
        counter["n"] += 1
        reply = AsyncMock(spec=Message)
        reply.message_id = 1000 + counter["n"]
        return reply

    msg.answer.side_effect = _answer

    state = _make_state()
    await handle_dream_by_id(
        msg, session=test_session, state=state, settings=_NoLimitSettings()
    )

    # 1. Последняя отправка несёт inline qa_options_keyboard с нашими чипами.
    last_kwargs = msg.answer.await_args.kwargs
    kb = last_kwargs.get("reply_markup")
    assert isinstance(kb, InlineKeyboardMarkup), "last message must carry inline kb"
    flat = [btn for row in kb.inline_keyboard for btn in row]
    # Чипы: только callback_data вида qa:<digit(s)> (qa:0, qa:1, …).
    import re as _re
    chip_btns = [b for b in flat if b.callback_data and _re.fullmatch(r"qa:\d+", b.callback_data)]
    assert [b.text for b in chip_btns] == labels, "chip texts must match labels exactly"
    assert chip_btns[0].callback_data == "qa:0"
    assert chip_btns[1].callback_data == "qa:1"
    # Управляющие кнопки присутствуют отдельно.
    ctrl_cbs = {b.callback_data for b in flat if b.callback_data and not _re.fullmatch(r"qa:\d+", b.callback_data)}
    assert "qa:custom" in ctrl_cbs, "«Свой вопрос» button must be present"
    assert "qa:finish" in ctrl_cbs, "«Завершить» button must be present"

    # 2. Ни одно сообщение не несёт reply-«Завершить».
    for call in msg.answer.await_args_list:
        rk = call.kwargs.get("reply_markup")
        assert not isinstance(rk, ReplyKeyboardMarkup), "reply-«Завершить» must be gone"

    # 3. FSM qa_options_msg_id выставлен после показа карточки.
    data = await state.get_data()
    assert isinstance(data.get(_FSM_KEY_QA_MSG_ID), int)


@pytest.mark.asyncio
async def test_dream_card_with_active_session_shows_already_active(
    test_session, onboarded_user
) -> None:
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст",
    )
    sess_repo = SessionRepository(test_session)
    await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    await test_session.commit()

    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    sent = msg.answer.await_args.args[0]
    assert QUICK_ALREADY_ACTIVE in sent
    kwargs = msg.answer.await_args.kwargs
    assert kwargs.get("reply_markup") is None


@pytest.mark.asyncio
async def test_dream_by_id_returns_not_found_for_alien(test_session, onboarded_user) -> None:
    msg = _message("/dream 9999")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    msg.answer.assert_awaited_once()
    assert "не найден" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_dream_by_id_usage_when_no_arg(test_session, onboarded_user) -> None:
    msg = _message("/dream")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())
    msg.answer.assert_awaited_once()
    assert "использование" in msg.answer.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_dream_by_id_handles_deep_link_format(test_session, onboarded_user) -> None:
    """Поддержать `/dream_1` (формат из списка) так же, как `/dream 1`."""
    repo = DreamRepository(test_session)
    dream = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="...",
    )
    await test_session.commit()

    msg = _message(f"/dream_{dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    sent = msg.answer.await_args.args[0]
    assert "10 мая" in sent


@pytest.mark.asyncio
async def test_dream_card_escapes_html_in_transcription(test_session, onboarded_user) -> None:
    """HTML-символы в транскрипции экранируются перед отправкой."""
    repo = DreamRepository(test_session)
    dream = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="<script>alert(1)</script> и Kate & Co",
    )
    await test_session.commit()

    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())

    sent = msg.answer.await_args.args[0]
    assert "&lt;script&gt;" in sent
    assert "&amp;" in sent
    assert "<script>" not in sent
    assert "Kate & Co" not in sent


# ---------------------------------------------------------------------------
# Phase 5: resume via /dreams auto-pauses other active session (T5.3)
# ---------------------------------------------------------------------------


def _callback(data: str, user_id: int = 77777) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="Тест")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=user_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


@pytest.fixture
def fsm_77777() -> FSMContext:
    return FSMContext(
        storage=MemoryStorage(),
        key=StorageKey(bot_id=1, chat_id=77777, user_id=77777),
    )


@pytest.mark.asyncio
async def test_dream_card_does_not_show_obsolete_status_footer(
    test_session, onboarded_user
) -> None:
    """Регресс: убедиться что 'Статус: ещё не разбирался' не появляется на карточке."""
    repo = DreamRepository(test_session)
    dream = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст",
    )
    await test_session.commit()
    msg = _message(f"/dream {dream.id}")
    await handle_dream_by_id(msg, session=test_session, state=_make_state(), settings=_NoLimitSettings())
    all_sent = " || ".join(call.args[0] for call in msg.answer.await_args_list)
    assert "ещё не разбирался" not in all_sent
    assert "следующих версиях" not in all_sent


@pytest.mark.asyncio
async def test_resume_dream_pauses_other_active_session(
    test_session, schema, fsm_77777
) -> None:
    """User has IN_PROGRESS for dream_a + PAUSED for dream_b.
    Tap continue_quick on dream_b → dream_a auto-paused, dream_b resumed.
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=77777, display_name="Тест")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream_a = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон A"
    )
    dream_b = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон B"
    )
    sess_repo = SessionRepository(test_session)
    active_a = await sess_repo.create(
        dream_id=dream_a.id, user_id=user.id, type_=SessionType.QUICK
    )
    paused_b = await sess_repo.create(
        dream_id=dream_b.id, user_id=user.id, type_=SessionType.QUICK
    )
    await sess_repo.pause(session_id=paused_b.id)
    await test_session.commit()

    # Add a bot message to dream_b session so resume has something to show
    from adapters.telegram.session_log import QUICK_PHASE_INITIAL, log_bot_message
    await log_bot_message(
        test_session, session_id=paused_b.id, content="первичный разбор B", phase_label=QUICK_PHASE_INITIAL
    )
    await test_session.commit()

    cb = _callback(f"continue_quick:{dream_b.id}")
    await handle_continue_quick_callback(cb, fsm_77777, session=test_session, settings=_NoLimitSettings())

    refreshed_a = await test_session.get(Session, active_a.id)
    refreshed_b = await test_session.get(Session, paused_b.id)
    assert refreshed_a.status == SessionStatus.PAUSED, "Other active session should be paused"
    assert refreshed_b.status == SessionStatus.IN_PROGRESS, "Resumed session should be in progress"

    # User received AUTO_PAUSED_NOTICE
    calls = cb.message.answer.await_args_list
    texts = [c.args[0] if c.args else "" for c in calls]
    assert any(AUTO_PAUSED_NOTICE in t for t in texts), (
        f"Expected AUTO_PAUSED_NOTICE in one of: {texts!r}"
    )


# ---------------------------------------------------------------------------
# Task 9: /dreams per-dream inline-keyboard + dream_open callback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dreams_list_sends_header_plus_per_dream_with_keyboard(
    test_session, onboarded_user
) -> None:
    """1 header + N per-dream messages, каждое с inline-keyboard dream_open/dream_delete."""
    repo = DreamRepository(test_session)
    dream_a = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="сон А",
    )
    dream_b = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 12, tzinfo=UTC),
        transcription="сон Б",
    )
    await test_session.commit()

    msg = _message("/dreams")
    await handle_dreams_list(msg, session=test_session, state=_make_state())

    # 1 header + 2 per-dream = 3 messages total
    assert msg.answer.await_count == 3
    calls = msg.answer.await_args_list

    # Header message has no reply_markup
    assert calls[0].kwargs.get("reply_markup") is None

    # Per-dream messages have inline keyboards with correct callback data
    for idx, dream in enumerate([dream_b, dream_a], start=1):  # newest first
        kb = calls[idx].kwargs.get("reply_markup")
        assert kb is not None, f"Expected inline keyboard for call {idx}"
        cb_data = {b.callback_data for row in kb.inline_keyboard for b in row}
        assert f"dream_open:{dream.id}" in cb_data, f"Expected dream_open:{dream.id}"
        assert f"dream_delete:{dream.id}" in cb_data, f"Expected dream_delete:{dream.id}"

    # Body texts do NOT contain /dream_<id> deeplinks
    for call in calls[1:]:
        body = call.args[0]
        assert "/dream_" not in body


@pytest.mark.asyncio
async def test_dream_open_callback_shows_card(
    test_session, onboarded_user
) -> None:
    """dream_open:<id> callback → показывает карточку сна (вызывает _show_dream_by_id)."""
    repo = DreamRepository(test_session)
    dream = await repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 15, tzinfo=UTC),
        transcription="тест колбэк открытия",
    )
    await test_session.commit()

    cb = _callback(f"dream_open:{dream.id}", user_id=42)
    state = _make_state(user_id=42)
    await handle_dream_open_callback(cb, state=state, session=test_session, settings=_NoLimitSettings())

    # callback.answer() вызывается
    cb.answer.assert_awaited()

    # Карточка отправлена через cb.message.answer
    texts = [c.args[0] for c in cb.message.answer.await_args_list if c.args]
    combined = " ".join(texts)
    assert "15 мая" in combined
    assert "тест колбэк открытия" in combined


@pytest.mark.asyncio
async def test_dream_open_callback_not_found(
    test_session, onboarded_user
) -> None:
    """dream_open для несуществующего сна → DREAM_NOT_FOUND, без исключения."""
    cb = _callback("dream_open:99999", user_id=42)
    state = _make_state(user_id=42)
    await handle_dream_open_callback(cb, state=state, session=test_session, settings=_NoLimitSettings())

    cb.answer.assert_awaited()
    texts = [c.args[0] for c in cb.message.answer.await_args_list if c.args]
    assert any("не найден" in t.lower() for t in texts)


# ---------------------------------------------------------------------------
# Task 9 fix: fallback inline keyboard when load_dream_qa_for_display returns None
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_card_with_analysis_fallback_keyboard_when_display_load_fails(
    test_session, onboarded_user
) -> None:
    """Регресс-защита: dream_has_analysis=True, но load_dream_qa_for_display → None
    (редкое рассогласование: одна сессия с initial есть, но ПОСЛЕДНЯЯ PAUSED-сессия
    без initial → load возвращает None).

    Ожидаем:
    (1) юзер получает сообщение с InlineKeyboardMarkup, содержащей две управляющие
        кнопки qa:custom и qa:finish (без чипов — labels=[]);
    (2) ни одно сообщение НЕ несёт ReplyKeyboardMarkup;
    (3) FSM qa_options_msg_id выставлен в int.
    """
    import re as _re

    from adapters.telegram.handlers.quick import _FSM_KEY_QA_MSG_ID
    from adapters.telegram.states import QuickAnalysisStates
    from data.constants import QUICK_PHASE_INITIAL

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=onboarded_user.id,
        recorded_at=datetime(2026, 5, 10, tzinfo=UTC),
        transcription="текст сна",
    )
    sess_repo = SessionRepository(test_session)

    # Старая сессия: PAUSED + initial-сообщение → dream_has_analysis = True.
    old_sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    await SessionMessageRepository(test_session).append(
        session_id=old_sess.id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="Старый initial-разбор",
        llm_role_used=LlmRole.GENERATOR,
        llm_phase_label=QUICK_PHASE_INITIAL,
    )
    await sess_repo.pause(session_id=old_sess.id)

    # Новая сессия: PAUSED без initial-сообщений → load_dream_qa_for_display вернёт None,
    # потому что latest_inactive возьмёт эту новую сессию без initial.
    new_sess = await sess_repo.create(
        dream_id=dream.id, user_id=onboarded_user.id, type_=SessionType.QUICK
    )
    await sess_repo.pause(session_id=new_sess.id)

    await test_session.commit()

    # Убеждаемся в предусловиях теста.
    assert await dream_repo.dream_has_analysis(dream_id=dream.id), (
        "dream_has_analysis should be True (old session has initial msg)"
    )

    # Настраиваем answer() возвращать Messages с реальным message_id.
    msg = _message(f"/dream {dream.id}")
    counter = {"n": 0}

    async def _answer(*args, **kwargs):
        counter["n"] += 1
        reply = AsyncMock(spec=Message)
        reply.message_id = 2000 + counter["n"]
        return reply

    msg.answer.side_effect = _answer
    state = _make_state()

    await handle_dream_by_id(
        msg, session=test_session, state=state, settings=_NoLimitSettings()
    )

    # (1) Хотя бы одно сообщение несёт InlineKeyboardMarkup с qa:custom и qa:finish.
    inline_kbs = [
        call.kwargs.get("reply_markup")
        for call in msg.answer.await_args_list
        if isinstance(call.kwargs.get("reply_markup"), InlineKeyboardMarkup)
    ]
    assert inline_kbs, "Expected at least one message with InlineKeyboardMarkup"
    fallback_kb = inline_kbs[-1]
    flat = [btn for row in fallback_kb.inline_keyboard for btn in row]
    cbs = {b.callback_data for b in flat if b.callback_data}
    assert "qa:custom" in cbs, f"Expected qa:custom in {cbs}"
    assert "qa:finish" in cbs, f"Expected qa:finish in {cbs}"
    # Никаких чипов — labels=[].
    chip_cbs = [c for c in cbs if _re.fullmatch(r"qa:\d+", c)]
    assert chip_cbs == [], f"Expected no chip buttons, got {chip_cbs}"

    # (2) Ни одно сообщение НЕ несёт reply-«Завершить».
    for call in msg.answer.await_args_list:
        rk = call.kwargs.get("reply_markup")
        assert not isinstance(rk, ReplyKeyboardMarkup), "reply-«Завершить» must not appear"

    # (3) FSM qa_options_msg_id выставлен.
    data = await state.get_data()
    assert isinstance(data.get(_FSM_KEY_QA_MSG_ID), int), (
        f"Expected int qa_options_msg_id, got {data.get(_FSM_KEY_QA_MSG_ID)!r}"
    )

    # FSM должен быть в IN_QA после resume.
    assert (await state.get_state()) == QuickAnalysisStates.IN_QA.state
