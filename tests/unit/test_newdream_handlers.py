"""Unit tests for /newdream handler (Task D.3)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.newdream import (
    handle_newdream_command,
    handle_newdream_dream_input,
    handle_newdream_dream_loop,
)
from adapters.telegram.states import NewDreamStates, QuickAnalysisStates
from adapters.telegram.texts import (
    AUTO_PAUSED_NOTICE,
    DREAM_CONFIRM_ANALYZE_PROMPT,
    LOOP_ADD_BTN_DONE,
    LOOP_ADD_PROMPT_DREAM,
    NEWDREAM_PROMPT,
    NOT_ONBOARDED_PROMPT,
)
from data.models import DreamStatus, SessionStatus, SessionType
from data.repositories import DreamRepository, SessionRepository, UserRepository


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()


@pytest_asyncio.fixture
async def fsm_context_with_state() -> FSMContext:
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


def _intent_client(outcome: str) -> AsyncMock:
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f'{{"outcome": "{outcome}", "reason": "x"}}'
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=5),
        )
    )
    return client


_INTENT_MODEL = "openai/gpt-4o-mini"


# ---------------------------------------------------------------------------
# 1. /newdream — not onboarded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_command_not_onboarded_shows_prompt(
    test_session, schema, fsm_context_with_state
) -> None:
    """User NOT onboarded → answer is NOT_ONBOARDED_PROMPT, state unchanged."""
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()

    # User exists but onboarding_completed_at is None → not onboarded
    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == NOT_ONBOARDED_PROMPT
    # State must remain None (unchanged)
    assert await fsm_context_with_state.get_state() is None


@pytest.mark.asyncio
async def test_newdream_command_not_onboarded_user_not_found_shows_prompt(
    test_session, schema, fsm_context_with_state
) -> None:
    """User not in DB → answer is NOT_ONBOARDED_PROMPT, state unchanged."""
    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == NOT_ONBOARDED_PROMPT
    assert await fsm_context_with_state.get_state() is None


@pytest.mark.asyncio
async def test_newdream_during_onboarding_qa_completes_onboarding(
    test_session, schema, fsm_context_with_state
) -> None:
    """Регрессия: /newdream во время Q&A онбординга → онбординг завершается и
    переходим к сбору нового сна (а не NOT_ONBOARDED-тупик с советом /start)."""
    from datetime import UTC, datetime

    from aiogram import Bot

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    # НЕ onboarded
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime(2026, 6, 1, tzinfo=UTC), transcription="сон"
    )
    sess_repo = SessionRepository(test_session)
    sess = await sess_repo.create(
        dream_id=dream.id, user_id=user.id, type_=SessionType.QUICK,
        is_onboarding_session=True,
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(QuickAnalysisStates.IN_QA)
    await fsm_context_with_state.set_data({"dream_id": dream.id, "session_id": sess.id})

    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings(),
        bot=AsyncMock(spec=Bot),
    )

    await test_session.refresh(user)
    assert user.onboarding_completed_at is not None
    await test_session.refresh(sess)
    assert sess.status == SessionStatus.PAUSED
    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state


@pytest.mark.asyncio
async def test_discard_pending_newdream_deletes_dream_and_clears_state(
    test_session, schema, fsm_context_with_state
) -> None:
    """Выход командой из режима рассказа сна: недосказанный сон удаляется, FSM чистится."""
    from datetime import UTC, datetime

    from adapters.telegram.handlers._common import discard_pending_newdream

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime(2026, 6, 1, tzinfo=UTC),
        transcription="недосказанный сон",
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.set_data({"newdream_dream_id": dream.id})

    discarded = await discard_pending_newdream(
        state=fsm_context_with_state, session=test_session, telegram_user_id=42
    )
    assert discarded is True
    assert await fsm_context_with_state.get_state() is None
    assert await dream_repo.get_by_id(dream.id, user_id=user.id) is None


@pytest.mark.asyncio
async def test_discard_pending_newdream_noop_outside_newdream(
    test_session, schema, fsm_context_with_state
) -> None:
    """Вне NewDreamStates — no-op (ничего не трогаем)."""
    from adapters.telegram.handlers._common import discard_pending_newdream

    discarded = await discard_pending_newdream(
        state=fsm_context_with_state, session=test_session, telegram_user_id=42
    )
    assert discarded is False


# ---------------------------------------------------------------------------
# 2. /newdream — onboarded user starts flow
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_command_onboarded_starts_flow(
    test_session, schema, fsm_context_with_state
) -> None:
    """Onboarded user → state moves to ASKING_DREAM, bot sends NEWDREAM_PROMPT."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()

    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state
    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == NEWDREAM_PROMPT


# ---------------------------------------------------------------------------
# 2.5. /newdream auto-closes активную Quick-сессию (quick.v2.1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_auto_closes_active_quick_session(
    test_session, schema, fsm_context_with_state
) -> None:
    """Если у юзера в FSM висит активная Quick (IN_QA, ASKING_PER_SESSION_CONTEXT и т.д.) —
    /newdream должна её закрыть: session COMPLETED, dream ANALYZED, state.clear()."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    old_dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="прошлый сон"
    )
    sess_repo = SessionRepository(test_session)
    active_sess = await sess_repo.create(
        dream_id=old_dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    old_dream.status = DreamStatus.IN_ANALYSIS
    await test_session.commit()

    # Юзер уже залип в Quick IN_QA
    await fsm_context_with_state.set_state(QuickAnalysisStates.IN_QA)
    await fsm_context_with_state.set_data(
        {"dream_id": old_dream.id, "session_id": active_sess.id}
    )

    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    # Старая Quick-сессия закрыта
    await test_session.refresh(active_sess)
    assert active_sess.status == SessionStatus.PAUSED
    await test_session.refresh(old_dream)
    # dream.status stays IN_ANALYSIS when session is paused
    assert old_dream.status == DreamStatus.IN_ANALYSIS

    # state.clear() + переход в новое состояние
    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state
    # старые FSM-данные стёрты (state.clear() выполнен), и новые ещё не записаны
    data = await fsm_context_with_state.get_data()
    assert data == {}


@pytest.mark.parametrize(
    "state_enum",
    [
        QuickAnalysisStates.ASKING_PER_SESSION_CONTEXT,
        QuickAnalysisStates.IN_QA,
    ],
)
@pytest.mark.asyncio
async def test_newdream_auto_closes_across_all_quick_states(
    test_session, schema, fsm_context_with_state, state_enum
) -> None:
    """Auto-close работает на ВСЕХ состояниях `QuickAnalysisStates.*`, не только IN_QA.
    Логика триггера: state.startswith("QuickAnalysisStates")."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="К")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    old_dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="старый"
    )
    sess_repo = SessionRepository(test_session)
    active_sess = await sess_repo.create(
        dream_id=old_dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    old_dream.status = DreamStatus.IN_ANALYSIS
    await test_session.commit()

    await fsm_context_with_state.set_state(state_enum)
    await fsm_context_with_state.set_data(
        {"dream_id": old_dream.id, "session_id": active_sess.id}
    )

    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    await test_session.refresh(active_sess)
    assert active_sess.status == SessionStatus.PAUSED
    await test_session.refresh(old_dream)
    assert old_dream.status == DreamStatus.IN_ANALYSIS
    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state


@pytest.mark.asyncio
async def test_newdream_does_not_close_other_users_sessions(
    test_session, schema, fsm_context_with_state
) -> None:
    """Auto-close затрагивает только сессию текущего юзера, не других."""
    repo = UserRepository(test_session)
    me = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    other = await repo.get_or_create_by_tg_id(telegram_user_id=99, display_name="Other")
    await repo.mark_onboarded(user_id=me.id)
    await repo.mark_onboarded(user_id=other.id)
    dream_repo = DreamRepository(test_session)
    other_dream = await dream_repo.create(
        user_id=other.id, recorded_at=datetime.now(UTC), transcription="чужой сон"
    )
    sess_repo = SessionRepository(test_session)
    other_sess = await sess_repo.create(
        dream_id=other_dream.id, user_id=other.id, type_=SessionType.QUICK
    )
    await test_session.commit()

    # У МЕНЯ нет активной сессии, FSM пустой
    message = _make_message("/newdream", user_id=42)
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    # Чужая сессия не тронута
    await test_session.refresh(other_sess)
    assert other_sess.status == SessionStatus.IN_PROGRESS


# ---------------------------------------------------------------------------
# 3. ASKING_DREAM → Dream created → DREAM_LOOP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_dream_input_creates_dream_and_goes_to_loop(
    test_session, schema, fsm_context_with_state
) -> None:
    """Set state ASKING_DREAM, send text; verify Dream row created, state → DREAM_LOOP."""
    from unittest import mock

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.ASKING_DREAM)

    message = _make_message("сон про синий лес")
    bot = AsyncMock()
    openai_client = AsyncMock()

    with mock.patch(
        "adapters.telegram.handlers.onboarding._transcribe_message_voice",
        new_callable=AsyncMock,
    ):
        await handle_newdream_dream_input(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=bot,
            openai_client=openai_client,
            settings=_NoLimitSettings(),
        )

    assert await fsm_context_with_state.get_state() == NewDreamStates.DREAM_LOOP.state

    state_data = await fsm_context_with_state.get_data()
    dream_id = state_data.get("newdream_dream_id")
    assert dream_id is not None

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    assert dream is not None
    assert dream.transcription == "сон про синий лес"

    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_DREAM


# ---------------------------------------------------------------------------
# 4. DREAM_LOOP Done → Session created, Dream promoted, Quick FSM seeded
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_dream_loop_done_shows_confirmation_gate(
    test_session, schema, fsm_context_with_state
) -> None:
    """Done → confirmation gate: state=CONFIRMING_ANALYSIS, карточка + DREAM_CONFIRM_ANALYZE_PROMPT.
    Quick-сессия НЕ создаётся сразу (создаётся только после тапа «Разобрать»).

    Обновлён в Task #4: старое поведение (Done → ASKING_PER_SESSION_CONTEXT напрямую)
    заменено на confirmation gate. Тест тапа «Разобрать» — в test_newdream_confirmation_gate.py.
    """
    from sqlalchemy import select

    from data.models import Session

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="сон про синий лес",
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(newdream_dream_id=dream.id)

    message = _make_message(LOOP_ADD_BTN_DONE)
    await handle_newdream_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=_intent_client("continue"),
        openrouter_intent_model=_INTENT_MODEL,
    )

    # Confirmation gate показан, Quick-сессия ещё НЕ создана
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 0

    # Dream статус НЕ изменён (остался RECORDED до явного тапа «Разобрать»)
    await test_session.refresh(dream)
    assert dream.status == DreamStatus.RECORDED

    # State → CONFIRMING_ANALYSIS (не Quick!)
    assert (
        await fsm_context_with_state.get_state()
        == NewDreamStates.CONFIRMING_ANALYSIS.state
    )

    # FSM data сохраняет dream_id для следующего шага
    fsm_data = await fsm_context_with_state.get_data()
    assert fsm_data.get("newdream_dream_id") == dream.id

    # Бот отправил карточку + DREAM_CONFIRM_ANALYZE_PROMPT
    assert message.answer.await_count >= 2
    texts_sent = [call.args[0] for call in message.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_sent)


# ---------------------------------------------------------------------------
# 5. DREAM_LOOP non-DONE text → append + stay
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_dream_loop_text_appends_and_stays(
    test_session, schema, fsm_context_with_state
) -> None:
    """Любой не-DONE текст в DREAM_LOOP → append к transcription + stay in DREAM_LOOP."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=datetime.now(UTC),
        transcription="первая часть",
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(newdream_dream_id=dream.id)

    message = _make_message("вторая часть")
    await handle_newdream_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=_intent_client("continue"),
        openrouter_intent_model=_INTENT_MODEL,
    )

    # transcription appended in-memory
    assert dream.transcription == "первая часть\n\nвторая часть"
    assert await fsm_context_with_state.get_state() == NewDreamStates.DREAM_LOOP.state
    # answer вызван дважды: placeholder thinking_status + реальный ответ — проверяем последний
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_DREAM


@pytest.mark.asyncio
async def test_newdream_dream_loop_counter_question_answers_no_append(
    test_session, schema, fsm_context_with_state
) -> None:
    """Встречный вопрос в /newdream DREAM_LOOP → бот отвечает, сон НЕ дописывается."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="часть 1"
    )
    await test_session.commit()
    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(newdream_dream_id=dream.id)

    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"outcome": "question", "answer": "Не обязательно.", "reason": "q"}'
            ))],
            usage=SimpleNamespace(prompt_tokens=40, completion_tokens=6),
        )
    )
    message = _make_message("а надо ещё?")
    await handle_newdream_dream_loop(
        message, fsm_context_with_state, session=test_session,
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
    )

    assert dream.transcription == "часть 1"  # не дописано
    assert await fsm_context_with_state.get_state() == NewDreamStates.DREAM_LOOP.state
    texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("Не обязательно" in t for t in texts)


# ---------------------------------------------------------------------------
# «Это всё» — inline-кнопка в /newdream DREAM_LOOP
# ---------------------------------------------------------------------------


def _make_callback(data: str, user_id: int = 42):
    from aiogram.types import CallbackQuery

    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=user_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


@pytest.mark.asyncio
async def test_newdream_dream_loop_done_callback_shows_gate_and_extinguishes(
    test_session, schema, fsm_context_with_state
) -> None:
    """Тап inline «Это всё» → confirmation gate + кнопка погашена, классификатор не зван."""
    from adapters.telegram.handlers.newdream import handle_newdream_dream_loop_done_cb

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="сон про лес"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(
        newdream_dream_id=dream.id, loop_prompt_msg_id=999
    )

    cb = _make_callback("loop_done")
    bot = AsyncMock()
    client = _intent_client("continue")
    await handle_newdream_dream_loop_done_cb(
        cb,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
        bot=bot,
    )

    assert (
        await fsm_context_with_state.get_state()
        == NewDreamStates.CONFIRMING_ANALYSIS.state
    )
    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=999, reply_markup=None
    )
    texts_sent = [call.args[0] for call in cb.message.answer.call_args_list]
    assert any(DREAM_CONFIRM_ANALYZE_PROMPT in t for t in texts_sent)
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_newdream_dream_loop_text_extinguishes_previous_button(
    test_session, schema, fsm_context_with_state
) -> None:
    """Новое сообщение в loop гасит кнопку у предыдущего prompt'а."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="часть 1"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(NewDreamStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(
        newdream_dream_id=dream.id, loop_prompt_msg_id=444
    )

    message = _make_message("часть 2")
    bot = AsyncMock()
    await handle_newdream_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=_intent_client("continue"),
        openrouter_intent_model=_INTENT_MODEL,
        bot=bot,
    )

    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=444, reply_markup=None
    )


# ---------------------------------------------------------------------------
# 8. /newdream auto-pause IN_PROGRESS session via DB (Phase 5: no active FSM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_pauses_active_session_without_fsm_state(
    test_session, schema, fsm_context_with_state
) -> None:
    """If user has IN_PROGRESS session in DB but no active FSM state,
    /newdream pauses it and notifies with AUTO_PAUSED_NOTICE."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=66666, display_name="Тест")
    await repo.mark_onboarded(user_id=user.id)

    dream_repo = DreamRepository(test_session)
    dream1 = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="старый сон"
    )
    sess_repo = SessionRepository(test_session)
    active_sess = await sess_repo.create(
        dream_id=dream1.id, user_id=user.id, type_=SessionType.QUICK
    )
    dream1.status = DreamStatus.IN_ANALYSIS
    await test_session.commit()

    # FSM state is None (no active Quick FSM — stale/leaked or user typed /newdream cold)
    assert await fsm_context_with_state.get_state() is None

    message = _make_message("/newdream", user_id=user.telegram_user_id)
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    # Active session now PAUSED
    await test_session.refresh(active_sess)
    assert active_sess.status == SessionStatus.PAUSED

    # User received AUTO_PAUSED_NOTICE first, then standard newdream prompt
    calls = message.answer.await_args_list
    assert len(calls) >= 2, f"Expected ≥2 calls, got {len(calls)}"
    first_call_text = calls[0].args[0]
    assert first_call_text == AUTO_PAUSED_NOTICE

    # State moved to ASKING_DREAM
    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state


# ---------------------------------------------------------------------------
# 9. /newdream with active FSM session — single AUTO_PAUSED_NOTICE (Fix 1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_newdream_with_active_fsm_session_sends_single_notice(
    test_session, schema, fsm_context_with_state
) -> None:
    """FSM-path fires → DB-path skipped → only one AUTO_PAUSED_NOTICE sent.

    Regression guard for the double-trigger risk: when user is inside a
    QuickAnalysisStates, both the FSM-path and DB-path used to run in
    sequence. After the fix the FSM-path returns early, so the user sees
    exactly one notice message (plus the NEWDREAM_PROMPT), never two notices.
    """
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.mark_onboarded(user_id=user.id)
    dream_repo = DreamRepository(test_session)
    old_dream = await dream_repo.create(
        user_id=user.id, recorded_at=datetime.now(UTC), transcription="прошлый сон"
    )
    sess_repo = SessionRepository(test_session)
    active_sess = await sess_repo.create(
        dream_id=old_dream.id, user_id=user.id, type_=SessionType.QUICK
    )
    old_dream.status = DreamStatus.IN_ANALYSIS
    await test_session.commit()

    # User is inside Quick IN_QA with a matching IN_PROGRESS session in DB.
    await fsm_context_with_state.set_state(QuickAnalysisStates.IN_QA)
    await fsm_context_with_state.set_data(
        {"dream_id": old_dream.id, "session_id": active_sess.id}
    )

    message = _make_message("/newdream")
    await handle_newdream_command(
        message, fsm_context_with_state, session=test_session, settings=_NoLimitSettings()
    )

    # Exactly one AUTO_PAUSED_NOTICE — not two.
    notice_calls = [
        c for c in message.answer.call_args_list if AUTO_PAUSED_NOTICE in c.args[0]
    ]
    assert len(notice_calls) == 1, f"Expected 1 notice, got {len(notice_calls)}"

    # Exactly two total answer calls: notice + NEWDREAM_PROMPT.
    assert message.answer.call_count == 2, (
        f"Expected 2 answer calls, got {message.answer.call_count}: "
        f"{[c.args[0][:40] for c in message.answer.call_args_list]}"
    )

    # State transitioned correctly.
    assert await fsm_context_with_state.get_state() == NewDreamStates.ASKING_DREAM.state
