from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio
from aiogram.fsm.context import FSMContext
from aiogram.fsm.storage.base import StorageKey
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import CallbackQuery, Chat, Message
from aiogram.types import User as TgUser

from adapters.telegram.handlers.onboarding import (
    handle_life_context_input,
    handle_life_context_loop,
    handle_per_session_context_onb,
)
from adapters.telegram.states import OnboardingStates, QuickAnalysisStates
from adapters.telegram.texts import (
    ASK_LIFE_CONTEXT_ONB,
    ASK_PER_SESSION_CONTEXT,
    LOOP_ADD_BTN_DONE,
    LOOP_ADD_PROMPT_DREAM,
    LOOP_ADD_PROMPT_LIFE,
    QUICK_LLM_FAILED,
)
from data.models import (
    DreamStatus,
    LlmService,
    RequestBranch,
    Session,
    SessionType,
    UsageMetric,
)
from data.repositories import DreamRepository, UserRepository
from llm.generator import ErrorCategory


class _NoLimitSettings:
    """Spend-limit stub: limit so high the guard never trips."""

    user_spend_limit_usd = 1_000_000.0
    new_user_spend_limit_usd = 1_000_000.0
    spend_limit_exempt_set = frozenset()
    # Инжект памяти в промпты (Task 9) в этих тестах выключен.
    memory_inject_enabled = False
    # Флаг-офф: _build_quick_context (quick.v3.14) читает флаг фичи image questions.
    image_questions_enabled = False


@pytest_asyncio.fixture
async def fsm_context_with_state(request) -> FSMContext:
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
    message.answer = AsyncMock()
    return message


# ---------------------------------------------------------------------------
# Voice helpers
# ---------------------------------------------------------------------------


def _make_voice_message(user_id: int = 42, file_id: str = "voice_file_1") -> Message:
    message = AsyncMock(spec=Message)
    message.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    message.chat = Chat(id=user_id, type="private")
    message.text = None
    message.voice = SimpleNamespace(file_id=file_id, duration=5, mime_type="audio/ogg")
    message.answer = AsyncMock()
    return message


def _intent_client(outcome: str) -> AsyncMock:
    """AsyncOpenAI-мок, отдающий strict-JSON intent-классификатора."""
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
# Life context loop-add handlers (Task C.3)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_life_context_input_saves_text_and_goes_to_loop(
    test_session, schema, fsm_context_with_state
) -> None:
    from unittest import mock

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)

    message = _make_message("работа меняется, нервничаю")
    bot = AsyncMock()
    openai_client = AsyncMock()

    with mock.patch(
        "adapters.telegram.handlers.onboarding._transcribe_message_voice",
        new_callable=AsyncMock,
    ):
        await handle_life_context_input(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=bot,
            openai_client=openai_client,
            settings=_NoLimitSettings(),
        )

    user = await repo.get_by_tg_id(42)
    assert user.profile_life_context == "работа меняется, нервничаю"
    assert await fsm_context_with_state.get_state() == OnboardingStates.LIFE_CONTEXT_LOOP.state
    # answer вызван дважды: placeholder thinking_status + реальный ответ — проверяем последний
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_LIFE


@pytest.mark.asyncio
async def test_life_context_loop_done_goes_to_per_session(
    test_session, schema, fsm_context_with_state
) -> None:
    """После «Это всё» на знакомстве онбординг идёт к per-session вопросу перед разбором."""
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(
        user_id=(await repo.get_by_tg_id(42)).id,
        profile_life_context="работа меняется",
    )
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    message = _make_message(LOOP_ADD_BTN_DONE)
    client = _intent_client("continue")  # кнопка → классификатор НЕ вызывается
    await handle_life_context_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state
    )
    message.answer.assert_awaited_once()
    sent_text = message.answer.await_args.args[0]
    assert sent_text == ASK_PER_SESSION_CONTEXT
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_life_context_loop_classifier_done_advances(
    test_session, schema, fsm_context_with_state
) -> None:
    """Свободный текст-отказ («нет, всё») → классификатор done → идём к per-session вопросу."""
    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(
        user_id=(await repo.get_by_tg_id(42)).id, profile_life_context="работа"
    )
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    message = _make_message("нет, всё, больше нечего")
    client = _intent_client("done")
    await handle_life_context_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state
    )
    # отказной текст НЕ дописан в профиль
    user = await repo.get_by_tg_id(42)
    assert user.profile_life_context == "работа"
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_life_context_loop_non_done_text_appends_and_stays(
    test_session, schema, fsm_context_with_state
) -> None:
    """Любой не-DONE текст в LIFE_CONTEXT_LOOP → append + stay in LIFE_CONTEXT_LOOP."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context="первая часть")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    message = _make_message("вторая часть")
    client = _intent_client("continue")
    await handle_life_context_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    await test_session.refresh(user)
    assert user.profile_life_context == "первая часть\n\nвторая часть"
    assert await fsm_context_with_state.get_state() == OnboardingStates.LIFE_CONTEXT_LOOP.state
    # answer вызван дважды: placeholder thinking_status + реальный ответ — проверяем последний
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_LIFE


@pytest.mark.asyncio
async def test_life_context_input_voice_uses_transcript(
    test_session, schema, fsm_context_with_state
) -> None:
    from unittest import mock

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.ASKING_LIFE_CONTEXT)

    message = _make_voice_message()
    bot = AsyncMock()
    openai_client = AsyncMock()

    with mock.patch(
        "adapters.telegram.handlers.onboarding._transcribe_message_voice",
        new_callable=AsyncMock,
        return_value="transcribed life context",
    ):
        await handle_life_context_input(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=bot,
            openai_client=openai_client,
            settings=_NoLimitSettings(),
        )

    user = await repo.get_by_tg_id(42)
    assert user.profile_life_context == "transcribed life context"
    assert await fsm_context_with_state.get_state() == OnboardingStates.LIFE_CONTEXT_LOOP.state


# ---------------------------------------------------------------------------
# Dream full-loop handlers (Task C.4)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dream_input_creates_dream_and_goes_to_loop(
    test_session, schema, fsm_context_with_state
) -> None:
    from unittest import mock

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.ASKING_DREAM)

    from adapters.telegram.handlers.onboarding import handle_dream_input

    message = _make_message("приснилось, что я лечу")
    bot = AsyncMock()
    openai_client = AsyncMock()

    with mock.patch(
        "adapters.telegram.handlers.onboarding._transcribe_message_voice",
        new_callable=AsyncMock,
    ):
        await handle_dream_input(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=bot,
            openai_client=openai_client,
            settings=_NoLimitSettings(),
        )

    assert await fsm_context_with_state.get_state() == OnboardingStates.DREAM_LOOP.state
    state_data = await fsm_context_with_state.get_data()
    dream_id = state_data.get("onboarding_dream_id")
    assert dream_id is not None

    user = await repo.get_by_tg_id(42)
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.get_by_id(dream_id, user_id=user.id)
    assert dream is not None
    assert dream.transcription == "приснилось, что я лечу"

    # answer вызван дважды: placeholder thinking_status + реальный ответ — проверяем последний
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_DREAM


@pytest.mark.asyncio
async def test_dream_loop_done_goes_to_life_context(
    test_session, schema, fsm_context_with_state
) -> None:
    """После «Это всё» на рассказе сна онбординг идёт к шагу знакомства (рассказ о себе)."""
    from adapters.telegram.handlers.onboarding import handle_dream_loop

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.DREAM_LOOP)

    message = _make_message(LOOP_ADD_BTN_DONE)
    client = _intent_client("continue")  # кнопка → классификатор не зовём
    await handle_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_LIFE_CONTEXT.state
    )
    message.answer.assert_awaited_once()
    assert message.answer.await_args.args[0] == ASK_LIFE_CONTEXT_ONB
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_loop_classifier_done_advances(
    test_session, schema, fsm_context_with_state
) -> None:
    """Свободный текст «всё, хватит» в DREAM_LOOP → классификатор done → к знакомству."""
    from adapters.telegram.handlers.onboarding import handle_dream_loop

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.DREAM_LOOP)

    message = _make_message("всё, хватит")
    client = _intent_client("done")
    await handle_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_LIFE_CONTEXT.state
    )
    client.chat.completions.create.assert_awaited_once()


@pytest.mark.asyncio
async def test_dream_loop_non_done_text_appends_and_stays(
    test_session, schema, fsm_context_with_state
) -> None:
    """Любой не-DONE текст в DREAM_LOOP → append к transcription + stay in DREAM_LOOP."""
    from datetime import UTC
    from datetime import datetime as dt

    from adapters.telegram.handlers.onboarding import handle_dream_loop

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()

    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id,
        recorded_at=dt.now(UTC),
        transcription="первая часть",
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(OnboardingStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    message = _make_message("вторая часть")
    client = _intent_client("continue")
    await handle_dream_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
    )

    # transcription appended in-memory
    assert dream.transcription == "первая часть\n\nвторая часть"
    # state stays in DREAM_LOOP
    assert await fsm_context_with_state.get_state() == OnboardingStates.DREAM_LOOP.state
    # answer вызван дважды: placeholder thinking_status + реальный ответ — проверяем последний
    assert message.answer.await_args.args[0] == LOOP_ADD_PROMPT_DREAM


def _question_client(answer: str) -> AsyncMock:
    """Классификатор, отдающий outcome=question + answer (встречный вопрос)."""
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=(
                            '{"outcome": "question", "answer": "%s", "reason": "вопрос"}'  # noqa: UP031
                            % answer
                        )
                    )
                )
            ],
            usage=SimpleNamespace(prompt_tokens=60, completion_tokens=12),
        )
    )
    return client


@pytest.mark.asyncio
async def test_life_context_loop_uses_profile_thinking_statuses(
    test_session, schema, fsm_context_with_state
) -> None:
    """На шаге знакомства плейсхолдер — из PROFILE_THINKING_STATUSES (без упоминания снов)."""
    from adapters.telegram.texts import PROFILE_THINKING_STATUSES

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context="работа")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    message = _make_message("ещё я люблю горы")
    client = _intent_client("continue")
    await handle_life_context_loop(
        message, fsm_context_with_state, session=test_session,
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
    )

    # Первый answer — это thinking-плейсхолдер; он из «профильного» набора.
    placeholder_text = message.answer.await_args_list[0].args[0]
    assert placeholder_text in PROFILE_THINKING_STATUSES


@pytest.mark.asyncio
async def test_life_context_loop_counter_question_answers_and_keeps_state(
    test_session, schema, fsm_context_with_state
) -> None:
    """Встречный вопрос в LIFE_CONTEXT_LOOP → бот отвечает, контекст НЕ дописывается."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context="работа")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)

    message = _make_message("а надо?")
    client = _question_client("Не обязательно — добавляйте, если есть что.")
    await handle_life_context_loop(
        message, fsm_context_with_state, session=test_session,
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
    )

    # Контекст НЕ изменился (встречный вопрос не записан как контент)
    await test_session.refresh(user)
    assert user.profile_life_context == "работа"
    # Остались в том же loop
    assert await fsm_context_with_state.get_state() == OnboardingStates.LIFE_CONTEXT_LOOP.state
    # Ответ на вопрос отправлен + повторён prompt
    texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("Не обязательно" in t for t in texts)
    assert any(t == LOOP_ADD_PROMPT_LIFE for t in texts)


@pytest.mark.asyncio
async def test_dream_loop_counter_question_answers_and_keeps_state(
    test_session, schema, fsm_context_with_state
) -> None:
    """Встречный вопрос в DREAM_LOOP → бот отвечает, сон НЕ дописывается."""
    from datetime import UTC
    from datetime import datetime as dt

    from adapters.telegram.handlers.onboarding import handle_dream_loop

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription="первая часть"
    )
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    message = _make_message("а это обязательно?")
    client = _question_client("Не обязательно, добавьте если хотите.")
    await handle_dream_loop(
        message, fsm_context_with_state, session=test_session,
        openrouter_classifier_client=client, openrouter_intent_model=_INTENT_MODEL,
    )

    assert dream.transcription == "первая часть"  # не дописано
    assert await fsm_context_with_state.get_state() == OnboardingStates.DREAM_LOOP.state
    texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("Не обязательно" in t for t in texts)


# ---------------------------------------------------------------------------
# «Это всё» — inline-кнопка: тап = callback, гашение устаревшей кнопки
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_life_context_loop_done_callback_advances_and_extinguishes(
    test_session, schema, fsm_context_with_state
) -> None:
    """Тап inline «Это всё» в LIFE_CONTEXT_LOOP → ASKING_PER_SESSION_CONTEXT_ONB + кнопка погашена."""
    from adapters.telegram.handlers.onboarding import handle_life_context_loop_done_cb

    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context="работа")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
    await fsm_context_with_state.update_data(loop_prompt_msg_id=777)

    cb = _make_callback("loop_done")
    bot = AsyncMock()
    client = _intent_client("continue")  # классификатор НЕ должен вызываться

    await handle_life_context_loop_done_cb(
        cb,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
        bot=bot,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB.state
    )
    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=777, reply_markup=None
    )
    assert cb.message.answer.await_args.args[0] == ASK_PER_SESSION_CONTEXT
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_loop_done_callback_advances_and_extinguishes(
    test_session, schema, fsm_context_with_state
) -> None:
    """Тап inline «Это всё» в DREAM_LOOP → ASKING_LIFE_CONTEXT + кнопка погашена."""
    from adapters.telegram.handlers.onboarding import handle_dream_loop_done_cb

    repo = UserRepository(test_session)
    await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.DREAM_LOOP)
    await fsm_context_with_state.update_data(loop_prompt_msg_id=888)

    cb = _make_callback("loop_done")
    bot = AsyncMock()
    client = _intent_client("continue")

    await handle_dream_loop_done_cb(
        cb,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
        bot=bot,
    )

    assert (
        await fsm_context_with_state.get_state()
        == OnboardingStates.ASKING_LIFE_CONTEXT.state
    )
    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=888, reply_markup=None
    )
    client.chat.completions.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_life_context_loop_text_extinguishes_previous_button(
    test_session, schema, fsm_context_with_state
) -> None:
    """Новое сообщение в loop гасит кнопку у предыдущего prompt'а (нельзя тапнуть дважды)."""
    repo = UserRepository(test_session)
    user = await repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await repo.update_profile(user_id=user.id, profile_life_context="часть 1")
    await test_session.commit()
    await fsm_context_with_state.set_state(OnboardingStates.LIFE_CONTEXT_LOOP)
    await fsm_context_with_state.update_data(loop_prompt_msg_id=555)

    message = _make_message("часть 2")
    bot = AsyncMock()
    client = _intent_client("continue")
    await handle_life_context_loop(
        message,
        fsm_context_with_state,
        session=test_session,
        openrouter_classifier_client=client,
        openrouter_intent_model=_INTENT_MODEL,
        bot=bot,
    )

    bot.edit_message_reply_markup.assert_awaited_once_with(
        chat_id=42, message_id=555, reply_markup=None
    )


# ---------------------------------------------------------------------------
# Bridge to Quick from onboarding (Task C.5)
# ---------------------------------------------------------------------------


def _make_callback(data: str, user_id: int = 42) -> CallbackQuery:
    cb = AsyncMock(spec=CallbackQuery)
    cb.from_user = TgUser(id=user_id, is_bot=False, first_name="Катя")
    cb.data = data
    cb.message = AsyncMock(spec=Message)
    cb.message.chat = Chat(id=user_id, type="private")
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


@pytest.mark.asyncio
async def test_per_session_context_onb_runs_onboarding_quick(
    test_session, schema, fsm_context_with_state
) -> None:
    """Шаг ветки убран: per-session ответ в онбординге → дефолт UNCLEAR + сразу разбор → IN_QA."""
    from datetime import UTC
    from datetime import datetime as dt
    from unittest import mock

    from sqlalchemy import select

    from data.models import LlmUsageEvent

    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context="меняю работу")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription="мне снился тёмный лес"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    message = _make_message("вчера трудный день")
    bot = AsyncMock()
    openrouter = AsyncMock()
    fake_result = SimpleNamespace(
        text="quick output", input_tokens=100, output_tokens=50,
        error_category=None, model_used="google/gemini-3.5-flash",
    )

    with mock.patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        await handle_per_session_context_onb(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=bot,
            openai_client=AsyncMock(),
            openrouter_client=openrouter,
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    # Онбординг-Quick-сессия создана с дефолтной веткой (вопрос убран)
    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    new_sess = sessions[0]
    assert new_sess.is_onboarding_session is True
    assert new_sess.type == SessionType.QUICK
    assert new_sess.request_branch == RequestBranch.UNCLEAR
    assert new_sess.per_session_context == "вчера трудный день"
    assert new_sess.dream_id == dream.id

    await test_session.refresh(dream)
    assert dream.status == DreamStatus.IN_ANALYSIS

    assert await fsm_context_with_state.get_state() == QuickAnalysisStates.IN_QA.state
    fsm_data = await fsm_context_with_state.get_data()
    assert fsm_data.get("session_id") == new_sess.id
    assert fsm_data.get("dream_id") == dream.id

    # Хвостовой хинт QA_READY_AFTER_QUICK убран — разбор заканчивается на самом тексте
    sent_texts = [c.args[0] for c in message.answer.await_args_list]
    assert any("quick output" in t for t in sent_texts)

    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    or_events = [e for e in events if e.service == LlmService.OPENROUTER]
    assert len(or_events) == 2
    by_metric = {e.usage_metric: e for e in or_events}
    assert by_metric[UsageMetric.INPUT_TOKENS].usage_amount == 100
    assert by_metric[UsageMetric.OUTPUT_TOKENS].usage_amount == 50

    user_after = await user_repo.get_by_tg_id(user.telegram_user_id)
    assert user_after.onboarding_completed_at is None


@pytest.mark.asyncio
async def test_per_session_onb_llm_failure_shows_error(
    test_session, schema, fsm_context_with_state
) -> None:
    from datetime import UTC
    from datetime import datetime as dt
    from unittest import mock

    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context="меняю работу")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription="тёмный лес"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    message = _make_message("ctx")

    with mock.patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new_callable=AsyncMock,
        side_effect=RuntimeError("boom"),
    ):
        await handle_per_session_context_onb(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    answer_texts = [c.args[0] for c in message.answer.await_args_list]
    assert QUICK_LLM_FAILED in answer_texts


# ---------------------------------------------------------------------------
# Onboarding first-dream analysis → inline chips (Task 6 fix)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_onboarding_initial_analysis_inline_kb_no_tail(
    test_session, schema, fsm_context_with_state
) -> None:
    """Первый разбор в онбординге: хвост ===QA_OPTIONS=== → inline-клава, проза без хвоста в БД."""
    from datetime import UTC
    from datetime import datetime as dt
    from unittest import mock

    from aiogram.types import InlineKeyboardMarkup
    from sqlalchemy import select

    from adapters.telegram.session_log import load_quick_qa_history
    from data.models import Session as DbSession
    from llm.prompts.quick import QA_OPTIONS_SENTINEL

    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context="меняю работу")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription="мне снился тёмный лес"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    prose = "Глубокий разбор сна про лес.\n\n**Копнём глубже:**"
    raw = (
        f"{prose}\n\n"
        f"{QA_OPTIONS_SENTINEL}\n"
        '{"kind":"choice","options":["Линия страха","Линия опоры"]}'
    )
    fake_result = SimpleNamespace(
        text=raw,
        input_tokens=100,
        output_tokens=50,
        error_category=None,
        model_used="google/gemini-flash",
    )

    message = _make_message("вчера трудный день")
    message.answer = AsyncMock(return_value=SimpleNamespace(message_id=888))

    with mock.patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new_callable=AsyncMock,
        return_value=fake_result,
    ):
        await handle_per_session_context_onb(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    # 1. sent message has inline keyboard (not reply keyboard)
    send_call = message.answer.await_args
    assert send_call is not None
    markup = send_call.kwargs.get("reply_markup")
    assert isinstance(markup, InlineKeyboardMarkup), (
        f"Expected InlineKeyboardMarkup, got {type(markup)}"
    )
    flat_labels = [btn.text for row in markup.inline_keyboard for btn in row]
    assert "Линия страха" in flat_labels
    assert "Линия опоры" in flat_labels
    # qa:custom and qa:finish chips must be present
    assert any("custom" in btn.callback_data for row in markup.inline_keyboard for btn in row)
    assert any("finish" in btn.callback_data for row in markup.inline_keyboard for btn in row)

    # 2. sentinel NOT in sent text
    sent_text = send_call.args[0]
    assert QA_OPTIONS_SENTINEL not in sent_text

    # 3. DB history contains prose WITHOUT the tail
    sessions = (await test_session.execute(select(DbSession))).scalars().all()
    assert sessions, "session was not created"
    sess = sessions[0]
    initial_text, _qa_pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert initial_text is not None, "no initial bot message in DB history"
    assert QA_OPTIONS_SENTINEL not in initial_text
    assert initial_text.strip().startswith("Глубокий разбор")

    # 4. qa_options_msg_id must survive in FSM after delivery
    #    (set_data wipe-bug: if set_data runs AFTER _deliver_analysis, the key is wiped)
    fsm_data = await fsm_context_with_state.get_data()
    assert fsm_data.get("qa_options_msg_id") == 888, (
        f"qa_options_msg_id wiped from FSM — got {fsm_data!r}"
    )


@pytest.mark.asyncio
async def test_per_session_onb_content_filter_restarts_onboarding(
    test_session, schema, fsm_context_with_state
) -> None:
    """Все модели зарубили сон на онбординге → онбординг с шага «расскажите сон».

    Регрессия инцидента 2026-07-20: юзер читал «Что-то сломалось в коде бота»
    и оставался в состоянии без выхода.
    """
    from datetime import UTC
    from datetime import datetime as dt
    from unittest import mock

    from sqlalchemy import select

    from data.models import SessionStatus

    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(telegram_user_id=42, display_name="Катя")
    await user_repo.update_profile(user_id=user.id, profile_life_context="меняю работу")
    dream_repo = DreamRepository(test_session)
    dream = await dream_repo.create(
        user_id=user.id, recorded_at=dt.now(UTC), transcription="тяжёлый сон"
    )
    await test_session.commit()

    await fsm_context_with_state.set_state(OnboardingStates.ASKING_PER_SESSION_CONTEXT_ONB)
    await fsm_context_with_state.update_data(onboarding_dream_id=dream.id)

    message = _make_message("ctx")
    blocked = SimpleNamespace(
        text="", input_tokens=0, output_tokens=0,
        error_category=ErrorCategory.CONTENT_FILTER, model_used=None,
    )

    with mock.patch(
        "adapters.telegram.handlers.onboarding._run_generator",
        new_callable=AsyncMock,
        return_value=blocked,
    ):
        await handle_per_session_context_onb(
            message,
            fsm_context_with_state,
            session=test_session,
            bot=AsyncMock(),
            openai_client=AsyncMock(),
            openrouter_client=AsyncMock(),
            openrouter_classifier_client=AsyncMock(),
            settings=_NoLimitSettings(),
        )

    assert await fsm_context_with_state.get_state() == OnboardingStates.ASKING_DREAM.state

    sessions = (await test_session.execute(select(Session))).scalars().all()
    assert len(sessions) == 1
    await test_session.refresh(sessions[0])
    assert sessions[0].status == SessionStatus.PAUSED

    sent = " ".join(str(c.args[0]) for c in message.answer.await_args_list)
    assert "безопасный режим" in sent
    assert "расскажите сон другими словами" in sent
    assert "/newdream" not in sent
    assert "Здравствуйте" not in sent
    assert "сломалось в коде" not in sent
