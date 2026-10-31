"""Клавиатуры и оркестратор фичи image questions."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event

from adapters.telegram.handlers.image_questions import (
    IMGQ_STATE_KEY,
    start_image_questions_flow,
)
from adapters.telegram.keyboards import (
    IMGQ_CALLBACK_NO,
    IMGQ_CALLBACK_SKIP,
    IMGQ_CALLBACK_SKIP_ALL,
    IMGQ_CALLBACK_YES,
    image_offer_keyboard,
    image_question_keyboard,
)
from adapters.telegram.states import QuickAnalysisStates
from adapters.telegram.texts import IMG_Q_ANSWER_BTN, IMG_Q_SKIP_ALL_BTN, QA_SKIP_BTN
from data.analytics_events import AnalyticsEventType
from data.models import (
    AnalyticsEvent,
    ExperienceLevel,
    LlmService,
    LlmUsageEvent,
    RequestBranch,
)
from data.repositories import SessionRepository, SessionStateRepository, UserRepository
from llm.intent import IntentResult
from llm.prompts.image_questions import IMAGE_QUESTIONS_VERSION


def _flat(kb):
    return [(b.text, b.callback_data) for row in kb.inline_keyboard for b in row]


def test_offer_keyboard_two_buttons():
    assert _flat(image_offer_keyboard()) == [
        (IMG_Q_ANSWER_BTN, IMGQ_CALLBACK_YES),
        (QA_SKIP_BTN, IMGQ_CALLBACK_NO),
    ]


def test_question_keyboard_single_image_no_skip_all():
    """При total==1 кнопки «Пропустить все вопросы» НЕТ (решение спеки §6.2)."""
    assert _flat(image_question_keyboard(total=1)) == [(QA_SKIP_BTN, IMGQ_CALLBACK_SKIP)]


def test_question_keyboard_multi_image_has_skip_all():
    assert _flat(image_question_keyboard(total=3)) == [
        (QA_SKIP_BTN, IMGQ_CALLBACK_SKIP),
        (IMG_Q_SKIP_ALL_BTN, IMGQ_CALLBACK_SKIP_ALL),
    ]


def test_offer_text_names_button_verbatim():
    """Текст приглашения дословно называет кнопку QA_SKIP_BTN (спека §6.1)."""
    from adapters.telegram.texts import IMG_Q_OFFER_TEXT
    assert f"«{QA_SKIP_BTN}»" in IMG_Q_OFFER_TEXT


def test_normalize_punctuation_public():
    from adapters.telegram.quick_markup import normalize_punctuation
    assert normalize_punctuation("тире — вот") == "тире – вот"
    assert normalize_punctuation("«кавычки»") == '"кавычки"'


@pytest.mark.asyncio
async def test_flag_off_regular_flow_byte_identical(db_session, make_user, make_quick_session,
                                                    fake_message, fsm_state, settings):
    """image_questions_enabled=False → _run_quick_after_context зовёт _run_initial_quick,
    ни одной imgq-функции, ни одного нового сообщения юзеру."""
    from adapters.telegram.handlers.quick import _run_quick_after_context

    settings.image_questions_enabled = False
    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id)
    await fsm_state.set_data({"session_id": sess.id, "dream_id": sess.dream_id})
    with patch("adapters.telegram.handlers.quick._run_initial_quick",
               new=AsyncMock()) as run_initial, \
         patch("adapters.telegram.handlers.image_questions.start_image_questions_flow",
               new=AsyncMock()) as flow:
        await _run_quick_after_context(
            message=fake_message(user), state=fsm_state, session=db_session,
            openrouter_client=AsyncMock(), context_client=AsyncMock(),
            context_model="m", settings=settings,
        )
    run_initial.assert_awaited_once()
    flow.assert_not_awaited()


@pytest.mark.asyncio
async def test_flag_on_regular_flow_starts_offer(db_session, make_user, make_quick_session,
                                                 fake_message, fsm_state, settings):
    settings.image_questions_enabled = True
    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id)
    await fsm_state.set_data({"session_id": sess.id, "dream_id": sess.dream_id})
    from adapters.telegram.handlers.quick import _run_quick_after_context
    with patch("adapters.telegram.handlers.quick._run_initial_quick",
               new=AsyncMock()) as run_initial, \
         patch("adapters.telegram.handlers.image_questions.start_image_questions_flow",
               new=AsyncMock()) as flow:
        await _run_quick_after_context(
            message=fake_message(user), state=fsm_state, session=db_session,
            openrouter_client=AsyncMock(), context_client=AsyncMock(),
            context_model="m", settings=settings,
        )
    flow.assert_awaited_once()
    run_initial.assert_not_awaited()


@pytest.mark.asyncio
async def test_offer_first_no_llm_and_state(db_session, make_user, make_quick_session,
                                            fake_message, fsm_state, settings):
    """Приглашение: ноль LLM-вызовов, status=offered, событие img_q_offered,
    FSM = OFFERING_IMAGE_QUESTIONS."""
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id)
    msg = fake_message(user)
    # create=True: generate_quick_response появится в модуле в Task 10-11 (хэндлер
    # согласия); Task 9 не импортирует его — offer-first гарантирует ноль вызовов.
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock(), create=True) as gen, \
         patch("adapters.telegram.handlers.image_questions.emit_append_event",
               new=AsyncMock()) as emit:
        await start_image_questions_flow(
            message=msg, state=fsm_state, session=db_session, user=user,
            session_id=sess.id, dream_id=sess.dream_id, is_onboarding=False,
            settings=settings, bot=None,
        )
    gen.assert_not_awaited()  # offer-first: до согласия — ноль затрат
    emit.assert_awaited_once_with(
        db_session, user_id=user.id, telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.IMG_Q_OFFERED,
        metadata={"session_id": sess.id, "is_onboarding": False,
                  "entry_type": "dream"},
    )
    data = await SessionStateRepository(db_session).get_data(session_id=sess.id)
    assert data[IMGQ_STATE_KEY]["status"] == "offered"
    assert await fsm_state.get_state() == QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS.state
    msg.answer.assert_awaited()  # приглашение отправлено (текст + inline-клава)


# ---------------------------------------------------------------------------
# Task 14: онбординг-поток за тем же флагом (+ funnel onb_img_q_offered)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_onboarding_flag_off_byte_identical(db_session, make_user, fake_message,
                                                  fsm_state, settings):
    """Флаг выключен → _start_onboarding_quick идёт прямо в _run_quick_for_onboarding."""
    from adapters.telegram.handlers.onboarding import _start_onboarding_quick

    settings.image_questions_enabled = False
    user = await make_user(onboarded=False, with_dream=True)
    with patch("adapters.telegram.handlers.onboarding._run_quick_for_onboarding",
               new=AsyncMock()) as run_onb, \
         patch("adapters.telegram.handlers.image_questions.start_image_questions_flow",
               new=AsyncMock()) as flow:
        await _start_onboarding_quick(
            message=fake_message(user), state=fsm_state, session=db_session, user=user,
            dream_id=user.last_dream_id, per_session_context="волновала работа",
            branch=RequestBranch.UNCLEAR, custom_text=None,
            openrouter_client=AsyncMock(), context_client=AsyncMock(),
            context_model="m", settings=settings,
        )
    run_onb.assert_awaited_once()
    flow.assert_not_awaited()


@pytest.mark.asyncio
async def test_onboarding_flag_on_offers_with_funnel(db_session, make_user, fake_message,
                                                     fsm_state, settings):
    """Флаг включен → приглашение; onb_img_q_offered идемпотентно записан."""
    from sqlalchemy import select

    from adapters.telegram.handlers.onboarding import _start_onboarding_quick

    settings.image_questions_enabled = True
    user = await make_user(onboarded=False, with_dream=True)
    with patch("adapters.telegram.handlers.onboarding._run_quick_for_onboarding",
               new=AsyncMock()) as run_onb:
        await _start_onboarding_quick(
            message=fake_message(user), state=fsm_state, session=db_session, user=user,
            dream_id=user.last_dream_id, per_session_context="волновала работа",
            branch=RequestBranch.UNCLEAR, custom_text=None,
            openrouter_client=AsyncMock(), context_client=AsyncMock(),
            context_model="m", settings=settings,
        )
    run_onb.assert_not_awaited()  # разбор НЕ стартовал — сначала приглашение
    assert await fsm_state.get_state() == \
        QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS.state
    rows = (await db_session.execute(select(AnalyticsEvent).where(
        AnalyticsEvent.event_type == "onb_img_q_offered"))).scalars().all()
    assert len(rows) == 1


# ---------------------------------------------------------------------------
# Task 10: offer-хэндлеры (кнопки / текст / голос) + выделение образов
# ---------------------------------------------------------------------------


def _gen_result(text: str):
    from llm.generator import GeneratorResult
    return GeneratorResult(text=text, input_tokens=1000, output_tokens=50,
                           model_used="google/gemini-3.5-flash")


_TWO_IMAGES_JSON = ('{"images":[{"name":"Отец","question":"Как отношения с отцом '
                    'в реальности?"},{"name":"Тигр","question":"Как вы относитесь '
                    'к тиграм в реальности?"}]}')


class _ImgqEnv:
    """Обвязка offer-хэндлеров: реальные db_session/FSM + AsyncMock-DI-клиенты.

    По образцу test_quick_handlers: message/callback — моки, хэндлеры вызываются
    напрямую с теми же именами DI-параметров, что инжектит workflow_data (main.py).
    """

    def __init__(self, *, db_session, user, sess, fsm_state, settings):
        self.db = db_session
        self.user = user
        self.session_id = sess.id
        self.dream_id = sess.dream_id
        self.fsm_state = fsm_state
        self.settings = settings
        self.answers: list[str] = []
        self._msg_counter = 1000
        self.bot = AsyncMock()
        self.openrouter_client = AsyncMock()
        self.classifier_client = AsyncMock()
        self.openai_client = AsyncMock()
        # Захват flushed-строк (identity map держит их weak — читаем в момент flush).
        self._events: list[tuple[str, dict | None]] = []
        self._usage: list[tuple[object, str | None]] = []

    # -- фабрики входящих апдейтов ------------------------------------------

    def _make_answer(self) -> AsyncMock:
        """message.answer, который пишет текст в self.answers и возвращает
        «отправленное сообщение» с int message_id (для FSM-запоминания) и
        awaitable edit_text/delete (для thinking_status)."""

        async def _answer(text=None, *args, **kwargs):
            self.answers.append(text)
            self._msg_counter += 1
            return SimpleNamespace(
                message_id=self._msg_counter,
                edit_text=AsyncMock(),
                delete=AsyncMock(),
            )

        return AsyncMock(side_effect=_answer)

    def make_message(self, *, text=None, voice=None):
        m = MagicMock()
        m.from_user = SimpleNamespace(
            id=self.user.telegram_user_id, is_bot=False, first_name="Катя"
        )
        m.chat = SimpleNamespace(id=self.user.telegram_user_id, type="private")
        m.text = text
        m.voice = voice
        m.answer = self._make_answer()
        return m

    def make_callback(self, data: str):
        cb = MagicMock()
        cb.from_user = SimpleNamespace(
            id=self.user.telegram_user_id, is_bot=False, first_name="Катя"
        )
        cb.data = data
        cb.message = self.make_message()
        cb.message.message_id = 500
        cb.answer = AsyncMock()
        return cb

    async def send_callback(self, data: str):
        from adapters.telegram.handlers.image_questions import handle_imgq_offer_callback

        cb = self.make_callback(data)
        await handle_imgq_offer_callback(
            cb, state=self.fsm_state, session=self.db, bot=self.bot,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            settings=self.settings,
        )
        return cb

    async def send_text(self, text: str):
        from adapters.telegram.handlers.image_questions import handle_imgq_offer_text

        m = self.make_message(text=text)
        await handle_imgq_offer_text(
            m, state=self.fsm_state, session=self.db, bot=self.bot,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            openrouter_intent_model="gpt-test", settings=self.settings,
        )
        return m

    async def send_voice(self, file_id: str = "v1"):
        from adapters.telegram.handlers.image_questions import handle_imgq_offer_voice

        m = self.make_message(
            voice=SimpleNamespace(file_id=file_id, duration=3, mime_type="audio/ogg")
        )
        await handle_imgq_offer_voice(
            m, state=self.fsm_state, session=self.db, bot=self.bot,
            openai_client=self.openai_client,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            openrouter_intent_model="gpt-test", settings=self.settings,
        )
        return m

    # -- аксессоры ------------------------------------------------------------

    async def imgq(self) -> dict:
        data = await SessionStateRepository(self.db).get_data(
            session_id=self.session_id
        ) or {}
        imgq = data.get(IMGQ_STATE_KEY)
        return imgq if isinstance(imgq, dict) else {}

    async def session_messages(self) -> list:
        """Все строки session_messages сессии (хронологически)."""
        from sqlalchemy import select

        from data.models import SessionMessage

        result = await self.db.execute(
            select(SessionMessage)
            .where(SessionMessage.session_id == self.session_id)
            .order_by(SessionMessage.id)
        )
        return list(result.scalars().all())

    def attach_flush_capture(self) -> None:
        """Слушатель after_flush: пишет вставленные analytics/usage-строки в списки.

        Хэндлеры не коммитят (это делает middleware), а identity map сессии держит
        объекты weak-ссылками — поэтому фиксируем данные в момент flush, синхронно.
        """

        def _capture(sync_session, _flush_context) -> None:
            for obj in sync_session.new:
                if isinstance(obj, AnalyticsEvent):
                    self._events.append((obj.event_type, obj.event_metadata))
                elif isinstance(obj, LlmUsageEvent):
                    self._usage.append((obj.service, obj.prompt_version))

        sa_event.listen(self.db.sync_session, "after_flush", _capture)
        self._detach = lambda: sa_event.remove(
            self.db.sync_session, "after_flush", _capture
        )

    def events_of(self, event_type: str) -> list[dict | None]:
        return [meta for etype, meta in self._events if etype == event_type]

    def usage_versions(self) -> list[str]:
        """prompt_version из llm_usage_events (уникальные, в порядке записи)."""
        out: list[str] = []
        for _service, version in self._usage:
            if version and version not in out:
                out.append(version)
        return out

    def whisper_events(self) -> list:
        return [
            (service, version) for service, version in self._usage
            if service == LlmService.WHISPER
        ]

    def last_answer_text(self) -> str:
        return self.answers[-1]

    def all_answer_texts(self) -> list[str]:
        return list(self.answers)


@pytest_asyncio.fixture
async def imgq_env(db_session, make_user, make_quick_session, fsm_state, settings):
    """Юзер (с профилем) + Quick-сессия (branch+context — для _build_quick_context),
    прогнанный start_image_questions_flow: status=offered, FSM=OFFERING."""
    user = await make_user(onboarded=True)
    await UserRepository(db_session).update_profile(
        user_id=user.id,
        profile_life_context="меняю работу",
        profile_experience_level=ExperienceLevel.NEWBIE,
    )
    sess = await make_quick_session(user_id=user.id)
    sess_repo = SessionRepository(db_session)
    await sess_repo.set_per_session_context(session_id=sess.id, context="тяжёлый день")
    await sess_repo.set_request_branch(session_id=sess.id, branch=RequestBranch.FEEL_STATE)
    await db_session.commit()

    env = _ImgqEnv(
        db_session=db_session, user=user, sess=sess, fsm_state=fsm_state,
        settings=settings,
    )
    offer_msg = env.make_message()
    await start_image_questions_flow(
        message=offer_msg, state=fsm_state, session=db_session, user=user,
        session_id=sess.id, dream_id=sess.dream_id, is_onboarding=False,
        settings=settings, bot=env.bot,
    )
    # Ответы/события setup-фазы (приглашение, img_q_offered) не попадают в assert'ы:
    # захват flush-строк включается только после старта, буфер ответов чистится.
    env.answers.clear()
    env.attach_flush_capture()
    yield env
    env._detach()


@pytest.mark.asyncio
async def test_decline_button_zero_extraction_calls(imgq_env):
    """«Не хочу отвечать» кнопкой: ноль вызовов выделения, status=declined,
    img_q_choice{decline,button}, старт разбора."""
    env = imgq_env  # фикстура: юзер+сессия в status=offered, FSM OFFERING, callback-фабрика
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock()) as gen, \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_callback(IMGQ_CALLBACK_NO)
    gen.assert_not_awaited()
    start.assert_awaited_once()
    assert (await env.imgq())["status"] == "declined"
    assert env.events_of("img_q_choice") == [
        {"session_id": env.session_id, "choice": "decline", "via": "button",
         "entry_type": "dream"}]


@pytest.mark.asyncio
async def test_yes_button_extracts_and_asks_first_question(imgq_env):
    env = imgq_env
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock(return_value=_gen_result(_TWO_IMAGES_JSON))):
        await env.send_callback(IMGQ_CALLBACK_YES)
    imgq = await env.imgq()
    assert imgq["status"] == "answering" and imgq["idx"] == 0 and imgq["total"] == 2
    assert [i["name"] for i in imgq["items"]] == ["Отец", "Тигр"]
    sent = env.last_answer_text()
    assert "Вопрос 1 из 2" in sent and "Отец" in sent
    assert await env.fsm_state.get_state() == \
        QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS.state
    # usage записан с версией фичи
    assert env.usage_versions() == [IMAGE_QUESTIONS_VERSION]


@pytest.mark.asyncio
async def test_single_image_no_counter_in_header(imgq_env):
    one = '{"images":[{"name":"Отец","question":"Как отношения с отцом?"}]}'
    env = imgq_env
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock(return_value=_gen_result(one))):
        await env.send_callback(IMGQ_CALLBACK_YES)
    assert "Вопрос 1" not in env.last_answer_text()  # счётчик скрыт при total==1


@pytest.mark.asyncio
async def test_zero_images_silent_fallback(imgq_env):
    """0 образов → молча разбор, status=no_images, событие с reason."""
    env = imgq_env
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock(return_value=_gen_result('{"images":[]}'))), \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_callback(IMGQ_CALLBACK_YES)
    start.assert_awaited_once()
    assert (await env.imgq())["status"] == "no_images"
    assert env.events_of("img_q_no_images") == [
        {"session_id": env.session_id, "reason": "empty", "entry_type": "dream"}]


@pytest.mark.asyncio
async def test_llm_crash_silent_fallback(imgq_env):
    env = imgq_env
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock(side_effect=RuntimeError("net"))), \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_callback(IMGQ_CALLBACK_YES)
    start.assert_awaited_once()
    assert env.events_of("img_q_no_images")[0]["reason"] == "llm_error"


@pytest.mark.asyncio
async def test_offer_text_question_reasks_with_buttons(imgq_env):
    """Встречный вопрос текстом: показать answer-поле, повторить приглашение,
    состояние не меняется, выделение не вызвано."""
    env = imgq_env
    intent = AsyncMock(return_value=IntentResult(
        outcome="question", input_tokens=10, output_tokens=5,
        answer="Вопросы сделают разбор точнее. Ответите?"))
    with patch("adapters.telegram.handlers.image_questions.classify_image_offer_intent",
               new=intent), \
         patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock()) as gen:
        await env.send_text("а зачем это?")
    gen.assert_not_awaited()
    assert await env.fsm_state.get_state() == \
        QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS.state
    texts = env.all_answer_texts()
    assert any("точнее" in t for t in texts)         # answer классификатора показан
    assert any("Можно я задам" in t for t in texts)  # приглашение повторено


@pytest.mark.asyncio
async def test_offer_text_unclear_reasks(imgq_env):
    env = imgq_env
    intent = AsyncMock(return_value=IntentResult("unclear", 10, 5))
    with patch("adapters.telegram.handlers.image_questions.classify_image_offer_intent",
               new=intent):
        await env.send_text("ммм")
    assert any("Не совсем понял" in t for t in env.all_answer_texts())


@pytest.mark.asyncio
async def test_offer_text_stale_after_status_advanced_no_reoffer(imgq_env):
    """Гонка: юзер написал текст ПОКА шло выделение (offer уже разрешён в answering).
    Устаревший offer-ввод НЕ должен переспрашивать приглашение и затирать answering;
    классификатор offer НЕ зовётся, вместо этого переспрашивается текущий вопрос."""
    from data.repositories import SessionStateRepository

    env = imgq_env
    await SessionStateRepository(env.db).upsert(session_id=env.session_id, data={
        IMGQ_STATE_KEY: {"status": "answering", "idx": 0, "total": 2, "items": [
            {"name": "Голумы", "question": "Как вы относитесь к голумам в реальности?",
             "answer": None, "skipped": False},
            {"name": "Город", "question": "q2", "answer": None, "skipped": False},
        ]}})
    intent = AsyncMock()
    with patch("adapters.telegram.handlers.image_questions.classify_image_offer_intent",
               new=intent):
        await env.send_text("frr")
    intent.assert_not_awaited()  # устаревший ввод не идёт в offer-классификатор
    imgq = await env.imgq()
    assert imgq["status"] == "answering" and imgq["idx"] == 0  # не затёрто
    texts = env.all_answer_texts()
    assert not any("Можно я задам" in t for t in texts)      # приглашение НЕ повторено
    assert not any("Не совсем понял" in t for t in texts)    # и не unclear
    assert any("голумам" in t for t in texts)                # переспрошен текущий вопрос


@pytest.mark.asyncio
async def test_offer_voice_decline_via_intent(imgq_env):
    """Голосовой отказ: STT → intent decline → status=declined, via=voice,
    whisper-usage записан, ноль вызовов выделения."""
    env = imgq_env
    stt = SimpleNamespace(text="не хочу, давай разбор", duration_seconds=3.0)
    intent = AsyncMock(return_value=IntentResult("decline", 10, 5))
    with patch("adapters.telegram.handlers.image_questions.transcribe_with_placeholder",
               new=AsyncMock(return_value=stt)), \
         patch("adapters.telegram.handlers.image_questions.classify_image_offer_intent",
               new=intent), \
         patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock()) as gen, \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_voice()
    gen.assert_not_awaited()
    start.assert_awaited_once()
    assert (await env.imgq())["status"] == "declined"
    assert env.events_of("img_q_choice") == [
        {"session_id": env.session_id, "choice": "decline", "via": "voice",
         "entry_type": "dream"}]
    assert len(env.whisper_events()) == 1


# ---------------------------------------------------------------------------
# Task 11: answer-хэндлеры (скипы, ответы с гардами, question-путь, paywall)
# ---------------------------------------------------------------------------


class _ImgqAnsweringEnv(_ImgqEnv):
    """Та же обвязка, но send_* роутят в хэндлеры ANSWERING_IMAGE_QUESTIONS."""

    async def send_callback(self, data: str):
        from adapters.telegram.handlers.image_questions import (
            handle_imgq_question_callback,
        )

        cb = self.make_callback(data)
        await handle_imgq_question_callback(
            cb, state=self.fsm_state, session=self.db, bot=self.bot,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            settings=self.settings,
        )
        return cb

    async def send_text(self, text: str):
        from adapters.telegram.handlers.image_questions import handle_imgq_answer_text

        m = self.make_message(text=text)
        await handle_imgq_answer_text(
            m, state=self.fsm_state, session=self.db, bot=self.bot,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            openrouter_intent_model="gpt-test", settings=self.settings,
        )
        return m

    async def send_voice(self, file_id: str = "v1"):
        from adapters.telegram.handlers.image_questions import handle_imgq_answer_voice

        m = self.make_message(
            voice=SimpleNamespace(file_id=file_id, duration=3, mime_type="audio/ogg")
        )
        await handle_imgq_answer_voice(
            m, state=self.fsm_state, session=self.db, bot=self.bot,
            openai_client=self.openai_client,
            openrouter_client=self.openrouter_client,
            openrouter_classifier_client=self.classifier_client,
            openrouter_intent_model="gpt-test", settings=self.settings,
        )
        return m


def _three_items() -> list[dict]:
    return [
        {"name": "Отец", "question": "Какие у вас отношения с отцом в реальности?",
         "answer": None, "skipped": False},
        {"name": "Тигр", "question": "Как вы относитесь к тиграм в реальности?",
         "answer": None, "skipped": False},
        {"name": "Вода", "question": "Что для вас вода в реальной жизни?",
         "answer": None, "skipped": False},
    ]


@pytest_asyncio.fixture
async def imgq_answering_env(db_session, make_user, make_quick_session, fsm_state,
                             settings):
    """Как imgq_env, но уже в answering: 3 образа, idx=0, FSM=ANSWERING."""
    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id)
    await SessionStateRepository(db_session).upsert(
        session_id=sess.id,
        data={IMGQ_STATE_KEY: {
            "status": "answering", "items": _three_items(), "idx": 0, "total": 3,
        }},
    )
    await db_session.commit()
    env = _ImgqAnsweringEnv(
        db_session=db_session, user=user, sess=sess, fsm_state=fsm_state,
        settings=settings,
    )
    await fsm_state.set_data({"session_id": sess.id, "dream_id": sess.dream_id})
    await fsm_state.set_state(QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS)
    env.attach_flush_capture()
    yield env
    env._detach()


@pytest.mark.asyncio
async def test_happy_path_three_answers_then_analysis(imgq_answering_env):
    """3 ответа → пары в session_messages (phase image_q), события, разбор."""
    env = imgq_answering_env
    intent = AsyncMock(return_value=IntentResult("answer", 10, 5))
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=intent), \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_text("отношения тёплые")
        await env.send_text("боюсь тигров")
        await env.send_text("вода успокаивает")
    start.assert_awaited_once()
    imgq = await env.imgq()
    assert imgq["status"] == "done"
    assert [i["answer"] for i in imgq["items"]] == \
        ["отношения тёплые", "боюсь тигров", "вода успокаивает"]
    assert len(env.events_of("img_q_answered")) == 3
    assert env.events_of("img_q_flow_done") == [
        {"session_id": env.session_id, "answered": 3, "skipped": 0, "total": 3,
         "entry_type": "dream"}]
    # Пары в session_messages: 3 бот-вопроса phase=image_q + 3 ответа юзера
    msgs = await env.session_messages()
    assert sum(1 for m in msgs if m.llm_phase_label == "image_q") == 3


@pytest.mark.asyncio
async def test_skip_button_advances_without_llm_and_log(imgq_answering_env):
    env = imgq_answering_env
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=AsyncMock()) as intent:
        await env.send_callback(IMGQ_CALLBACK_SKIP)
    intent.assert_not_awaited()
    imgq = await env.imgq()
    assert imgq["items"][0]["skipped"] is True and imgq["idx"] == 1
    assert env.events_of("img_q_skipped")[0]["via"] == "button"
    assert await env.session_messages() == []  # скипы не логируются


@pytest.mark.asyncio
async def test_skip_btn_exact_text_no_llm(imgq_answering_env):
    """Точный текст кнопки «Не хочу отвечать» текстом → skip БЕЗ LLM."""
    env = imgq_answering_env
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=AsyncMock()) as intent:
        await env.send_text(QA_SKIP_BTN)
    intent.assert_not_awaited()
    assert (await env.imgq())["items"][0]["skipped"] is True
    assert env.events_of("img_q_skipped")[0]["via"] == "text"


@pytest.mark.asyncio
async def test_skip_all_marks_remaining_and_starts_analysis(imgq_answering_env):
    env = imgq_answering_env
    intent = AsyncMock(return_value=IntentResult("answer", 10, 5))
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=intent), \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await env.send_text("ответ на первый")
        await env.send_callback(IMGQ_CALLBACK_SKIP_ALL)
    start.assert_awaited_once()
    imgq = await env.imgq()
    assert imgq["status"] == "done"
    assert [i["skipped"] for i in imgq["items"]] == [False, True, True]
    skipped = env.events_of("img_q_skipped")
    assert len(skipped) == 2 and all(e["via"] == "skip_all" for e in skipped)
    assert env.events_of("img_q_flow_done")[0]["answered"] == 1


@pytest.mark.asyncio
async def test_question_reasks_same_idx_no_log(imgq_answering_env):
    """Встречный вопрос: показать перефраз, тот же вопрос, idx на месте, пара НЕ логируется."""
    env = imgq_answering_env
    intent = AsyncMock(return_value=IntentResult(
        "question", 10, 5, answer="Я про реальную жизнь: как вам отец наяву?"))
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=intent):
        await env.send_text("в каком смысле?")
    imgq = await env.imgq()
    assert imgq["idx"] == 0 and imgq["items"][0]["answer"] is None
    assert await env.session_messages() == []
    assert env.events_of("img_q_answered") == []


@pytest.mark.asyncio
async def test_deliver_analysis_seeds_grounded_answered_only(db_session, make_user,
                                                             make_quick_session,
                                                             fake_message, fsm_state):
    """qa_map.grounded после _deliver_analysis содержит отвеченные и НЕ содержит скипнутые."""
    from adapters.telegram.handlers.image_questions import IMGQ_STATE_KEY
    from adapters.telegram.handlers.quick import QA_MAP_KEY, _deliver_analysis
    from data.repositories import SessionStateRepository

    user = await make_user()
    sess = await make_quick_session(user_id=user.id)
    repo = SessionStateRepository(db_session)
    await repo.upsert(session_id=sess.id, data={IMGQ_STATE_KEY: {
        "status": "done", "idx": 2, "total": 2,
        "items": [
            {"name": "Отец", "question": "q", "answer": "тёплые", "skipped": False},
            {"name": "Тигр", "question": "q", "answer": None, "skipped": True},
        ]}})
    raw = ('Разбор.\n===QA_OPTIONS===\n{"kind":"choice","options":["Линия А","Линия Б"],'
           '"map":{"focus":"f","type":"unknown","stage":"вступление","covered":[]}}')
    await _deliver_analysis(message=fake_message(user), state=fsm_state,
                            session=db_session, bot=None, session_id=sess.id,
                            raw_text=raw, phase_label="initial")
    qa_map = (await repo.get_data(session_id=sess.id))[QA_MAP_KEY]
    assert "Отец" in qa_map["grounded"]
    assert "Тигр" not in qa_map["grounded"]


# ---------------------------------------------------------------------------
# Task 15: базовое resume — уход посреди вопросов → тот же вопрос при возврате
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resume_answering_reasks_current_question(db_session, make_user,
                                                        make_quick_session, fake_message,
                                                        fsm_state, settings):
    """Пауза при answering → возврат переспрашивает items[idx], состояние ANSWERING."""
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await SessionStateRepository(db_session).upsert(session_id=sess.id, data={
        IMGQ_STATE_KEY: {"status": "answering", "idx": 1, "total": 3, "items": [
            {"name": "Отец", "question": "q1", "answer": "тёплые", "skipped": False},
            {"name": "Тигр", "question": "Как вы относитесь к тиграм?",
             "answer": None, "skipped": False},
            {"name": "Вода", "question": "q3", "answer": None, "skipped": False},
        ]}})
    msg = fake_message(user)
    await _resume_qa_for_dream(
        callback_message=msg, state=fsm_state, session=db_session, user_id=user.id,
        dream_id=sess.dream_id, auto_pause_other=False, intro_text=None,
        include_last_reply=True, source="test", settings=settings,
    )
    assert await fsm_state.get_state() == \
        QuickAnalysisStates.ANSWERING_IMAGE_QUESTIONS.state
    sent = " ".join(c.args[0] for c in msg.answer.await_args_list)
    assert "Вопрос 2 из 3" in sent and "тиграм" in sent  # тот же вопрос, не заново


@pytest.mark.asyncio
async def test_resume_with_initial_keeps_old_behavior(db_session, make_user,
                                                      make_quick_session, fake_message,
                                                      fsm_state, settings):
    """Сессия С initial → старое поведение IN_QA (фича не вмешивается)."""
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from adapters.telegram.session_log import QUICK_PHASE_INITIAL, log_bot_message

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await log_bot_message(db_session, session_id=sess.id, content="Разбор",
                          phase_label=QUICK_PHASE_INITIAL)
    await _resume_qa_for_dream(
        callback_message=fake_message(user), state=fsm_state, session=db_session,
        user_id=user.id, dream_id=sess.dream_id, auto_pause_other=False,
        intro_text=None, include_last_reply=False, source="test", settings=settings,
    )
    assert await fsm_state.get_state() == QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_paywall_before_final_analysis(imgq_answering_env):
    """Лимит перейдён к финишу → LIMIT_REACHED, пауза, разбор НЕ стартует (§7.8)."""
    env = imgq_answering_env
    intent = AsyncMock(return_value=IntentResult("answer", 10, 5))
    with patch("adapters.telegram.handlers.image_questions.classify_image_answer_intent",
               new=intent), \
         patch("adapters.telegram.handlers.image_questions.is_over_limit",
               new=AsyncMock(return_value=True)), \
         patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start, \
         patch("adapters.telegram.handlers.image_questions._pause_session_safe",
               new=AsyncMock()) as pause:
        await env.send_callback(IMGQ_CALLBACK_SKIP_ALL)
    start.assert_not_awaited()
    pause.assert_awaited_once()
    assert (await env.imgq())["status"] == "done"  # прогресс сохранён для resume


# ---------------------------------------------------------------------------
# Task 17: resume-края (offered / done без initial) + тихие устаревшие кнопки
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_callback():
    """CallbackQuery-мок для stale-теста: .data, .answer, .message.answer, .bot."""

    def _make(data: str):
        cb = MagicMock()
        cb.data = data
        cb.from_user = SimpleNamespace(id=42, is_bot=False, first_name="Катя")
        cb.answer = AsyncMock()
        cb.message = MagicMock()
        cb.message.chat = SimpleNamespace(id=42, type="private")
        cb.message.message_id = 500
        cb.message.answer = AsyncMock()
        cb.bot = MagicMock()
        cb.bot.edit_message_reply_markup = AsyncMock()
        return cb

    return _make


@pytest.mark.asyncio
async def test_resume_offered_reshows_invite(db_session, make_user, make_quick_session,
                                             fake_message, fsm_state, settings):
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await SessionStateRepository(db_session).upsert(
        session_id=sess.id, data={IMGQ_STATE_KEY: {"status": "offered"}})
    msg = fake_message(user)
    await _resume_qa_for_dream(
        callback_message=msg, state=fsm_state, session=db_session, user_id=user.id,
        dream_id=sess.dream_id, auto_pause_other=False, intro_text=None,
        include_last_reply=True, source="test", settings=settings,
    )
    assert await fsm_state.get_state() == \
        QuickAnalysisStates.OFFERING_IMAGE_QUESTIONS.state
    assert any("Можно я задам" in c.args[0] for c in msg.answer.await_args_list)


@pytest.mark.asyncio
async def test_resume_done_without_initial_starts_analysis(db_session, make_user,
                                                           make_quick_session,
                                                           fake_message, fsm_state,
                                                           settings):
    """done без initial (напр. paywall на финише) → deny_entry → сразу разбор."""
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await SessionStateRepository(db_session).upsert(session_id=sess.id, data={
        IMGQ_STATE_KEY: {"status": "done", "idx": 1, "total": 1, "items": [
            {"name": "Отец", "question": "q", "answer": "тёплые", "skipped": False}]}})
    with patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start, \
         patch("adapters.telegram.handlers.image_questions.deny_entry",
               new=AsyncMock(return_value=False)):
        await _resume_qa_for_dream(
            callback_message=fake_message(user), state=fsm_state, session=db_session,
            user_id=user.id, dream_id=sess.dream_id, auto_pause_other=False,
            intro_text=None, include_last_reply=True, source="test", settings=settings,
            openrouter_client=MagicMock(), openrouter_classifier_client=MagicMock(),
        )
    start.assert_awaited_once()


@pytest.mark.asyncio
async def test_resume_done_without_clients_falls_back_to_old_path(
        db_session, make_user, make_quick_session, fake_message, fsm_state, settings):
    """Легаси-вызов без клиентов: done-ветка возвращает False → старый путь IN_QA."""
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await SessionStateRepository(db_session).upsert(session_id=sess.id, data={
        IMGQ_STATE_KEY: {"status": "done", "idx": 1, "total": 1, "items": [
            {"name": "Отец", "question": "q", "answer": "тёплые", "skipped": False}]}})
    with patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start:
        await _resume_qa_for_dream(
            callback_message=fake_message(user), state=fsm_state, session=db_session,
            user_id=user.id, dream_id=sess.dream_id, auto_pause_other=False,
            intro_text=None, include_last_reply=False, source="test", settings=settings,
        )
    start.assert_not_awaited()
    assert await fsm_state.get_state() == QuickAnalysisStates.IN_QA.state


@pytest.mark.asyncio
async def test_resume_done_paywall_shows_paywall_only(db_session, make_user,
                                                      make_quick_session, fake_message,
                                                      fsm_state, settings):
    """deny_entry=True → пейвол показан, разбор НЕ стартует, статус не трогаем."""
    from adapters.telegram.handlers.quick import _resume_qa_for_dream
    from data.repositories import SessionStateRepository

    user = await make_user(onboarded=True)
    sess = await make_quick_session(user_id=user.id, status="paused")
    await SessionStateRepository(db_session).upsert(session_id=sess.id, data={
        IMGQ_STATE_KEY: {"status": "done", "idx": 1, "total": 1, "items": [
            {"name": "Отец", "question": "q", "answer": "тёплые", "skipped": False}]}})
    with patch("adapters.telegram.handlers.image_questions._start_analysis",
               new=AsyncMock()) as start, \
         patch("adapters.telegram.handlers.image_questions.deny_entry",
               new=AsyncMock(return_value=True)) as deny:
        await _resume_qa_for_dream(
            callback_message=fake_message(user), state=fsm_state, session=db_session,
            user_id=user.id, dream_id=sess.dream_id, auto_pause_other=False,
            intro_text=None, include_last_reply=True, source="test", settings=settings,
            openrouter_client=MagicMock(), openrouter_classifier_client=MagicMock(),
        )
    deny.assert_awaited_once()
    start.assert_not_awaited()
    data = await SessionStateRepository(db_session).get_data(session_id=sess.id)
    assert data[IMGQ_STATE_KEY]["status"] == "done"  # ответы не потеряны


@pytest.mark.asyncio
async def test_stale_imgq_callback_is_silent(db_session, fake_callback, fsm_state,
                                             settings):
    """imgq:yes в чужом состоянии (IN_QA) → тихий answer, никаких действий."""
    from adapters.telegram.handlers.image_questions import handle_imgq_stale_callback

    await fsm_state.set_state(QuickAnalysisStates.IN_QA)
    cb = fake_callback(IMGQ_CALLBACK_YES)
    with patch("adapters.telegram.handlers.image_questions.generate_quick_response",
               new=AsyncMock()) as gen:
        await handle_imgq_stale_callback(cb, fsm_state)
    gen.assert_not_awaited()
    cb.message.answer.assert_not_awaited()  # никаких сообщений — тихо


def test_stale_imgq_callback_registered_last():
    """Stale-хэндлер зарегистрирован ПОСЛЕДНИМ (без StateFilter) — иначе он
    перехватывал бы живые кнопки раньше state-bound хэндлеров."""
    from adapters.telegram.handlers.image_questions import (
        create_router,
        handle_imgq_stale_callback,
    )

    router = create_router()
    callbacks = [h.callback for h in router.callback_query.handlers]
    assert callbacks[-1] is handle_imgq_stale_callback
