"""Фикс load_quick_qa_history: пре-initial пары image_q не создают фантомов."""
import pytest

from adapters.telegram.session_log import (
    QUICK_PHASE_IMAGE_Q,
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_QA,
    load_dream_qa_for_display,
    load_quick_qa_history,
    log_bot_message,
    log_user_message,
)


def test_phase_constant():
    assert QUICK_PHASE_IMAGE_Q == "image_q"


@pytest.mark.asyncio
async def test_pre_initial_pairs_do_not_create_phantoms(db_session, make_quick_session):
    """bot Q1 → user A1 → bot Q2 → user A2 → initial → user f1 → bot a1."""
    sess = await make_quick_session()
    await log_bot_message(db_session, session_id=sess.id, content="Вопрос про отца?",
                          phase_label=QUICK_PHASE_IMAGE_Q)
    await log_user_message(db_session, session_id=sess.id, content="Отношения тёплые")
    await log_bot_message(db_session, session_id=sess.id, content="Вопрос про тигра?",
                          phase_label=QUICK_PHASE_IMAGE_Q)
    await log_user_message(db_session, session_id=sess.id, content="Боюсь тигров")
    await log_bot_message(db_session, session_id=sess.id, content="Большой разбор сна",
                          phase_label=QUICK_PHASE_INITIAL)
    await log_user_message(db_session, session_id=sess.id, content="а почему тигр?")
    await log_bot_message(db_session, session_id=sess.id, content="Потому что…",
                          phase_label=QUICK_PHASE_QA)

    initial_text, qa_pairs = await load_quick_qa_history(db_session, session_id=sess.id)
    assert initial_text == "Большой разбор сна"
    # Ровно одна НАСТОЯЩАЯ пара; ни (A1, Q2)-фантома, ни висячего A2 в первой паре.
    assert qa_pairs == [("а почему тигр?", "Потому что…")]


@pytest.mark.asyncio
async def test_hanging_pre_initial_answer_not_glued(db_session, make_quick_session):
    """Пре-initial ответ юзера не липнет к первому QA-ответу бота."""
    sess = await make_quick_session()
    await log_bot_message(db_session, session_id=sess.id, content="Вопрос?",
                          phase_label=QUICK_PHASE_IMAGE_Q)
    await log_user_message(db_session, session_id=sess.id, content="ответ на образ")
    await log_bot_message(db_session, session_id=sess.id, content="Разбор",
                          phase_label=QUICK_PHASE_INITIAL)
    await log_bot_message(db_session, session_id=sess.id, content="QA-ответ",
                          phase_label=QUICK_PHASE_QA)
    initial_text, qa_pairs = await load_quick_qa_history(db_session, session_id=sess.id)
    assert initial_text == "Разбор"
    assert qa_pairs == []  # «ответ на образ» не спарился с «QA-ответ»


@pytest.mark.asyncio
async def test_regular_session_unchanged(db_session, make_quick_session):
    """Регресс: обычная сессия без фичи — поведение байт-в-байт."""
    sess = await make_quick_session()
    await log_bot_message(db_session, session_id=sess.id, content="Разбор",
                          phase_label=QUICK_PHASE_INITIAL)
    await log_user_message(db_session, session_id=sess.id, content="вопрос 1")
    await log_bot_message(db_session, session_id=sess.id, content="ответ 1",
                          phase_label=QUICK_PHASE_QA)
    initial_text, qa_pairs = await load_quick_qa_history(db_session, session_id=sess.id)
    assert initial_text == "Разбор"
    assert qa_pairs == [("вопрос 1", "ответ 1")]


@pytest.mark.asyncio
async def test_no_initial_returns_none_empty(db_session, make_quick_session):
    """Сессия без initial (пауза посреди вопросов) → (None, []) как раньше."""
    sess = await make_quick_session()
    await log_bot_message(db_session, session_id=sess.id, content="Вопрос?",
                          phase_label=QUICK_PHASE_IMAGE_Q)
    await log_user_message(db_session, session_id=sess.id, content="ответ")
    initial_text, qa_pairs = await load_quick_qa_history(db_session, session_id=sess.id)
    assert initial_text is None and qa_pairs == []


@pytest.mark.asyncio
async def test_dream_card_display_no_phantoms(db_session, make_quick_session):
    """load_dream_qa_for_display делегирует — фантомов нет и на карточке."""
    sess = await make_quick_session(status="paused")
    await log_bot_message(db_session, session_id=sess.id, content="Вопрос про отца?",
                          phase_label=QUICK_PHASE_IMAGE_Q)
    await log_user_message(db_session, session_id=sess.id, content="Тёплые")
    await log_bot_message(db_session, session_id=sess.id, content="Разбор",
                          phase_label=QUICK_PHASE_INITIAL)
    res = await load_dream_qa_for_display(db_session, dream_id=sess.dream_id)
    assert res is not None
    initial_text, qa_pairs = res
    assert initial_text == "Разбор" and qa_pairs == []
