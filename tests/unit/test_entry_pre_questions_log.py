"""Фаза pre_q: маркер для экстрактора памяти + фильтрация из Q&A-истории."""

import pytest

from adapters.telegram.session_log import (
    load_quick_qa_history,
    log_bot_message,
    log_user_message,
)
from data.constants import (
    GROUNDING_QUESTION_MARKER,
    PRE_QUESTION_MARKER,
    QUICK_PHASE_IMAGE_Q,
    QUICK_PHASE_INITIAL,
    QUICK_PHASE_PRE_Q,
)
from data.repositories import SessionMessageRepository


def test_constants():
    assert QUICK_PHASE_PRE_Q == "pre_q"
    assert PRE_QUESTION_MARKER != GROUNDING_QUESTION_MARKER
    assert PRE_QUESTION_MARKER.endswith(" ")


@pytest.mark.asyncio
async def test_pre_q_pairs_marked_for_memory(test_session, make_quick_session):
    sess = await make_quick_session()
    await log_bot_message(
        test_session, session_id=sess.id,
        content="Что вы почувствовали в тот момент?", phase_label=QUICK_PHASE_PRE_Q,
    )
    await log_user_message(test_session, session_id=sess.id, content="обиду и злость")
    pairs = await SessionMessageRepository(test_session).user_replies_with_prompts(
        session_id=sess.id
    )
    assert pairs == [
        (PRE_QUESTION_MARKER + "Что вы почувствовали в тот момент?", "обиду и злость")
    ]


@pytest.mark.asyncio
async def test_image_q_marker_unchanged(test_session, make_quick_session):
    sess = await make_quick_session()
    await log_bot_message(
        test_session, session_id=sess.id,
        content="Как вы относитесь к тиграм?", phase_label=QUICK_PHASE_IMAGE_Q,
    )
    await log_user_message(test_session, session_id=sess.id, content="боюсь их")
    pairs = await SessionMessageRepository(test_session).user_replies_with_prompts(
        session_id=sess.id
    )
    assert pairs == [(GROUNDING_QUESTION_MARKER + "Как вы относитесь к тиграм?", "боюсь их")]


@pytest.mark.asyncio
async def test_load_history_skips_pre_q_as_pre_initial(test_session, make_quick_session):
    sess = await make_quick_session()
    await log_bot_message(
        test_session, session_id=sess.id, content="Вопрос?", phase_label=QUICK_PHASE_PRE_Q
    )
    await log_user_message(test_session, session_id=sess.id, content="ответ до разбора")
    await log_bot_message(
        test_session, session_id=sess.id, content="РАЗБОР", phase_label=QUICK_PHASE_INITIAL
    )
    initial, qa_pairs = await load_quick_qa_history(test_session, session_id=sess.id)
    assert initial == "РАЗБОР"
    assert qa_pairs == []  # пре-вопросная пара НЕ считается Q&A-парой
