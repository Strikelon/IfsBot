"""Тесты долговременной памяти о юзере (user_memory_items)."""

from datetime import UTC, datetime

import pytest

from data.models import (
    Dream,
    Session,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionType,
    User,
)
from data.repositories import SessionMessageRepository, UserMemoryItemRepository


async def _make_user(session, tg_id: int = 100) -> User:
    user = User(telegram_user_id=tg_id, locale="ru")
    session.add(user)
    await session.commit()
    await session.refresh(user)
    return user


async def _make_session(session, user_id: int) -> Session:
    dream = Dream(
        user_id=user_id,
        recorded_at=datetime.now(UTC),
        transcription="Мне снился лес",
    )
    session.add(dream)
    await session.flush()
    qa_session = Session(dream_id=dream.id, user_id=user_id, type=SessionType.QUICK)
    session.add(qa_session)
    await session.flush()
    return qa_session


@pytest.mark.asyncio
async def test_memory_item_persists(test_session, schema) -> None:
    from data.models import UserMemoryItem

    user = await _make_user(test_session)
    item = UserMemoryItem(
        user_id=user.id,
        kind="fact",
        text="Работаю на двух работах",
        realm="real_life",
        source="qa_answer",
    )
    test_session.add(item)
    await test_session.flush()
    assert item.id is not None
    assert item.status == "active"
    assert item.confidence == 1


@pytest.mark.asyncio
async def test_add_and_list_active(test_session, schema) -> None:
    repo = UserMemoryItemRepository(test_session)
    user = await _make_user(test_session)
    a = await repo.add(
        user_id=user.id,
        kind="attitude",
        text="К подросткам держится подальше",
        realm="association",
        source="qa_grounding",
    )
    items = await repo.list_active(user_id=user.id)
    assert [i.id for i in items] == [a.id]


@pytest.mark.asyncio
async def test_confirm_bumps_confidence(test_session, schema) -> None:
    repo = UserMemoryItemRepository(test_session)
    user = await _make_user(test_session)
    a = await repo.add(
        user_id=user.id,
        kind="pattern",
        text="Действует под давлением",
        realm="real_life",
        source="qa_answer",
    )
    await repo.confirm(a.id)
    await test_session.refresh(a)
    assert a.confidence == 2
    assert a.last_confirmed_at is not None


@pytest.mark.asyncio
async def test_supersede_marks_old(test_session, schema) -> None:
    repo = UserMemoryItemRepository(test_session)
    user = await _make_user(test_session)
    old = await repo.add(
        user_id=user.id,
        kind="fact",
        text="Ищу работу",
        realm="real_life",
        source="qa_answer",
    )
    new = await repo.add(
        user_id=user.id,
        kind="fact",
        text="Вышел на новую работу",
        realm="real_life",
        source="qa_answer",
    )
    await repo.supersede(old.id, new.id)
    await test_session.refresh(old)
    assert old.status == "superseded"
    assert old.superseded_by_id == new.id
    assert [i.id for i in await repo.list_active(user_id=user.id)] == [new.id]


@pytest.mark.asyncio
async def test_find_by_entity(test_session, schema) -> None:
    repo = UserMemoryItemRepository(test_session)
    user = await _make_user(test_session)
    p = await repo.add(
        user_id=user.id,
        kind="part",
        text="Обесценивающий критик",
        realm="real_life",
        source="qa_answer",
        entity_key="critic-devaluing",
    )
    found = await repo.find_by_entity(user.id, "critic-devaluing")
    assert found is not None
    assert found.id == p.id
    assert await repo.find_by_entity(user.id, "nope") is None


@pytest.mark.asyncio
async def test_find_by_entity_kind_filter(test_session, schema) -> None:
    """fact и part с одним entity_key: kind="part" возвращает именно PART, не факт."""
    repo = UserMemoryItemRepository(test_session)
    user = await _make_user(test_session)
    fact = await repo.add(
        user_id=user.id,
        kind="fact",
        text="Часто критикует себя вслух",
        realm="real_life",
        source="qa_answer",
        entity_key="critic-devaluing",
    )
    part = await repo.add(
        user_id=user.id,
        kind="part",
        text="Обесценивающий критик",
        realm="real_life",
        source="qa_answer",
        entity_key="critic-devaluing",
    )
    found = await repo.find_by_entity(user.id, "critic-devaluing", kind="part")
    assert found is not None
    assert found.id == part.id
    assert found.id != fact.id
    # Без kind — прежнее поведение (любой active по ключу) сохраняется.
    assert await repo.find_by_entity(user.id, "critic-devaluing") is not None
    # kind без совпадений → None.
    assert await repo.find_by_entity(user.id, "critic-devaluing", kind="person") is None


@pytest.mark.asyncio
async def test_user_replies_with_prompts(test_session, schema) -> None:
    user = await _make_user(test_session)
    qa_session = await _make_session(test_session, user.id)
    repo = SessionMessageRepository(test_session)

    async def _append(direction: SessionMessageDirection, content: str) -> None:
        await repo.append(
            session_id=qa_session.id,
            direction=direction,
            content_type=SessionMessageContentType.TEXT,
            content=content,
        )

    # Реплика юзера ДО первой прозы бота → prompt=None
    await _append(SessionMessageDirection.USER_TO_BOT, "Мне снился лес")
    await _append(SessionMessageDirection.BOT_TO_USER, "Что вы чувствовали в лесу?")
    await _append(SessionMessageDirection.BOT_TO_USER, "Не спешите с ответом.")
    await _append(SessionMessageDirection.USER_TO_BOT, "Тревогу")
    await _append(SessionMessageDirection.BOT_TO_USER, "Где в жизни похожая тревога?")
    await _append(SessionMessageDirection.USER_TO_BOT, "На работе")

    pairs = await repo.user_replies_with_prompts(session_id=qa_session.id)
    assert pairs == [
        (None, "Мне снился лес"),
        ("Не спешите с ответом.", "Тревогу"),
        ("Где в жизни похожая тревога?", "На работе"),
    ]


@pytest.mark.asyncio
async def test_user_replies_with_prompts_marks_image_q_pairs(test_session, schema) -> None:
    """Пара после бот-сообщения фазы image_q помечается маркером; qa-пары — нет.

    Фикс дев-смоука: без фазы экстрактор принимал заземляющий ответ (про РЕАЛЬНУЮ
    жизнь) за разговор о сне и дропал его.
    """
    from data.constants import GROUNDING_QUESTION_MARKER, QUICK_PHASE_IMAGE_Q, QUICK_PHASE_QA

    user = await _make_user(test_session)
    qa_session = await _make_session(test_session, user.id)
    repo = SessionMessageRepository(test_session)

    async def _append(
        direction: SessionMessageDirection, content: str, phase: str | None = None
    ) -> None:
        await repo.append(
            session_id=qa_session.id,
            direction=direction,
            content_type=SessionMessageContentType.TEXT,
            content=content,
            llm_phase_label=phase,
        )

    imgq_q = "Что в вашей реальной жизни связано с образом одинокой девочки?"
    await _append(SessionMessageDirection.BOT_TO_USER, imgq_q, QUICK_PHASE_IMAGE_Q)
    await _append(SessionMessageDirection.USER_TO_BOT, "ассоциация с моей дочкой")
    await _append(SessionMessageDirection.BOT_TO_USER, "Где в жизни похожая тревога?", QUICK_PHASE_QA)
    await _append(SessionMessageDirection.USER_TO_BOT, "На работе")

    pairs = await repo.user_replies_with_prompts(session_id=qa_session.id)
    assert pairs == [
        (GROUNDING_QUESTION_MARKER + imgq_q, "ассоциация с моей дочкой"),
        ("Где в жизни похожая тревога?", "На работе"),
    ]


@pytest.mark.asyncio
async def test_user_replies_with_prompts_empty_session(test_session, schema) -> None:
    user = await _make_user(test_session)
    qa_session = await _make_session(test_session, user.id)
    repo = SessionMessageRepository(test_session)
    assert await repo.user_replies_with_prompts(session_id=qa_session.id) == []


@pytest.mark.asyncio
async def test_last_bot_message(test_session, schema) -> None:
    user = await _make_user(test_session)
    qa_session = await _make_session(test_session, user.id)
    repo = SessionMessageRepository(test_session)

    async def _append(direction: SessionMessageDirection, content: str) -> None:
        await repo.append(
            session_id=qa_session.id, direction=direction,
            content_type=SessionMessageContentType.TEXT, content=content,
        )

    # Пустая сессия → None
    assert await repo.last_bot_message(session_id=qa_session.id) is None

    await _append(SessionMessageDirection.BOT_TO_USER, "Что вам снилось?")
    await _append(SessionMessageDirection.USER_TO_BOT, "Лес")
    await _append(SessionMessageDirection.BOT_TO_USER, "Как вы в жизни относитесь к Олегу?")
    # Реплика юзера ПОСЛЕ вопроса не меняет «последнее сообщение бота»
    await _append(SessionMessageDirection.USER_TO_BOT, "Очень благодарен ему")

    assert await repo.last_bot_message(session_id=qa_session.id) == (
        "Как вы в жизни относитесь к Олегу?"
    )
