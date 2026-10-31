"""Хук экстрактора памяти на паузе сессии (shadow-режим, Task 6).

`run_memory_extraction_on_pause` дергается после `SessionRepository.pause(...)`:
читает реплики юзера + per_session_context + существующие единицы, зовёт
`extract_memory_items` и применяет кандидатов через UserMemoryItemRepository.

Защиты:
- fire-and-forget: любой взрыв экстрактора НЕ бросает наружу (пауза уже состоялась);
- shadow-режим: пишем строки памяти, в промпты ничего не подставляем;
- zero-guard: usage-событие пишем только при ненулевых токенах;
- per_session_context берётся из реальной сессии (не пустышка).
"""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from sqlalchemy import select

from adapters.telegram.handlers._memory import run_memory_extraction_on_pause
from data.models import (
    Dream,
    LlmUsageEvent,
    Session,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
    User,
)
from data.repositories import SessionMessageRepository, UserMemoryItemRepository
from llm.memory_extract import (
    MEMORY_EXTRACT_VERSION,
    MemoryCandidate,
    MemoryExtractResult,
)

_EXTRACT_PATCH = "adapters.telegram.handlers._memory.extract_memory_items"


async def _make_user(session, tg_id: int = 100, life_context: str | None = None) -> User:
    user = User(telegram_user_id=tg_id, locale="ru", profile_life_context=life_context)
    session.add(user)
    await session.flush()
    return user


async def _make_session_with_qa(
    session,
    *,
    user_id: int,
    replies: list[tuple[str, str]] | None = None,
    per_session_context: str = "",
) -> Session:
    """Сон + quick-сессия (+ Q&A-пары бот→юзер) — как перед паузой."""
    dream = Dream(
        user_id=user_id,
        recorded_at=datetime.now(UTC),
        transcription="Мне снился вампир на чердаке",
    )
    session.add(dream)
    await session.flush()
    qa_session = Session(
        dream_id=dream.id,
        user_id=user_id,
        type=SessionType.QUICK,
        status=SessionStatus.PAUSED,
        per_session_context=per_session_context,
    )
    session.add(qa_session)
    await session.flush()
    msg_repo = SessionMessageRepository(session)
    for bot_q, user_a in replies or []:
        await msg_repo.append(
            session_id=qa_session.id,
            direction=SessionMessageDirection.BOT_TO_USER,
            content_type=SessionMessageContentType.TEXT,
            content=bot_q,
        )
        await msg_repo.append(
            session_id=qa_session.id,
            direction=SessionMessageDirection.USER_TO_BOT,
            content_type=SessionMessageContentType.TEXT,
            content=user_a,
        )
    return qa_session


def _result_one_association() -> MemoryExtractResult:
    return MemoryExtractResult(
        candidates=[
            MemoryCandidate(
                action="add",
                target_id=None,
                kind="attitude",
                text="К подросткам держится подальше",
                realm="association",
                entity_key=None,
                source="qa_grounding",
            )
        ],
        input_tokens=100,
        output_tokens=20,
    )


@pytest.mark.asyncio
async def test_pause_writes_memory_items(test_session, schema) -> None:
    """На паузе экстрактор пишет association-единицу с source_session_id + dream_id."""
    user = await _make_user(test_session, life_context="Эмигрант в Грузии.")
    sess = await _make_session_with_qa(
        test_session,
        user_id=user.id,
        replies=[
            ("Как вы в реальной жизни относитесь к подросткам?", "Держусь подальше, наглые."),
            ("Вампир во сне был каким?", "Носфератусом на чердаке."),
        ],
        per_session_context="Переехал в Белград, стресс.",
    )
    mock_extract = AsyncMock(return_value=_result_one_association())
    with patch(_EXTRACT_PATCH, new=mock_extract):
        await run_memory_extraction_on_pause(
            session=test_session,
            user_id=user.id,
            session_id=sess.id,
            dream_id=sess.dream_id,
            client=AsyncMock(),
            model="m",
        )

    items = await UserMemoryItemRepository(test_session).list_active(user_id=user.id)
    assert len(items) == 1
    assert items[0].realm == "association"
    assert items[0].kind == "attitude"
    assert items[0].source_session_id == sess.id
    assert items[0].dream_id == sess.dream_id
    # dream_content НЕ пишется — единственная строка именно association.
    assert all(i.realm != "dream_content" for i in items)

    # per_session_context подтянут из РЕАЛЬНОЙ сессии, а не пустышка.
    assert mock_extract.call_args.kwargs["per_session_context"] == "Переехал в Белград, стресс."
    # Реплики юзера дошли до экстрактора парами (вопрос бота, ответ юзера).
    passed_replies = mock_extract.call_args.kwargs["user_replies"]
    assert ("Вампир во сне был каким?", "Носфератусом на чердаке.") in passed_replies

    # Usage записан (zero-guard пройден: 100/20 токенов) с версией memory.extract.
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert events, "usage-события должны быть записаны при ненулевых токенах"
    assert all(e.prompt_version == MEMORY_EXTRACT_VERSION for e in events)


@pytest.mark.asyncio
async def test_confirm_and_supersede_applied(test_session, schema) -> None:
    """confirm бампает существующую единицу; supersede добавляет новую и гасит старую."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    confirmed = await mem_repo.add(
        user_id=user.id, kind="fact", text="Работает на двух работах",
        realm="real_life", source="qa_answer",
    )
    outdated = await mem_repo.add(
        user_id=user.id, kind="fact", text="Ищет работу",
        realm="real_life", source="qa_answer",
    )
    result = MemoryExtractResult(
        candidates=[
            MemoryCandidate("confirm", confirmed.id, "fact", "Работает на двух работах",
                            "real_life", None, "qa_answer"),
            MemoryCandidate("supersede", outdated.id, "fact", "Вышел на новую работу",
                            "real_life", None, "qa_answer"),
        ],
        input_tokens=50,
        output_tokens=10,
    )
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=result)):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )

    await test_session.refresh(confirmed)
    assert confirmed.confidence == 2
    await test_session.refresh(outdated)
    assert outdated.status == "superseded"
    active = await mem_repo.list_active(user_id=user.id)
    texts = {i.text for i in active}
    assert "Вышел на новую работу" in texts
    assert "Ищет работу" not in texts


@pytest.mark.asyncio
async def test_part_matches_existing_entity(test_session, schema) -> None:
    """Вторая сессия с той же стороной по entity_key → confirm существующей, не дубль."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    original = await mem_repo.add(
        user_id=user.id, kind="part", text="Обесценивающий критик",
        realm="real_life", source="qa_answer", entity_key="critic-devaluing",
    )
    result = MemoryExtractResult(
        candidates=[
            MemoryCandidate("add", None, "part", "Критик обесценивает выбранный путь",
                            "real_life", "critic-devaluing", "qa_answer"),
        ],
        input_tokens=50,
        output_tokens=10,
    )
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=result)):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )

    parts = [i for i in await mem_repo.list_active(user_id=user.id) if i.kind == "part"]
    assert len(parts) == 1  # не задвоилось
    assert parts[0].id == original.id
    assert parts[0].confidence == 2  # подтверждено (confirm бампнул confidence)
    assert parts[0].text == "Обесценивающий критик"  # текст исходной единицы сохранён


@pytest.mark.asyncio
async def test_part_with_new_entity_key_adds_row(test_session, schema) -> None:
    """Часть с НОВЫМ entity_key (совпадений нет) добавляется обычным add."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    await mem_repo.add(
        user_id=user.id, kind="part", text="Обесценивающий критик",
        realm="real_life", source="qa_answer", entity_key="critic-devaluing",
    )
    result = MemoryExtractResult(
        candidates=[
            MemoryCandidate("add", None, "part", "Ревизор: проверяет работу до придирок",
                            "real_life", "reviewer-work", "qa_answer"),
        ],
        input_tokens=50,
        output_tokens=10,
    )
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=result)):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )

    parts = [i for i in await mem_repo.list_active(user_id=user.id) if i.kind == "part"]
    assert {p.entity_key for p in parts} == {"critic-devaluing", "reviewer-work"}
    new = next(p for p in parts if p.entity_key == "reviewer-work")
    assert new.confidence == 1  # обычный add, без confirm
    assert new.source_session_id == sess.id


def _result_core(
    text: str = "Работает продакт-менеджером",
    key: str = "work",
    *,
    replaces_outdated: bool = False,
) -> MemoryExtractResult:
    return MemoryExtractResult(
        candidates=[
            MemoryCandidate(
                "add", None, "core", text, "real_life", key, "qa_answer",
                replaces_outdated=replaces_outdated,
            ),
        ],
        input_tokens=40,
        output_tokens=8,
    )


@pytest.mark.asyncio
async def test_core_added_when_no_existing(test_session, schema) -> None:
    """Новая категория → обычный add с провенансом сессии."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_result_core())):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    items = await UserMemoryItemRepository(test_session).list_active(user_id=user.id)
    core = [i for i in items if i.kind == "core"]
    assert len(core) == 1
    assert core[0].entity_key == "work"
    assert core[0].realm == "real_life"
    assert core[0].source_session_id == sess.id


@pytest.mark.asyncio
async def test_core_same_key_new_text_supersedes(test_session, schema) -> None:
    """Та же категория + флаг обновления → старая ячейка superseded со ссылкой."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    old = await mem_repo.add(
        user_id=user.id, kind="core", text="Ищет работу",
        realm="real_life", source="qa_answer", entity_key="work",
    )
    with patch(
        _EXTRACT_PATCH,
        new=AsyncMock(
            return_value=_result_core("Вышел на новую работу", replaces_outdated=True)
        ),
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    await test_session.refresh(old)
    assert old.status == "superseded"
    active = [
        i for i in await mem_repo.list_active(user_id=user.id) if i.kind == "core"
    ]
    assert len(active) == 1
    assert active[0].text == "Вышел на новую работу"
    assert old.superseded_by_id == active[0].id


@pytest.mark.asyncio
async def test_core_same_key_same_text_confirms(test_session, schema) -> None:
    """Дословный повтор (нормализованно) → confirm существующей, не дубль и не supersede."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    old = await mem_repo.add(
        user_id=user.id, kind="core", text="Работает продакт-менеджером",
        realm="real_life", source="qa_answer", entity_key="work",
    )
    with patch(
        _EXTRACT_PATCH,
        new=AsyncMock(return_value=_result_core("  работает  продакт-менеджером ")),
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    await test_session.refresh(old)
    assert old.status == "active"
    assert old.confidence == 2
    active = [
        i for i in await mem_repo.list_active(user_id=user.id) if i.kind == "core"
    ]
    assert len(active) == 1
    assert active[0].id == old.id


@pytest.mark.asyncio
async def test_core_same_key_new_text_without_flag_keeps_cell(test_session, schema) -> None:
    """Прод-дефект 2026-08-15: без флага обновления ядро НЕ разрушается.

    «соло-человек, учится опираться на людей» (work) не должно вытесняться вечером
    на даче. Информация не теряется: уезжает обычной ячейкой fact без категории.
    """
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    mem_repo = UserMemoryItemRepository(test_session)
    old = await mem_repo.add(
        user_id=user.id, kind="core",
        text="Юзер по натуре соло-человек, но начинает учиться опираться на людей",
        realm="real_life", source="qa_answer", entity_key="work",
    )
    with patch(
        _EXTRACT_PATCH,
        new=AsyncMock(return_value=_result_core("Юзер работает в саду")),
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    await test_session.refresh(old)
    assert old.status == "active"  # ядро цело
    assert old.superseded_by_id is None
    active = await mem_repo.list_active(user_id=user.id)
    core = [i for i in active if i.kind == "core"]
    assert len(core) == 1
    assert core[0].id == old.id
    assert core[0].confidence == 1  # это не подтверждение, а другой факт
    # Новая информация сохранена обычной ячейкой, без категории.
    spillover = [i for i in active if i.kind == "fact"]
    assert len(spillover) == 1
    assert spillover[0].text == "Юзер работает в саду"
    assert spillover[0].entity_key is None
    assert spillover[0].source_session_id == sess.id


@pytest.mark.asyncio
async def test_core_new_category_ignores_flag(test_session, schema) -> None:
    """Пустая ячейка категории: флаг обновления не влияет — обычный add в core."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_result_core())):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    items = await UserMemoryItemRepository(test_session).list_active(user_id=user.id)
    assert [(i.kind, i.entity_key) for i in items] == [("core", "work")]


@pytest.mark.asyncio
async def test_extraction_failure_does_not_raise(test_session, schema) -> None:
    """Fire-and-forget: взрыв экстрактора не бросает наружу."""
    with patch(_EXTRACT_PATCH, new=AsyncMock(side_effect=RuntimeError("boom"))):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=1, session_id=1,
            dream_id=1, client=AsyncMock(), model="m",
        )
    # ничего не упало — и ничего не записано
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert events == []


@pytest.mark.asyncio
async def test_failed_item_write_isolated_by_savepoint(test_session, schema) -> None:
    """Взрыв записи единицы (например truncation на Postgres) изолируется SAVEPOINT'ом.

    Наружу не бросает, общая транзакция запроса остаётся рабочей (сессией можно
    пользоваться дальше), а usage-строка вызова модели не теряется.
    """
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    with (
        patch(_EXTRACT_PATCH, new=AsyncMock(return_value=_result_one_association())),
        patch.object(
            UserMemoryItemRepository, "add",
            new=AsyncMock(side_effect=Exception("StringDataRightTruncation boom")),
        ),
    ):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )

    # Сессия жива после отката к SAVEPOINT: обычные запросы работают.
    items = await UserMemoryItemRepository(test_session).list_active(user_id=user.id)
    assert items == []  # сбойная единица не записана
    # Usage записан ВНЕ savepoint'а единиц — расход модели не потерян.
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert events, "usage должен быть записан несмотря на сбой записи единиц"
    assert all(e.prompt_version == MEMORY_EXTRACT_VERSION for e in events)


@pytest.mark.asyncio
async def test_no_usage_on_zero_tokens(test_session, schema) -> None:
    """Zero-guard: no-op экстрактора (tokens=0) не пишет usage-событий."""
    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)
    noop = MemoryExtractResult(candidates=[], input_tokens=0, output_tokens=0,
                               reject_reason="empty_input")
    with patch(_EXTRACT_PATCH, new=AsyncMock(return_value=noop)):
        await run_memory_extraction_on_pause(
            session=test_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m",
        )
    events = (await test_session.execute(select(LlmUsageEvent))).scalars().all()
    assert events == []
    assert await UserMemoryItemRepository(test_session).list_active(user_id=user.id) == []


@pytest.mark.asyncio
async def test_grounded_labels_include_answered_image_names(db_session, make_user) -> None:
    """Имена отвеченных образов фичи попадают в grounded_labels экстрактора."""
    user = await make_user()
    sess = await _make_session_with_qa(db_session, user_id=user.id, replies=[])
    from data.repositories import SessionStateRepository
    await SessionStateRepository(db_session).upsert(session_id=sess.id, data={
        "qa_map": {"grounded": ["Пустой город"]},
        "image_questions": {"status": "done", "idx": 2, "total": 2, "items": [
            {"name": "Отец", "question": "q", "answer": "тёплые", "skipped": False},
            {"name": "Тигр", "question": "q", "answer": None, "skipped": True},
        ]},
    })
    with patch(_EXTRACT_PATCH,
               new=AsyncMock(return_value=MemoryExtractResult(
                   candidates=[], input_tokens=0, output_tokens=0))) as extract:
        await run_memory_extraction_on_pause(
            session=db_session, user_id=user.id, session_id=sess.id,
            dream_id=sess.dream_id, client=AsyncMock(), model="m")
    labels = extract.call_args.kwargs["grounded_labels"]
    assert "Пустой город" in labels and "Отец" in labels
    assert "Тигр" not in labels  # скипнутый образ — не заземлён


@pytest.mark.asyncio
async def test_pause_session_wiring_respects_flag(test_session, schema) -> None:
    """_pause_session зовёт экстрактор только при флаге + клиенте; без них — нет."""
    from adapters.telegram.handlers.quick import _pause_session

    user = await _make_user(test_session)
    sess = await _make_session_with_qa(test_session, user_id=user.id)

    run_mock = AsyncMock()
    settings_on = SimpleNamespace(
        memory_write_enabled=True, openrouter_context_model="openai/gpt-4o-mini",
        memory_compact_enabled=True, memory_compact_budget_chars=1600,
    )
    with patch("adapters.telegram.handlers.quick.run_memory_extraction_on_pause", run_mock):
        await _pause_session(
            session=test_session, session_id=sess.id,
            user_id=user.id, dream_id=sess.dream_id,
            settings=settings_on, openrouter_classifier_client=AsyncMock(),
        )
    run_mock.assert_awaited_once()
    assert run_mock.call_args.kwargs["session_id"] == sess.id
    assert run_mock.call_args.kwargs["model"] == "openai/gpt-4o-mini"
    # Настройки уплотнения доезжают до хука, а не берутся из дефолтов модуля.
    assert run_mock.call_args.kwargs["compact_enabled"] is True
    assert run_mock.call_args.kwargs["compact_budget_chars"] == 1600

    # Флаг выключен → экстрактор не дергается (и пауза не падает).
    run_mock.reset_mock()
    settings_off = SimpleNamespace(
        memory_write_enabled=False, openrouter_context_model="openai/gpt-4o-mini",
        memory_compact_enabled=True, memory_compact_budget_chars=1600,
    )
    with patch("adapters.telegram.handlers.quick.run_memory_extraction_on_pause", run_mock):
        await _pause_session(
            session=test_session, session_id=sess.id,
            user_id=user.id, dream_id=sess.dream_id,
            settings=settings_off, openrouter_classifier_client=AsyncMock(),
        )
    run_mock.assert_not_awaited()

    # Клиент не прокинут (легаси-вызов) → тоже graceful skip.
    run_mock.reset_mock()
    with patch("adapters.telegram.handlers.quick.run_memory_extraction_on_pause", run_mock):
        await _pause_session(session=test_session, session_id=sess.id)
    run_mock.assert_not_awaited()
