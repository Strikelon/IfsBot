from datetime import UTC, datetime
from decimal import Decimal

import pytest

from admin.queries import subscriber_rows, user_card_summary
from data.models import (
    Dream,
    LlmService,
    LlmUsageEvent,
    Session,
    SessionMessage,
    SessionMessageContentType,
    SessionMessageDirection,
    SessionStatus,
    SessionType,
    UsageMetric,
    User,
)


def _service():
    return list(LlmService)[0]


def _metric():
    return list(UsageMetric)[0]


def _qa_msg(session_id: int) -> SessionMessage:
    """Реплика бота фазы qa — из них считается «Вопросов» в админке (B9)."""
    return SessionMessage(
        session_id=session_id,
        direction=SessionMessageDirection.BOT_TO_USER,
        content_type=SessionMessageContentType.TEXT,
        content="ответ", llm_phase_label="qa",
    )


async def _seed_entries(test_session, user, *, kind: str, dreams: int, questions: int):
    """Создать `dreams` обращений типа `kind` и `questions` qa-реплик по первому из них."""
    now = datetime.now(UTC)
    first = None
    for _ in range(dreams):
        d = Dream(user_id=user.id, recorded_at=now, transcription="x", kind=kind)
        test_session.add(d)
        await test_session.flush()
        first = first or d
    if questions and first is not None:
        sess = Session(dream_id=first.id, user_id=user.id, type=SessionType.QUICK,
                       status=SessionStatus.PAUSED)
        test_session.add(sess)
        await test_session.flush()
        test_session.add_all([_qa_msg(sess.id) for _ in range(questions)])
        await test_session.flush()


@pytest.mark.asyncio
async def test_counts_dreams_questions_and_cost(test_session):
    # Admin считает обращения агрегатом по dreams.kind (B9); cost — суммой live.
    u = User(telegram_user_id=1, display_name="Аня", is_active=True,
             last_seen_at=datetime(2026, 6, 8, tzinfo=UTC))
    test_session.add(u)
    await test_session.flush()
    await _seed_entries(test_session, u, kind="dream", dreams=1, questions=2)
    test_session.add_all([
        LlmUsageEvent(user_id=u.id, session_id=None, service=_service(), model="m",
                      usage_metric=_metric(), usage_amount=10, cost_usd=Decimal("0.0012")),
        LlmUsageEvent(user_id=u.id, session_id=None, service=_service(), model="m",
                      usage_metric=_metric(), usage_amount=10, cost_usd=Decimal("0.0030")),
    ])
    await test_session.flush()

    rows = await subscriber_rows(test_session, search=None, sort="created", page=1, per_page=50)
    assert rows.total == 1
    assert len(rows.items) == 1
    row = rows.items[0]
    assert row.dreams_count == 1
    assert row.questions_count == 2
    assert row.cost_usd == Decimal("0.0042")


@pytest.mark.asyncio
async def test_user_with_no_activity_has_zeros(test_session):
    u = User(telegram_user_id=2, display_name="Боб")
    test_session.add(u)
    await test_session.flush()
    rows = await subscriber_rows(test_session, search=None, sort="created", page=1, per_page=50)
    row = next(r for r in rows.items if r.telegram_user_id == 2)
    assert row.dreams_count == 0
    assert row.questions_count == 0
    assert row.cost_usd == Decimal("0")


@pytest.mark.asyncio
async def test_search_by_username_matches(test_session):
    test_session.add_all([
        User(telegram_user_id=1, display_name="Аня", username="anya"),
        User(telegram_user_id=2, display_name="Боб", username="bob"),
    ])
    await test_session.flush()
    rows = await subscriber_rows(test_session, search="@anya", sort="created", page=1, per_page=50)
    assert rows.total == 1
    assert rows.items[0].telegram_user_id == 1


@pytest.mark.asyncio
async def test_sort_by_cost_desc(test_session):
    u1 = User(telegram_user_id=1, display_name="A")
    u2 = User(telegram_user_id=2, display_name="B")
    test_session.add_all([u1, u2])
    await test_session.flush()
    test_session.add_all([
        LlmUsageEvent(user_id=u1.id, session_id=None, service=_service(), model="m",
                      usage_metric=_metric(), usage_amount=1, cost_usd=Decimal("0.10")),
        LlmUsageEvent(user_id=u2.id, session_id=None, service=_service(), model="m",
                      usage_metric=_metric(), usage_amount=1, cost_usd=Decimal("0.50")),
    ])
    await test_session.flush()
    rows = await subscriber_rows(test_session, search=None, sort="cost", page=1, per_page=50, descending=True)
    assert [r.telegram_user_id for r in rows.items] == [2, 1]


@pytest.mark.asyncio
async def test_counts_come_from_live_rows(test_session):
    # B9: счётчики считаются агрегатом по живым строкам, а не из users.dreams_total.
    u = User(telegram_user_id=1, display_name="A", dreams_total=999, qa_answers_total=999)
    test_session.add(u)
    await test_session.flush()
    await _seed_entries(test_session, u, kind="dream", dreams=5, questions=3)
    rows = await subscriber_rows(test_session, search=None, sort="created", page=1, per_page=50)
    row = rows.items[0]
    assert row.dreams_count == 5
    assert row.questions_count == 3


@pytest.mark.asyncio
async def test_rows_and_card_carry_access_tier(test_session):
    u = User(
        telegram_user_id=555,
        display_name="Ева",
        username="eva",
        is_active=True,
        access_tier="free_interview",
        access_tier_until=datetime(2026, 11, 1, tzinfo=UTC),
    )
    test_session.add(u)
    await test_session.flush()

    page = await subscriber_rows(test_session, search=None, sort="created", page=1, per_page=10)
    row = next(r for r in page.items if r.telegram_user_id == 555)
    assert row.access_tier == "free_interview"
    assert row.access_tier_until is not None

    card = await user_card_summary(test_session, telegram_user_id=555)
    assert card.access_tier == "free_interview"
    assert card.access_tier_until is not None


@pytest.mark.asyncio
async def test_counts_are_kind_aggregates(test_session):
    """Три пары счётчиков считаются раздельно по dreams.kind (спека B9)."""
    u = User(telegram_user_id=100, display_name="K")
    test_session.add(u)
    await test_session.flush()
    await _seed_entries(test_session, u, kind="dream", dreams=1, questions=1)
    await _seed_entries(test_session, u, kind="external_situation", dreams=1, questions=2)

    page = await subscriber_rows(
        test_session, search=None, sort="created", page=1, per_page=10
    )
    row = page.items[0]
    assert (row.dreams_count, row.questions_count) == (1, 1)
    assert (row.situations_count, row.situation_questions_count) == (1, 2)
    assert (row.conflicts_count, row.conflict_questions_count) == (0, 0)


@pytest.mark.asyncio
async def test_sort_by_situations(test_session):
    for tg, n_sit in ((201, 2), (202, 0)):
        u = User(telegram_user_id=tg, display_name=str(tg))
        test_session.add(u)
        await test_session.flush()
        await _seed_entries(
            test_session, u, kind="external_situation", dreams=n_sit, questions=0
        )
    page = await subscriber_rows(
        test_session, search=None, sort="situations", page=1, per_page=10
    )
    assert page.items[0].situations_count == 2


# --- Батч 2026-08-18: колонки «Пейвол» и «Интервью» ---


@pytest.mark.asyncio
async def test_rows_carry_paywall_and_interview(test_session):
    from datetime import UTC, datetime

    from admin.queries import subscriber_rows
    from data.models import User

    shown = datetime(2026, 8, 10, tzinfo=UTC)
    test_session.add(User(telegram_user_id=1001, display_name="A", paywall_shown_at=shown))
    test_session.add(User(telegram_user_id=1002, display_name="B"))
    await test_session.flush()

    page = await subscriber_rows(
        test_session, search=None, sort="paywall", page=1, per_page=50, descending=True
    )
    by_tg = {r.telegram_user_id: r for r in page.items}
    assert by_tg[1001].paywall_shown_at is not None
    assert by_tg[1002].paywall_shown_at is None
    assert by_tg[1002].interview_passed_at is None
    # сортировка реально работает: видевший пейвол — первым при desc
    assert page.items[0].telegram_user_id == 1001


@pytest.mark.asyncio
async def test_interview_sort_key_is_wired(test_session):
    from datetime import UTC, datetime

    from admin.queries import subscriber_rows
    from data.models import User

    test_session.add(User(telegram_user_id=1003, display_name="C"))
    test_session.add(
        User(
            telegram_user_id=1004,
            display_name="D",
            interview_passed_at=datetime(2026, 8, 5, tzinfo=UTC),
        )
    )
    await test_session.flush()
    page = await subscriber_rows(
        test_session, search=None, sort="interview", page=1, per_page=50, descending=True
    )
    assert page.items[0].telegram_user_id == 1004
