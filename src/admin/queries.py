from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

from sqlalchemy import Select, case, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import (
    Dream,
    LlmUsageEvent,
    Session,
    SessionMessage,
    SessionMessageDirection,
    User,
)


@dataclass(frozen=True)
class SubscriberRow:
    user_id: int
    telegram_user_id: int
    display_name: str | None
    username: str | None
    dreams_count: int
    questions_count: int
    cost_usd: Decimal
    last_seen_at: object | None
    created_at: object
    is_active: bool
    situations_count: int = 0
    situation_questions_count: int = 0
    conflicts_count: int = 0
    conflict_questions_count: int = 0
    access_tier: str | None = None
    access_tier_until: object | None = None
    # Батч 2026-08-18: колонки «Пейвол» и «Интервью» + значки у аватара.
    paywall_shown_at: object | None = None
    interview_passed_at: object | None = None


@dataclass(frozen=True)
class Page:
    items: list[SubscriberRow]
    total: int
    page: int
    per_page: int


_SORT_COLUMNS = {
    "dreams": "dreams_count",
    "questions": "questions_count",
    "situations": "situations_count",
    "conflicts": "conflicts_count",
    "cost": "cost_usd",
    "last_seen": "last_seen_at",
    "created": "created_at",
    "status": "is_active",
    "paywall": "paywall_shown_at",
    "interview": "interview_passed_at",
}


def _base_query() -> Select:
    cost_sub = (
        select(
            LlmUsageEvent.user_id,
            func.coalesce(func.sum(LlmUsageEvent.cost_usd), 0).label("cost_usd"),
        )
        .group_by(LlmUsageEvent.user_id)
        .subquery()
    )
    # Агрегаты обращений по типам (B9). Денормализованные users.dreams_total /
    # qa_answers_total больше НЕ читаются (в БД остаются нетронутыми).
    entry_counts = (
        select(
            Dream.user_id.label("user_id"),
            func.sum(case((Dream.kind == "dream", 1), else_=0)).label("dreams_n"),
            func.sum(
                case((Dream.kind == "external_situation", 1), else_=0)
            ).label("situations_n"),
            func.sum(
                case((Dream.kind == "internal_conflict", 1), else_=0)
            ).label("conflicts_n"),
        )
        .group_by(Dream.user_id)
        .subquery()
    )
    q_counts = (
        select(
            Session.user_id.label("user_id"),
            func.sum(case((Dream.kind == "dream", 1), else_=0)).label("dream_q_n"),
            func.sum(
                case((Dream.kind == "external_situation", 1), else_=0)
            ).label("situation_q_n"),
            func.sum(
                case((Dream.kind == "internal_conflict", 1), else_=0)
            ).label("conflict_q_n"),
        )
        .select_from(SessionMessage)
        .join(Session, SessionMessage.session_id == Session.id)
        .join(Dream, Session.dream_id == Dream.id)
        .where(
            SessionMessage.direction == SessionMessageDirection.BOT_TO_USER,
            SessionMessage.llm_phase_label == "qa",
        )
        .group_by(Session.user_id)
        .subquery()
    )
    return (
        select(
            User.id,
            User.telegram_user_id,
            User.display_name,
            User.username,
            func.coalesce(entry_counts.c.dreams_n, 0).label("dreams_count"),
            func.coalesce(q_counts.c.dream_q_n, 0).label("questions_count"),
            func.coalesce(entry_counts.c.situations_n, 0).label("situations_count"),
            func.coalesce(q_counts.c.situation_q_n, 0).label("situation_questions_count"),
            func.coalesce(entry_counts.c.conflicts_n, 0).label("conflicts_count"),
            func.coalesce(q_counts.c.conflict_q_n, 0).label("conflict_questions_count"),
            func.coalesce(cost_sub.c.cost_usd, 0).label("cost_usd"),
            User.last_seen_at,
            User.created_at,
            User.is_active,
            User.access_tier,
            User.access_tier_until,
            User.paywall_shown_at,
            User.interview_passed_at,
            User.spend_reset_at,
        )
        .outerjoin(cost_sub, cost_sub.c.user_id == User.id)
        .outerjoin(entry_counts, entry_counts.c.user_id == User.id)
        .outerjoin(q_counts, q_counts.c.user_id == User.id)
    )


async def subscriber_rows(
    session: AsyncSession,
    *,
    search: str | None,
    sort: str,
    page: int,
    per_page: int,
    descending: bool = True,
) -> Page:
    query = _base_query()
    if search:
        pattern = f"%{search.lstrip('@')}%"
        query = query.where(or_(User.display_name.ilike(pattern), User.username.ilike(pattern)))

    count_q = select(func.count()).select_from(query.subquery())
    total = (await session.execute(count_q)).scalar_one()

    sort_label = _SORT_COLUMNS.get(sort, "created_at")
    order_col = query.selected_columns[sort_label]
    query = query.order_by(order_col.desc() if descending else order_col.asc())
    query = query.limit(per_page).offset((page - 1) * per_page)

    result = await session.execute(query)
    items = [
        SubscriberRow(
            user_id=r.id,
            telegram_user_id=r.telegram_user_id,
            display_name=r.display_name,
            username=r.username,
            dreams_count=r.dreams_count,
            questions_count=r.questions_count,
            situations_count=r.situations_count,
            situation_questions_count=r.situation_questions_count,
            conflicts_count=r.conflicts_count,
            conflict_questions_count=r.conflict_questions_count,
            cost_usd=Decimal(str(r.cost_usd)),
            last_seen_at=r.last_seen_at,
            created_at=r.created_at,
            is_active=r.is_active,
            access_tier=r.access_tier,
            access_tier_until=r.access_tier_until,
            paywall_shown_at=r.paywall_shown_at,
            interview_passed_at=r.interview_passed_at,
        )
        for r in result
    ]
    return Page(items=items, total=total, page=page, per_page=per_page)


@dataclass(frozen=True)
class CardSummary:
    user_id: int
    telegram_user_id: int
    display_name: str | None
    username: str | None
    dreams_count: int
    questions_count: int
    cost_usd: Decimal
    last_seen_at: object | None
    created_at: object
    is_active: bool
    situations_count: int = 0
    situation_questions_count: int = 0
    conflicts_count: int = 0
    conflict_questions_count: int = 0
    access_tier: str | None = None
    access_tier_until: object | None = None
    # Батч 2026-08-18: колонки «Пейвол» и «Интервью» + значки у аватара.
    paywall_shown_at: object | None = None
    interview_passed_at: object | None = None
    # Точка отсчёта лимита и траты после неё (полная сумма — в cost_usd).
    spend_reset_at: object | None = None
    cost_since_reset: Decimal = Decimal("0")


async def _cost_since(session: AsyncSession, *, user_id: int, since) -> Decimal:
    """Траты юзера после точки отсчёта. Без сброса — 0 (в карточке блок не показывается)."""
    if since is None:
        return Decimal("0")
    total = await session.scalar(
        select(func.coalesce(func.sum(LlmUsageEvent.cost_usd), 0)).where(
            LlmUsageEvent.user_id == user_id, LlmUsageEvent.created_at >= since
        )
    )
    return Decimal(str(total))


async def user_card_summary(session: AsyncSession, *, telegram_user_id: int) -> CardSummary | None:
    query = _base_query().where(User.telegram_user_id == telegram_user_id)
    row = (await session.execute(query)).first()
    if row is None:
        return None
    return CardSummary(
        user_id=row.id,
        telegram_user_id=row.telegram_user_id,
        display_name=row.display_name,
        username=row.username,
        dreams_count=row.dreams_count,
        questions_count=row.questions_count,
        situations_count=row.situations_count,
        situation_questions_count=row.situation_questions_count,
        conflicts_count=row.conflicts_count,
        conflict_questions_count=row.conflict_questions_count,
        cost_usd=Decimal(str(row.cost_usd)),
        last_seen_at=row.last_seen_at,
        created_at=row.created_at,
        is_active=row.is_active,
        access_tier=row.access_tier,
        access_tier_until=row.access_tier_until,
        paywall_shown_at=row.paywall_shown_at,
        interview_passed_at=row.interview_passed_at,
        spend_reset_at=row.spend_reset_at,
        cost_since_reset=await _cost_since(session, user_id=row.id, since=row.spend_reset_at),
    )
