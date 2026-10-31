"""Запись аналитических событий (write side).

`emit_funnel_once` — идемпотентная вставка одного шага воронки на юзера
(повторный /start или повторный вход в шаг НЕ плодит дубль). НЕ коммитит —
полагается на транзакцию внешнего DbSessionMiddleware.
"""
from __future__ import annotations

from datetime import UTC, datetime
from datetime import date as _date_cls
from typing import Final

import structlog
from sqlalchemy import func, select, union
from sqlalchemy.ext.asyncio import AsyncSession

from data.models import (
    AnalyticsEvent,
    Dream,
    LlmUsageEvent,
    SessionMessage,
    SessionMessageDirection,
    User,
)
from data.models import (
    Session as AnalysisSession,
)

log = structlog.get_logger(__name__)


class AnalyticsEventType:
    FIRST_LAUNCH: Final = "first_launch"
    # Экран выбора типа обращения (once-per-user, emit_funnel_once — идемпотентен).
    # В FUNNEL_STEPS отчёта НЕ входит (имена шагов не меняем, спека B8).
    ONB_TYPE_CHOSEN: Final = "onb_type_chosen"
    ONB_LIFE_CONTEXT: Final = "onb_life_context"
    ONB_DREAM_TOLD: Final = "onb_dream_told"
    ONB_INTENT_GIVEN: Final = "onb_intent_given"
    ONB_ANALYSIS_DELIVERED: Final = "onb_analysis_delivered"
    ONB_QA_ANSWERED: Final = "onb_qa_answered"
    ONB_COMPLETED: Final = "onb_completed"
    DAILY_ACTIVE: Final = "daily_active"
    # Вопросы по образам сна (per-session/append; см. emit_append_event)
    IMG_Q_OFFERED: Final = "img_q_offered"
    IMG_Q_CHOICE: Final = "img_q_choice"
    IMG_Q_NO_IMAGES: Final = "img_q_no_images"
    IMG_Q_ANSWERED: Final = "img_q_answered"
    IMG_Q_SKIPPED: Final = "img_q_skipped"
    IMG_Q_FLOW_DONE: Final = "img_q_flow_done"
    # Funnel-шаги онбординга (once-per-user, emit_funnel_once)
    ONB_IMG_Q_OFFERED: Final = "onb_img_q_offered"
    ONB_IMG_Q_DONE: Final = "onb_img_q_done"

    # Батч 2026-08-18: показ пейвола (append-событие, не funnel-шаг)
    PAYWALL_SHOWN: Final = "paywall_shown"


class AnalyticsEventRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def emit_funnel_once(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        event_type: str,
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Вставляет событие шага воронки, если его ещё нет у юзера.

        Возвращает True если вставлено, False если уже было.
        """
        existing = await self._session.scalar(
            select(func.count())
            .select_from(AnalyticsEvent)
            .where(
                AnalyticsEvent.user_id == user_id,
                AnalyticsEvent.event_type == event_type,
            )
        )
        if existing:
            return False
        when = occurred_at or datetime.now(UTC)
        self._session.add(
            AnalyticsEvent(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                event_type=event_type,
                occurred_at=when,
                event_day=when.date(),
                event_metadata=metadata,
            )
        )
        await self._session.flush()
        return True

    async def emit(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        event_type: str,
        occurred_at: datetime | None = None,
        metadata: dict | None = None,
    ) -> None:
        """Append-вставка события БЕЗ once-проверки (per-session/per-question
        события img_q_*). Дедупликация per-session-once — на чтении."""
        when = occurred_at or datetime.now(UTC)
        self._session.add(
            AnalyticsEvent(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                event_type=event_type,
                occurred_at=when,
                event_day=when.date(),
                event_metadata=metadata,
            )
        )
        await self._session.flush()

    async def emit_daily_active(
        self,
        *,
        user_id: int,
        telegram_user_id: int,
        occurred_at: datetime | None = None,
    ) -> bool:
        """Одна запись daily_active на юзера в день (idempotent по event_day)."""
        when = occurred_at or datetime.now(UTC)
        day = when.date()
        existing = await self._session.scalar(
            select(func.count())
            .select_from(AnalyticsEvent)
            .where(
                AnalyticsEvent.user_id == user_id,
                AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE,
                AnalyticsEvent.event_day == day,
            )
        )
        if existing:
            return False
        self._session.add(
            AnalyticsEvent(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                event_type=AnalyticsEventType.DAILY_ACTIVE,
                occurred_at=when,
                event_day=day,
                event_metadata=None,
            )
        )
        await self._session.flush()
        return True


async def emit_funnel_step(
    session: AsyncSession,
    *,
    user_id: int,
    telegram_user_id: int,
    event_type: str,
    metadata: dict | None = None,
) -> None:
    """Безопасный one-liner для хэндлеров: пишет шаг воронки, не роняя онбординг.

    Оборачивает запись в SAVEPOINT — сбой аналитики откатывает только её,
    не доменную транзакцию хэндлера.
    """
    try:
        async with session.begin_nested():
            await AnalyticsEventRepository(session).emit_funnel_once(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                event_type=event_type,
                metadata=metadata,
            )
    except Exception:  # noqa: BLE001
        log.warning("funnel_emit_failed", event_type=event_type, exc_info=True)


async def emit_append_event(
    session: AsyncSession,
    *,
    user_id: int,
    telegram_user_id: int,
    event_type: str,
    metadata: dict | None = None,
) -> None:
    """Безопасный one-liner для append-событий: SAVEPOINT, сбой аналитики
    откатывает только её, не доменную транзакцию хэндлера (копия emit_funnel_step
    без идемпотентности)."""
    try:
        async with session.begin_nested():
            await AnalyticsEventRepository(session).emit(
                user_id=user_id,
                telegram_user_id=telegram_user_id,
                event_type=event_type,
                metadata=metadata,
            )
    except Exception:  # noqa: BLE001
        log.warning("append_event_failed", event_type=event_type, exc_info=True)


def _as_date(value) -> _date_cls:
    """Normalize func.date(...) result: sqlite returns str, PG returns date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, _date_cls):
        return value
    return _date_cls.fromisoformat(str(value)[:10])


async def backfill_daily_active(session: AsyncSession) -> int:
    """Восстанавливает daily_active из существующих таймстемпов (idempotent).

    Источник дат: dreams.created_at, sessions.started_at,
    llm_usage_events.created_at (user_id not null),
    session_messages.created_at (direction=user_to_bot, joined to sessions).
    Возвращает число вставленных строк.
    """
    dreams_q = select(
        Dream.user_id.label("uid"),
        func.date(Dream.created_at).label("day"),
    )
    sessions_q = select(
        AnalysisSession.user_id.label("uid"),
        func.date(AnalysisSession.started_at).label("day"),
    )
    usage_q = select(
        LlmUsageEvent.user_id.label("uid"),
        func.date(LlmUsageEvent.created_at).label("day"),
    ).where(LlmUsageEvent.user_id.is_not(None))
    msgs_q = (
        select(
            AnalysisSession.user_id.label("uid"),
            func.date(SessionMessage.created_at).label("day"),
        )
        .join(AnalysisSession, SessionMessage.session_id == AnalysisSession.id)
        .where(
            SessionMessage.direction == SessionMessageDirection.USER_TO_BOT
        )
    )
    combined = union(dreams_q, sessions_q, usage_q, msgs_q).subquery()
    distinct_rows = (
        await session.execute(
            select(combined.c.uid, combined.c.day).distinct()
        )
    ).all()

    # Already-existing (user_id, day) for daily_active — skip duplicates.
    existing_rows = (
        await session.execute(
            select(AnalyticsEvent.user_id, AnalyticsEvent.event_day).where(
                AnalyticsEvent.event_type == AnalyticsEventType.DAILY_ACTIVE
            )
        )
    ).all()
    existing = {(uid, day) for uid, day in existing_rows}

    # Build tg_id map for the denormalized telegram_user_id column.
    tg_map = dict(
        (await session.execute(select(User.id, User.telegram_user_id))).all()
    )

    inserted = 0
    for uid, day in distinct_rows:
        if uid is None:
            continue
        day_val = _as_date(day)
        if (uid, day_val) in existing:
            continue
        session.add(
            AnalyticsEvent(
                user_id=uid,
                telegram_user_id=tg_map.get(uid, 0),
                event_type=AnalyticsEventType.DAILY_ACTIVE,
                occurred_at=datetime.combine(
                    day_val, datetime.min.time(), tzinfo=UTC
                ),
                event_day=day_val,
                event_metadata={"backfilled": True},
            )
        )
        existing.add((uid, day_val))
        inserted += 1
    await session.flush()
    return inserted
