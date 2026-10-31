"""Единственная точка решения «пускать / блокировать» платное действие.

Порог зависит от тира доступа юзера (`src/domain/access.py`): первые тестеры, активная
интервью-группа без лимита, все остальные — на более жёстком пороге для новых.

Будущий пейвол заменит здесь cost-check на subscription-check, не трогая хендлеры.
Две модели поведения:
- ``deny_entry`` — entry-guard: блок до начала платного действия;
- ``exit_with_paywall`` — in-flow-guard: сохранить+выйти, когда лимит перейдён
  посреди потока (сон уже сохранён, FSM сбрасывается, опц. mark onboarded).
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message
from sqlalchemy.ext.asyncio import AsyncSession

from adapters.telegram.callback_utils import safe_answer
from adapters.telegram.texts import LIMIT_REACHED
from data.analytics_events import AnalyticsEventType, emit_append_event
from data.models import User
from data.repositories import SessionRepository, UsageEventRepository, UserRepository
from domain.access import resolve_limit_usd


def user_limit_usd(user: User, settings) -> Decimal | None:
    """Порог трат для юзера с учётом тира. None — лимита нет (основатель / активное интервью)."""
    return resolve_limit_usd(
        username=user.username,
        access_tier=user.access_tier,
        access_tier_until=user.access_tier_until,
        exempt=settings.spend_limit_exempt_set,
        early_tester_limit_usd=settings.user_spend_limit_usd,
        new_user_limit_usd=settings.new_user_spend_limit_usd,
    )


async def is_over_limit(session: AsyncSession, user: User, settings) -> bool:
    """True, если у юзера есть порог И траты ПОСЛЕ точки отсчёта его достигли.

    Точка отсчёта (`spend_reset_at`) сдвигается вручную скриптом: юзеру дают лимит
    заново, но история трат остаётся полной — админка показывает её без `since`.
    """
    limit = user_limit_usd(user, settings)
    if limit is None:
        return False
    total = await UsageEventRepository(session).total_cost_for_user(
        user.id, since=user.spend_reset_at
    )
    return total >= limit


async def _send(target: Message | CallbackQuery, text: str) -> None:
    if isinstance(target, CallbackQuery):
        await safe_answer(target)
        if target.message is not None:
            await target.message.answer(text)
    else:
        await target.answer(text)


def limit_reached_text(settings) -> str:
    """Текст пейвола с контактом владельца бота из настроек."""
    return LIMIT_REACHED.format(contact=settings.support_contact_url)


async def show_paywall(
    target: Message | CallbackQuery, session: AsyncSession, user: User, settings
) -> None:
    """Показать пейвол и учесть показ. ЕДИНСТВЕННЫЙ способ отправить LIMIT_REACHED.

    Все места показа (карточка сна, список без разбора, финал пре-вопросов,
    голосовое над лимитом) обязаны идти через неё, иначе колонка «Пейвол» в админке
    врёт. Структурный тест `test_paywall_single_send_point` держит запрет на прямые
    отправки.
    """
    await _mark_paywall_shown(session, user, settings)
    await _send(target, limit_reached_text(settings))


async def _mark_paywall_shown(session: AsyncSession, user: User, settings) -> None:
    """Дата первого показа пейвола + append-событие на каждый показ.

    Единственная точка учёта: её зовёт `show_paywall`, через который проходят все
    показы. Дата нужна админке (колонка «Пейвол» и значок), события — отчёту.
    """
    limit = user_limit_usd(user, settings)
    total = await UsageEventRepository(session).total_cost_for_user(
        user.id, since=user.spend_reset_at
    )
    if user.paywall_shown_at is None:
        user.paywall_shown_at = datetime.now(UTC)
        await session.flush()
    await emit_append_event(
        session,
        user_id=user.id,
        telegram_user_id=user.telegram_user_id,
        event_type=AnalyticsEventType.PAYWALL_SHOWN,
        metadata={"limit_usd": str(limit), "spent_usd": str(total)},
    )


async def deny_entry(
    target: Message | CallbackQuery, session: AsyncSession, user: User, settings
) -> bool:
    """Entry-guard: при превышении лимита шлёт пейвол и возвращает True (caller делает return)."""
    if await is_over_limit(session, user, settings):
        await show_paywall(target, session, user, settings)
        return True
    return False


async def exit_with_paywall(
    target: Message | CallbackQuery,
    state: FSMContext,
    session: AsyncSession,
    user: User,
    settings,
    *,
    session_id: int | None = None,
    mark_onboarded: bool = False,
) -> bool:
    """In-flow-guard: при превышении лимита закрывает активную сессию (сон остаётся
    сохранён), чистит FSM, опционально помечает онбординг завершённым, шлёт пейвол.
    Возвращает True, если действие заблокировано."""
    if not await is_over_limit(session, user, settings):
        return False
    if session_id is not None:
        await SessionRepository(session).complete(session_id=session_id)
    if mark_onboarded:
        await UserRepository(session).mark_onboarded(user_id=user.id)
    await state.clear()
    await show_paywall(target, session, user, settings)
    return True
