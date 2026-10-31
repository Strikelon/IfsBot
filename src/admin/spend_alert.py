"""Админ-алёрт по дневным тратам LLM/STT.

ТОЛЬКО уведомление — НЕ блокирует юзеров и не отказывает им (memory
feedback-no-rate-limit). Ручной выключатель — на стороне оператора
(Auto Top-Up off / остановка сервиса), не автоматический cost-guard.

Анти-спам: DM шлётся не чаще раза в сутки UTC (in-memory флаг). При рестарте
сервиса может прийти повторный алёрт — это приемлемо.
"""
from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal

import structlog

from data.repositories import UsageEventRepository

log = structlog.get_logger(__name__)

# Anti-spam: запоминаем дату последнего DM-алёрта (UTC). Сбрасывается при рестарте.
_last_alert_date: date | None = None


async def today_spend_usd(repo: UsageEventRepository) -> Decimal:
    """Суммарные траты LLM/STT с начала текущих суток UTC."""
    since = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return await repo.sum_cost_since(since)


async def check_spend_alert(repo, threshold_usd: Decimal) -> bool:
    """Проверить, превышен ли дневной порог трат.

    Логирует `spend.alert.threshold_crossed` и возвращает True при `total >= threshold`.
    НЕ блокирует пользователей, НЕ отказывает юзерам.
    """
    total = await today_spend_usd(repo)
    if total >= threshold_usd:
        log.warning(
            "spend.alert.threshold_crossed",
            total_usd=str(total),
            threshold_usd=str(threshold_usd),
        )
        return True
    return False


async def run_spend_alert_job(app) -> None:
    """APScheduler-job: проверить дневные траты, при пробое порога — DM на admin.

    Регистрируется в `admin.scheduler.start_scheduler` интервальным заданием.
    Толерантен к ошибкам: логирует, не бросает исключения наружу.
    """
    global _last_alert_date

    settings = app.state.settings
    threshold = Decimal(str(settings.daily_spend_alert_usd))
    admin_tg_id: int | None = getattr(settings, "admin_id", None)

    factory = app.state.session_factory
    try:
        async with factory() as session:
            repo = UsageEventRepository(session)
            crossed = await check_spend_alert(repo, threshold)

            if not crossed:
                return

            if admin_tg_id is None:
                log.info("spend.alert.no_admin_id_configured")
                return

            today = datetime.now(UTC).date()
            if _last_alert_date == today:
                log.debug("spend.alert.already_sent_today", date=str(today))
                return

            # Получаем актуальную сумму для тела сообщения
            total = await today_spend_usd(repo)
            text = (
                f"⚠️ Дневные траты LLM/STT превысили порог!\n"
                f"Сегодня: ${total:.4f} (порог: ${threshold:.2f})\n"
                f"Это уведомление — бот продолжает работу в штатном режиме."
            )
            try:
                await app.state.bot.send_message(chat_id=admin_tg_id, text=text)
                ok, error = True, None
            except Exception as exc:  # noqa: BLE001 — недоставка алёрта не роняет джобу
                ok, error = False, str(exc)
            if ok:
                _last_alert_date = today
                log.info(
                    "spend.alert.dm_sent",
                    admin_tg_id=admin_tg_id,
                    total_usd=str(total),
                )
            else:
                log.warning("spend.alert.dm_failed", error=error)
    except Exception:  # noqa: BLE001
        log.exception("spend.alert.job_error")
