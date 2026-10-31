"""APScheduler админки: периодический алёрт по дневным тратам LLM/STT."""
from __future__ import annotations

from apscheduler.schedulers.asyncio import AsyncIOScheduler

from admin.spend_alert import run_spend_alert_job


def start_scheduler(app):
    scheduler = AsyncIOScheduler(timezone="UTC")
    scheduler.start()
    # Дневной алёрт по тратам LLM/STT — ТОЛЬКО уведомление, не блокирует юзеров
    scheduler.add_job(
        run_spend_alert_job,
        "interval",
        minutes=15,
        args=[app],
        id="spend-alert",
        replace_existing=True,
        coalesce=True,
    )
    return scheduler


def shutdown_scheduler(scheduler):
    if scheduler is not None:
        scheduler.shutdown(wait=False)
