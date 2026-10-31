"""Unit tests for admin.spend_alert — daily LLM cost alert.

NOTIFICATION ONLY — never blocks users (memory feedback-no-rate-limit).
Anti-spam: at most one DM per UTC day (in-memory flag).
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from decimal import Decimal

import admin.spend_alert as mod
from admin.spend_alert import check_spend_alert, run_spend_alert_job

# ─── Fake helpers ─────────────────────────────────────────────────────────────


class _FakeRepo:
    def __init__(self, total):
        self._total = total

    async def sum_cost_since(self, _since):
        return Decimal(str(self._total))


class _Settings:
    daily_spend_alert_usd: float = 50.0
    admin_id: int | None = 99999


class _SettingsNoAdmin:
    daily_spend_alert_usd: float = 50.0
    admin_id: int | None = None


@asynccontextmanager
async def _fake_factory():
    """Minimal session factory; real DB calls are monkeypatched in job tests."""
    yield object()


def _make_app(settings, bot=None):
    class _State:
        pass

    state = _State()
    state.settings = settings
    state.bot = bot if bot is not None else object()
    state.session_factory = _fake_factory

    class _App:
        pass

    app = _App()
    app.state = state
    return app


def _make_spend_patcher(total: str):
    """Return a patched today_spend_usd that always returns Decimal(total)."""

    async def _fake(repo):
        return Decimal(total)

    return _fake


class _FakeBot:
    """Ловит send_message вместо настоящего Telegram."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def send_message(self, *, chat_id: int, text: str) -> None:
        self.calls.append({"user_id": chat_id, "text": text})


# ─── Basic threshold checks ───────────────────────────────────────────────────


async def test_alert_fires_above_threshold() -> None:
    assert await check_spend_alert(_FakeRepo(60), Decimal("50")) is True


async def test_no_alert_below_threshold() -> None:
    assert await check_spend_alert(_FakeRepo(10), Decimal("50")) is False


async def test_alert_at_exact_threshold_fires() -> None:
    assert await check_spend_alert(_FakeRepo(50), Decimal("50")) is True


# ─── run_spend_alert_job: DM sending ─────────────────────────────────────────


async def test_job_sends_dm_when_crossed(monkeypatch) -> None:
    """DM is sent when spend > threshold and admin_id is set."""
    mod._last_alert_date = None  # reset anti-spam state

    monkeypatch.setattr(mod, "today_spend_usd", _make_spend_patcher("60"))
    bot = _FakeBot()
    calls = bot.calls

    await run_spend_alert_job(_make_app(_Settings(), bot))

    assert len(calls) == 1
    assert calls[0]["user_id"] == 99999
    # message must include threshold so admin knows the limit
    assert "50" in (calls[0]["text"] or "")


async def test_job_antispam_same_day(monkeypatch) -> None:
    """DM fires at most once per UTC day even if the job runs multiple times."""
    mod._last_alert_date = None  # reset anti-spam state

    monkeypatch.setattr(mod, "today_spend_usd", _make_spend_patcher("60"))
    bot = _FakeBot()
    calls = bot.calls

    app = _make_app(_Settings(), bot)
    await run_spend_alert_job(app)  # first run → DM sent
    await run_spend_alert_job(app)  # same UTC day → should NOT fire again

    assert len(calls) == 1  # exactly one DM


async def test_job_no_crash_when_no_admin_id(monkeypatch) -> None:
    """When admin_id is None, job only logs; no DM, no crash."""
    mod._last_alert_date = None

    monkeypatch.setattr(mod, "today_spend_usd", _make_spend_patcher("60"))
    bot = _FakeBot()
    calls = bot.calls

    await run_spend_alert_job(_make_app(_SettingsNoAdmin(), bot))  # must not raise

    assert len(calls) == 0


async def test_job_no_dm_when_below_threshold(monkeypatch) -> None:
    """No DM when daily spend is below the configured threshold."""
    mod._last_alert_date = None

    monkeypatch.setattr(mod, "today_spend_usd", _make_spend_patcher("10"))
    bot = _FakeBot()
    calls = bot.calls

    await run_spend_alert_job(_make_app(_Settings(), bot))

    assert len(calls) == 0
