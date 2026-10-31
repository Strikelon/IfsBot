"""Regression (item 12 audit): whisper STT cost MUST be recorded per transcription.

Operator reported a transcribed voice message didn't visibly bump «Нейросеть».
Audit found every voice path already records whisper cost (idle deliberately does
NOT transcribe), so the symptom is a perception issue — whisper is ~$0.0001/sec,
a fraction of a cent, masked by the much larger analysis cost.

These tests pin the invariant that `record_whisper` writes a nonzero-cost
`LlmUsageEvent` attributed to the user AND that it surfaces in the admin «$»
column (`admin.queries.subscriber_rows` cost / a SUM query), so a future
regression that drops the recording is caught.
"""

from __future__ import annotations

from decimal import Decimal

import pytest
from sqlalchemy import func, select

from admin.queries import subscriber_rows
from data.models import LlmService, LlmUsageEvent, UsageMetric
from data.repositories import UsageEventRepository, UserRepository
from llm.usage import UsageRecorder


@pytest.mark.asyncio
async def test_record_whisper_writes_nonzero_cost(test_session, schema):
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(
        telegram_user_id=42001, display_name="Voice User"
    )
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_whisper(user_id=user.id, audio_seconds=12.0)
    await test_session.flush()

    row = (
        await test_session.execute(
            select(LlmUsageEvent).where(LlmUsageEvent.user_id == user.id)
        )
    ).scalar_one()
    assert row.service == LlmService.WHISPER
    assert row.usage_metric == UsageMetric.AUDIO_SECONDS
    assert row.cost_usd > Decimal("0")

    total = (
        await test_session.execute(
            select(func.coalesce(func.sum(LlmUsageEvent.cost_usd), 0)).where(
                LlmUsageEvent.user_id == user.id
            )
        )
    ).scalar_one()
    assert Decimal(str(total)) > Decimal("0")


@pytest.mark.asyncio
async def test_whisper_cost_surfaces_in_admin_subscriber_rows(test_session, schema):
    """The admin «$» column SUMs cost_usd — a recorded whisper event must show there."""
    user_repo = UserRepository(test_session)
    user = await user_repo.get_or_create_by_tg_id(
        telegram_user_id=42002, display_name="Admin Voice"
    )
    await test_session.flush()

    recorder = UsageRecorder(UsageEventRepository(test_session))
    await recorder.record_whisper(user_id=user.id, audio_seconds=30.0)
    await test_session.flush()

    page = await subscriber_rows(
        test_session, search=None, sort="cost", page=1, per_page=50
    )
    rows = {r.user_id: r for r in page.items}
    assert user.id in rows
    assert rows[user.id].cost_usd > Decimal("0")
