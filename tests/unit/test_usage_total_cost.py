from decimal import Decimal

import pytest

from data.models import LlmService, UsageMetric, User
from data.repositories import UsageEventRepository


@pytest.mark.asyncio
async def test_total_cost_sums_user_events(test_session):
    u = User(telegram_user_id=1, display_name="A")
    test_session.add(u)
    await test_session.flush()
    repo = UsageEventRepository(test_session)
    await repo.record(
        user_id=u.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("0.40"),
    )
    await repo.record(
        user_id=u.id, service=list(LlmService)[0], model="m",
        usage_metric=list(UsageMetric)[0], usage_amount=1, cost_usd=Decimal("0.65"),
    )
    await test_session.flush()
    assert await repo.total_cost_for_user(u.id) == Decimal("1.05")


@pytest.mark.asyncio
async def test_total_cost_zero_for_unknown(test_session):
    repo = UsageEventRepository(test_session)
    assert await repo.total_cost_for_user(999) == Decimal("0")
