from datetime import UTC, datetime

from analytics.funnel import FUNNEL_STEPS, UserFunnel, funnel_by_cohort


def test_funnel_counts_and_percentages() -> None:
    reg = datetime(2026, 6, 1, 10, tzinfo=UTC)
    users = [
        UserFunnel(user_id=1, registered_at=reg, source=None,
                   reached={"first_launch", "onb_life_context", "onb_dream_told"}),
        UserFunnel(user_id=2, registered_at=reg, source=None,
                   reached={"first_launch", "onb_life_context"}),
        UserFunnel(user_id=3, registered_at=reg, source=None,
                   reached={"first_launch"}),
    ]
    rows = funnel_by_cohort(users, granularity="month")
    assert len(rows) == 1
    row = rows[0]
    assert row.cohort_size == 3
    assert row.step_counts["first_launch"] == 3
    assert row.step_counts["onb_life_context"] == 2
    assert row.step_counts["onb_dream_told"] == 1
    assert row.step_pct["first_launch"] == 100.0
    assert row.step_pct["onb_life_context"] == 66.7
    assert row.step_pct["onb_completed"] == 0.0


def test_funnel_steps_order_stable() -> None:
    assert FUNNEL_STEPS[0] == "first_launch"
    assert FUNNEL_STEPS[-1] == "onb_completed"
    assert len(FUNNEL_STEPS) == 9
