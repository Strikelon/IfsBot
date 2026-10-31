from datetime import UTC, date, datetime

from analytics.retention import (
    UserActivity,
    cohort_key,
    period_index,
    retention_matrix,
)


def test_cohort_key_week_buckets_to_monday() -> None:
    # 2026-06-29 is a Monday.
    assert cohort_key(datetime(2026, 6, 29, 15, tzinfo=UTC), "week") == date(2026, 6, 29)
    assert cohort_key(datetime(2026, 7, 1, 9, tzinfo=UTC), "week") == date(2026, 6, 29)


def test_cohort_key_month_buckets_to_first() -> None:
    assert cohort_key(datetime(2026, 6, 29, tzinfo=UTC), "month") == date(2026, 6, 1)


def test_period_index_day_granularity() -> None:
    reg = datetime(2026, 6, 1, 12, tzinfo=UTC)
    assert period_index(reg, date(2026, 6, 1), "day") == 0
    assert period_index(reg, date(2026, 6, 4), "day") == 3


def test_launch_retention_single_cohort() -> None:
    reg = datetime(2026, 6, 1, 10, tzinfo=UTC)
    users = [
        UserActivity(
            user_id=1, registered_at=reg, source=None,
            active_days=[date(2026, 6, 1), date(2026, 6, 2)],
            analysis_days=[], qa_days=[], return_days=[], spend_by_day={},
        ),
        UserActivity(
            user_id=2, registered_at=reg, source=None,
            active_days=[date(2026, 6, 1)],  # day0 only
            analysis_days=[], qa_days=[], return_days=[], spend_by_day={},
        ),
    ]
    res = retention_matrix(users, metric="launch", granularity="day", periods=3)
    row = res.rows[0]
    assert row.cohort_size == 2
    assert row.values[0] == 100.0   # both active day0
    assert row.values[1] == 50.0    # only user1 active day1
    assert row.values[2] == 0.0


def test_return_distinct_from_analysis() -> None:
    """R_return: активность на уже-разобранном сне в более позднем периоде.
    R_analysis: факт запуска разбора. Один и тот же юзер может дать analysis в
    период 0 и return в период 2, и это РАЗНЫЕ метрики."""
    reg = datetime(2026, 6, 1, tzinfo=UTC)
    u = UserActivity(
        user_id=1, registered_at=reg, source=None,
        active_days=[date(2026, 6, 1), date(2026, 6, 3)],
        analysis_days=[date(2026, 6, 1)],          # разобрал в день 0
        qa_days=[],
        return_days=[date(2026, 6, 3)],            # вернулся к разобранному в день 2
        spend_by_day={},
    )
    a = retention_matrix([u], metric="analysis", granularity="day", periods=3)
    r = retention_matrix([u], metric="return", granularity="day", periods=3)
    assert a.rows[0].values[0] == 100.0   # analysis в период 0
    assert a.rows[0].values[2] == 0.0     # no analysis в период 2
    assert r.rows[0].values[0] == 0.0     # no return в период 0
    assert r.rows[0].values[2] == 100.0   # return в период 2


def test_spend_matrix_has_dollars() -> None:
    reg = datetime(2026, 6, 1, tzinfo=UTC)
    u = UserActivity(
        user_id=1, registered_at=reg, source=None,
        active_days=[], analysis_days=[], qa_days=[], return_days=[],
        spend_by_day={date(2026, 6, 1): 0.12, date(2026, 6, 2): 0.30},
    )
    res = retention_matrix([u], metric="spend", granularity="day", periods=3)
    assert res.rows[0].values[0] == 100.0          # spent in period 0
    assert res.rows[0].dollars[0] == 0.12
    assert res.rows[0].dollars[1] == 0.30
