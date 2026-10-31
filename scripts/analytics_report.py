"""Read-only аналитический отчёт: воронка онбординга + 5 retention-матриц +
таблица вовлечённости. SSH→psql на прод (ничего не пишет в прод).

Доступ из .env: SERVER_IP, SERVER_USER, SERVER_SSH_PORT и SERVER_SSH_KEY или SERVER_PASSWORD.
Usage: python scripts/analytics_report.py --cohort week --periods 8 [--by-source]
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import sys
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import paramiko  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from analytics.funnel import FUNNEL_STEPS, UserFunnel, funnel_by_cohort  # noqa: E402
from analytics.image_questions import (  # noqa: E402
    ImgQEvent,
    answered_histogram,
    drop_by_question,
    offer_choice_split,
    split_by_entry_type,
)
from analytics.retention import UserActivity, retention_matrix  # noqa: E402

load_dotenv()
OUT = Path("analytics_out")

# --- Raw SQL (read-only). Each returns CSV with header. ---
# Registration + source + reached funnel steps (one row per user).
SQL_USERS = (
    "SELECT u.id, to_char(u.created_at,'YYYY-MM-DD\"T\"HH24:MI:SS') AS registered_at, "
    "coalesce(u.acquisition_source,'') AS source, "
    "coalesce(u.telegram_user_id,0) AS tg_id, "
    "coalesce(u.display_name,'') AS name "
    "FROM users u ORDER BY u.id;"
)
# distinct daily_active days per user.
SQL_ACTIVE = (
    "SELECT user_id, to_char(event_day,'YYYY-MM-DD') AS day "
    "FROM analytics_events WHERE event_type='daily_active' "
    "AND user_id IS NOT NULL;"
)
# analysis days: день запуска разбора (session started).
SQL_ANALYSIS = (
    "SELECT user_id, to_char(date(started_at),'YYYY-MM-DD') AS day "
    "FROM sessions WHERE user_id IS NOT NULL;"
)
# qa days: user-direction реплики в Q&A.
SQL_QA = (
    "SELECT s.user_id, to_char(date(sm.created_at),'YYYY-MM-DD') AS day "
    "FROM session_messages sm JOIN sessions s ON sm.session_id=s.id "
    "WHERE sm.direction='user_to_bot' AND s.user_id IS NOT NULL;"
)
# return days: активность на сне, у которого РАНЕЕ уже была завершённая сессия
# разбора, причём активность позже того разбора.
SQL_RETURN = (
    "SELECT DISTINCT later.user_id, to_char(date(later.started_at),'YYYY-MM-DD') AS day "
    "FROM sessions later "
    "JOIN sessions prior ON prior.dream_id = later.dream_id "
    "AND prior.finished_at IS NOT NULL "
    "AND prior.finished_at < later.started_at "
    "WHERE later.user_id IS NOT NULL;"
)
# spend per day.
SQL_SPEND = (
    "SELECT user_id, to_char(date(created_at),'YYYY-MM-DD') AS day, "
    "sum(cost_usd) AS usd FROM llm_usage_events "
    "WHERE user_id IS NOT NULL GROUP BY user_id, date(created_at);"
)
# reached funnel steps.
SQL_FUNNEL = (
    "SELECT DISTINCT user_id, event_type FROM analytics_events "
    "WHERE event_type LIKE 'onb_%' OR event_type='first_launch';"
)
# engagement table (JTBD).
SQL_ENGAGE = (
    "SELECT u.telegram_user_id AS tg_id, coalesce(u.username,'') AS username, "
    "coalesce(u.display_name,'') AS name, "
    "to_char(u.created_at,'YYYY-MM-DD') AS joined, "
    "(SELECT count(DISTINCT event_day) FROM analytics_events a "
    " WHERE a.user_id=u.id AND a.event_type='daily_active') AS active_days, "
    "(SELECT count(*) FROM dreams d WHERE d.user_id=u.id "
    " AND d.kind='dream') AS dreams, "
    "(SELECT count(*) FROM dreams d WHERE d.user_id=u.id "
    " AND d.kind='external_situation') AS situations, "
    "(SELECT count(*) FROM dreams d WHERE d.user_id=u.id "
    " AND d.kind='internal_conflict') AS conflicts, "
    "(SELECT count(*) FROM sessions s WHERE s.user_id=u.id "
    " AND s.finished_at IS NOT NULL) AS analyzed, "
    "(SELECT count(*) FROM session_messages sm JOIN sessions s ON sm.session_id=s.id "
    " WHERE s.user_id=u.id AND sm.direction='user_to_bot') AS qa_answers, "
    "coalesce((SELECT sum(cost_usd) FROM llm_usage_events l WHERE l.user_id=u.id),0) AS spend_usd, "
    "(u.onboarding_completed_at IS NOT NULL) AS onboarded, "
    "coalesce(u.acquisition_source,'') AS source "
    "FROM users u ORDER BY active_days DESC, dreams DESC;"
)
# img_q_* events (feature «вопросы по образам»), append-only; дедуп на чтении.
IMGQ_EVENTS_SQL = (
    "SELECT user_id, event_type, event_metadata::text AS meta "
    "FROM analytics_events "
    "WHERE event_type IN ('img_q_offered','img_q_choice','img_q_no_images',"
    "'img_q_answered','img_q_skipped','img_q_flow_done') "
    "ORDER BY occurred_at;"
)


# Показы пейвола (append-only события).
PAYWALL_EVENTS_SQL = (
    "SELECT count(*) AS shows, count(DISTINCT user_id) AS users "
    "FROM analytics_events WHERE event_type = 'paywall_shown';"
)
# Кто пришёл на интервью ПОСЛЕ того, как увидел пейвол — конверсия пейвол→интервью.
PAYWALL_INTERVIEW_SQL = (
    "SELECT count(*) FILTER (WHERE paywall_shown_at IS NOT NULL) AS saw_paywall, "
    "count(*) FILTER (WHERE interview_passed_at IS NOT NULL) AS interviewed, "
    "count(*) FILTER (WHERE paywall_shown_at IS NOT NULL "
    "  AND interview_passed_at > paywall_shown_at) AS interviewed_after_paywall "
    "FROM users;"
)


def _psql(sql: str) -> str:
    return f'sudo -u postgres psql -d dream_bot -P pager=off --csv -c "{sql}"'


def _fetch(ssh: paramiko.SSHClient, sql: str) -> list[dict[str, str]]:
    _, out, err = ssh.exec_command(_psql(sql))
    text = out.read().decode("utf-8", "replace")
    errt = err.read().decode("utf-8", "replace").strip()
    if errt:
        print(f"[stderr] {errt}", file=sys.stderr)
    return list(csv.DictReader(io.StringIO(text)))


def _d(s: str) -> date:
    return date.fromisoformat(s[:10])


def _build_activities(data: dict[str, list[dict]]) -> list[UserActivity]:
    reg: dict[int, datetime] = {}
    src: dict[int, str | None] = {}
    for r in data["users"]:
        uid = int(r["id"])
        reg[uid] = datetime.fromisoformat(r["registered_at"])
        src[uid] = r["source"] or None

    def days(key: str) -> dict[int, list[date]]:
        m: dict[int, list[date]] = defaultdict(list)
        for r in data[key]:
            m[int(r["user_id"])].append(_d(r["day"]))
        return m

    active, analysis, qa, ret = days("active"), days("analysis"), days("qa"), days("return")
    spend: dict[int, dict[date, float]] = defaultdict(dict)
    for r in data["spend"]:
        spend[int(r["user_id"])][_d(r["day"])] = float(r["usd"] or 0)

    return [
        UserActivity(
            user_id=uid, registered_at=reg[uid], source=src[uid],
            active_days=active.get(uid, []), analysis_days=analysis.get(uid, []),
            qa_days=qa.get(uid, []), return_days=ret.get(uid, []),
            spend_by_day=spend.get(uid, {}),
        )
        for uid in reg
    ]


def _build_funnels(data: dict[str, list[dict]]) -> list[UserFunnel]:
    reached: dict[int, set[str]] = defaultdict(set)
    for r in data["funnel"]:
        reached[int(r["user_id"])].add(r["event_type"])
    reg, src = {}, {}
    for r in data["users"]:
        uid = int(r["id"])
        reg[uid] = datetime.fromisoformat(r["registered_at"])
        src[uid] = r["source"] or None
    return [
        UserFunnel(user_id=uid, registered_at=reg[uid], source=src[uid],
                   reached=reached.get(uid, set()))
        for uid in reg
    ]


def _build_imgq_events(rows: list[dict]) -> list[ImgQEvent]:
    events: list[ImgQEvent] = []
    for r in rows:
        try:
            meta = json.loads(r["meta"]) if r.get("meta") else {}
        except (ValueError, TypeError):
            meta = {}
        events.append(ImgQEvent(
            user_id=int(r["user_id"]) if r.get("user_id") else 0,
            event_type=r["event_type"],
            metadata=meta if isinstance(meta, dict) else {},
        ))
    return events


_ENTRY_TYPE_TITLES = {
    "dream": "Сны (вопросы по образам)",
    "external_situation": "Волнующие ситуации (уточняющие вопросы)",
    "internal_conflict": "Внутренние конфликты (уточняющие вопросы)",
}


def _print_imgq_report(events: list[ImgQEvent], img_q_since: str) -> None:
    """Отчёт по пре-вопросам, РАЗДЕЛЬНО по типу обращения.

    Имена событий одни на все типы (append-only, отчёт живёт с июля), поэтому
    разрез идёт по metadata.entry_type; события без этого поля — вся история
    до появления типов — считаются снами.
    """
    print("\n=== Вопросы перед разбором (img_q_*) ===")
    if not events:
        print("(событий нет)")
        return

    buckets = split_by_entry_type(events)
    for kind, title in _ENTRY_TYPE_TITLES.items():
        bucket = buckets.get(kind)
        if not bucket:
            continue
        sessions = {e.metadata.get("session_id") for e in bucket}
        print(f"\n-- {title}: сессий с событиями {len(sessions)} --")
        _print_imgq_block(bucket)
    print(f"\n(шаги img_q_* активны с {img_q_since})")


def _print_paywall_report(event_rows: list[dict], paywall_rows: list[dict]) -> None:
    """Блок «Пейвол»: показы + воронка пейвол→интервью."""
    ev = event_rows[0] if event_rows else {}
    print("\n=== Пейвол ===")
    print(f"Показов пейвола: {ev.get('shows', 0)} (юзеров: {ev.get('users', 0)})")
    if paywall_rows:
        row = paywall_rows[0]
        print(
            f"По колонкам users: видели пейвол {row.get('saw_paywall', '?')}, "
            f"прошли интервью {row.get('interviewed', '?')}, "
            f"из них после пейвола {row.get('interviewed_after_paywall', '?')}"
        )


def _print_imgq_block(events: list[ImgQEvent]) -> None:
    # 1. Split выбора «ответить vs быстро».
    split = offer_choice_split(events)
    print(f"Предложено (сессий): {split['offered']}")
    print(f"  ответить: {split['answer']}  быстро (decline): {split['decline']}  "
          f"без выбора: {split['no_choice']}")
    for is_onb, label in ((True, "онбординг"), (False, "обычный поток")):
        b = split["by_onboarding"][is_onb]
        print(f"  [{label}] offered={b['offered']} answer={b['answer']} "
              f"decline={b['decline']}")
    if split["by_via"]:
        via = "  ".join(f"{k}={v}" for k, v in sorted(split["by_via"].items()))
        print(f"  выбор via: {via}")

    # 2. Гистограмма answered/total по завершённым сессиям.
    hist = answered_histogram(events)
    print("Гистограмма ответов (answered/total → сессий):")
    if hist:
        for (answered, total), n in sorted(hist.items()):
            print(f"  {answered}/{total}: {n}")
    else:
        print("  (нет завершённых сессий)")

    # 3. Застревание: «дошёл до вопроса N и исчез».
    drops = drop_by_question(events)
    print("Бросили на вопросе N (без flow_done):")
    if drops:
        for q_num, n in sorted(drops.items()):
            print(f"  вопрос {q_num}: {n}")
    else:
        print("  (нет)")

    # 4. img_q_no_images по reason (дедуп по session_id, где он есть).
    seen_sids: set = set()
    reasons: dict[str, int] = {}
    for e in events:
        if e.event_type != "img_q_no_images":
            continue
        sid = e.metadata.get("session_id")
        if sid is not None:
            if sid in seen_sids:
                continue
            seen_sids.add(sid)
        reason = str(e.metadata.get("reason", "?"))
        reasons[reason] = reasons.get(reason, 0) + 1
    if reasons:
        line = "  ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        print(f"Вопросы не сформулированы (no_images) по reason: {line}")


def _write_matrix_csv(path: Path, res, *, spend: bool) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["cohort", "source", "size"] + [f"p{i}" for i in range(res.periods)]
        w.writerow(header)
        for row in res.rows:
            w.writerow([row.cohort, row.source, row.cohort_size, *row.values])
        if spend:
            w.writerow([])
            w.writerow(["$ cohort", "source", "size"] + [f"p{i}" for i in range(res.periods)])
            for row in res.rows:
                w.writerow([row.cohort, row.source, row.cohort_size, *row.dollars])


def _heatmap(path: Path, res, title: str) -> None:
    if not res.rows:
        return
    grid = [r.values for r in res.rows]
    fig, ax = plt.subplots(figsize=(1.2 * res.periods + 2, 0.5 * len(grid) + 2))
    im = ax.imshow(grid, aspect="auto", cmap="YlGn", vmin=0, vmax=100)
    ax.set_xticks(range(res.periods), [f"p{i}" for i in range(res.periods)])
    ax.set_yticks(range(len(res.rows)), [f"{r.cohort} {r.source}" for r in res.rows])
    for i, r in enumerate(res.rows):
        for j, v in enumerate(r.values):
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="% retained")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _funnel_chart(path: Path, rows) -> None:
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(rows) + 2))
    x = range(len(FUNNEL_STEPS))
    for r in rows:
        ax.plot(x, [r.step_pct[s] for s in FUNNEL_STEPS], marker="o",
                label=f"{r.cohort} {r.source} (n={r.cohort_size})")
    ax.set_xticks(list(x), FUNNEL_STEPS, rotation=30, ha="right")
    ax.set_ylabel("% когорты")
    ax.set_title("Воронка онбординга")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["day", "week", "month"], default="week")
    ap.add_argument("--periods", type=int, default=8)
    ap.add_argument("--by-source", action="store_true")
    ap.add_argument(
        "--img-q-since", default="НЕ-ЗАДАНО (укажите --img-q-since ГГГГ-ММ-ДД)",
        help="дата включения фичи img_q (пометка в funnel-отчёте); "
             "default — очевидная заглушка, при запуске подставить фактическую дату деплоя",
    )
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(
        hostname=os.environ["SERVER_IP"],
        port=int(os.getenv("SERVER_SSH_PORT", "22")),
        username=os.environ["SERVER_USER"],
        timeout=30,
        **(
            {"key_filename": os.path.expanduser(os.environ["SERVER_SSH_KEY"])}
            if os.getenv("SERVER_SSH_KEY")
            else {"password": os.environ["SERVER_PASSWORD"]}
        ),
    )
    try:
        data = {
            "users": _fetch(ssh, SQL_USERS),
            "active": _fetch(ssh, SQL_ACTIVE),
            "analysis": _fetch(ssh, SQL_ANALYSIS),
            "qa": _fetch(ssh, SQL_QA),
            "return": _fetch(ssh, SQL_RETURN),
            "spend": _fetch(ssh, SQL_SPEND),
            "funnel": _fetch(ssh, SQL_FUNNEL),
            "engage": _fetch(ssh, SQL_ENGAGE),
            "imgq": _fetch(ssh, IMGQ_EVENTS_SQL),
            "paywall_events": _fetch(ssh, PAYWALL_EVENTS_SQL),
            "paywall": _fetch(ssh, PAYWALL_INTERVIEW_SQL),
        }
    finally:
        ssh.close()

    activities = _build_activities(data)
    funnels = _build_funnels(data)

    # Funnel.
    frows = funnel_by_cohort(funnels, granularity=args.cohort, by_source=args.by_source)
    with (OUT / "funnel.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cohort", "source", "size", *FUNNEL_STEPS])
        for r in frows:
            w.writerow([r.cohort, r.source, r.cohort_size,
                        *[r.step_pct[s] for s in FUNNEL_STEPS]])
    _funnel_chart(OUT / "funnel.png", frows)
    print(f"Примечание: шаги onb_img_q_* активны с {args.img_q_since}; "
          "для когорт ранее — нули, проценты от когорты не ломаются.")

    # 5 retention matrices.
    for metric in ("launch", "analysis", "qa", "return", "spend"):
        res = retention_matrix(
            activities, metric=metric, granularity=args.cohort,
            periods=args.periods, by_source=args.by_source,
        )
        _write_matrix_csv(OUT / f"retention_{metric}.csv", res, spend=(metric == "spend"))
        _heatmap(OUT / f"retention_{metric}.png", res, f"R_{metric} ({args.cohort})")

    # Image-questions metrics (печать в stdout).
    _print_imgq_report(_build_imgq_events(data["imgq"]), args.img_q_since)

    # Пейвол (печать в stdout).
    _print_paywall_report(data["paywall_events"], data["paywall"])

    # Engagement table (passthrough).
    with (OUT / "engagement.csv").open("w", newline="", encoding="utf-8") as f:
        if data["engage"]:
            w = csv.DictWriter(f, fieldnames=list(data["engage"][0].keys()))
            w.writeheader()
            w.writerows(data["engage"])

    print(f"OK → {OUT.resolve()} (cohort={args.cohort}, periods={args.periods})")


if __name__ == "__main__":
    main()
