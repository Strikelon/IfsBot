#!/usr/bin/env python
"""Synthetic spike load-test for the IfsBot generator path.

MANUAL OPS TOOL — DO NOT import from the app or from the test suite.
When run without --dry-run it makes REAL API calls to OpenRouter (costs money).

Purpose
-------
Drive the core interpretation path (generator) directly — no Telegram, no DB —
to measure the throughput ceiling and find the right LLM_MAX_CONCURRENCY value
for production before a traffic spike.

Usage
-----
    # Validate arg parsing / task-generation logic without real calls:
    PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py --dry-run

    # Small real test against dev OpenRouter key (~$0.05):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py \\
        --n 50 --arrival-sec 60 --concurrency 10

    # Full spike simulation (~$1-2):
    PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py \\
        --n 1000 --arrival-sec 600 --concurrency 30

Tuning procedure (see also docs/ops/spike-runbook.md §2):
    1. Run with --concurrency 10.  Note 429-rate.
    2. Raise by +5 until 429s appear.
    3. Step back one notch — that is your prod LLM_MAX_CONCURRENCY.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import AsyncOpenAI

import argparse
import asyncio
import math
import random
import sys
import time
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# CLI — parsed before any heavy imports so --help works with no PYTHONPATH
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Synthetic spike load-test: drives the generator core path directly, "
            "bypassing Telegram and the DB, to find the LLM_MAX_CONCURRENCY ceiling."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Tuning procedure:
  1. Start with --concurrency 10 --n 50 (real run against dev key).
  2. Raise --concurrency by 5 each round until 429s appear.
  3. Step back one notch — that value is prod LLM_MAX_CONCURRENCY.
  4. Record the value in docs/ops/spike-runbook.md and in prod .env.

Examples:
  # dry-run (no real calls, ~0 cost):
  PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py --dry-run

  # small real test:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py \\
      --n 50 --arrival-sec 60 --concurrency 10

  # full spike:
  PYTHONPATH=src .venv/Scripts/python.exe scripts/loadtest_spike.py \\
      --n 1000 --arrival-sec 600 --concurrency 30
""",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=100,
        metavar="N",
        help="Total synthetic interpretation tasks (default: 100)",
    )
    parser.add_argument(
        "--arrival-sec",
        type=float,
        default=120.0,
        metavar="SECS",
        help="Spread arrivals over SECS seconds using Poisson inter-arrivals (default: 120)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=30,
        metavar="LIMIT",
        help="Shared concurrency gate size — mirrors LLM_MAX_CONCURRENCY (default: 30)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=150.0,
        metavar="SECS",
        help=(
            "Per-task wall-clock timeout in seconds — live runs only (default: 150). "
            "The generator already has a 120 s per-attempt timeout internally; "
            "this outer timeout fires if the full fallback chain stalls. "
            "Dry-run latency is always synthetic."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Simulate tasks without real API calls. "
            "The concurrency gate and Poisson scheduler are exercised; "
            "each task sleeps for a random 2-8 s to mimic latency."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for Poisson inter-arrival times and synthetic dreams (default: 42)",
    )
    return parser


# ---------------------------------------------------------------------------
# Synthetic dream corpus (no DB needed)
# ---------------------------------------------------------------------------

_DREAM_CORPUS = [
    "Мне снилось, что я лечу над городом и не могу приземлиться — земля всё время ускользает.",
    "Я оказался в старой квартире из детства. Там было темно, тихо, и пахло чем-то знакомым.",
    "Мне снилась огромная волна, которая накрывала меня, но я не тонул — стоял под ней.",
    "Я опаздывал на важный экзамен и не мог найти нужную аудиторию в огромном здании.",
    "В сне была длинная дорога через тёмный лес, и в конце её стоял кто-то, кого я когда-то знал.",
    "Мне снился потерянный ребёнок, которому я пытался помочь найти дорогу домой.",
    "Я был на многолюдной вечеринке, но никто не видел меня — как будто я был невидимкой.",
    "Во сне я плыл по широкой реке, вода была тёплой и спокойной, берега медленно уплывали.",
    "Мне снилась огромная библиотека с книгами на неизвестных языках; я понимал каждое слово.",
    "Я искал выход из лабиринта, но каждый коридор возвращал меня к тому же месту.",
    "В сне стояло большое дерево посреди пустого поля, и я знал, что за ним что-то важное.",
    "Мне снился поезд, который я едва не успел поймать, и чувство, что упустил что-то важное.",
    "Я стоял на краю высокого обрыва и чувствовал странное притяжение вниз, но не страх.",
    "В сне была незнакомая комната с высокими потолками, в которой я жил и считала своей.",
    "Мне снилась собака, которую я когда-то потерял — она смотрела на меня без осуждения.",
]

_PER_SESSION_CONTEXTS = [
    "Хочу понять, что меня сдерживает в работе",
    "Чувствую тревогу и неопределённость",
    "Ощущаю себя потерянным в последнее время",
    "Хочу разобраться в близких отношениях",
    "Чувствую усталость и внутреннюю пустоту",
    "Не могу принять важное решение",
    "Хочу лучше понять свои желания",
]


def _synthetic_dream(seed: int) -> dict[str, str]:
    rng = random.Random(seed)
    return {
        "dream": rng.choice(_DREAM_CORPUS),
        "per_session": rng.choice(_PER_SESSION_CONTEXTS),
        "name": f"TestUser{seed % 50}",
        "life_context": "Городской житель, работает в IT, ищет себя.",
    }


# ---------------------------------------------------------------------------
# Task result
# ---------------------------------------------------------------------------


@dataclass
class TaskResult:
    task_id: int
    success: bool
    latency_s: float
    error_kind: str | None = None  # "429" | "402" | "timeout" | "other" | None


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = max(0, math.ceil(p / 100.0 * len(sorted_values)) - 1)
    return sorted_values[min(idx, len(sorted_values) - 1)]


# ---------------------------------------------------------------------------
# In-flight tracker (asyncio is single-threaded — plain int is safe)
# ---------------------------------------------------------------------------

_in_flight: int = 0
_peak_in_flight: int = 0


def _enter_inflight() -> None:
    global _in_flight, _peak_in_flight
    _in_flight += 1
    if _in_flight > _peak_in_flight:
        _peak_in_flight = _in_flight


def _exit_inflight() -> None:
    global _in_flight
    _in_flight -= 1


# ---------------------------------------------------------------------------
# Per-task runners
# ---------------------------------------------------------------------------


async def _run_task_dry(
    task_id: int,
    semaphore: asyncio.Semaphore,
    timeout_s: float,
) -> TaskResult:
    """Dry-run: simulate generator latency behind the same gate, no real HTTP."""
    rng = random.Random(task_id ^ 0xDEADBEEF)
    simulated_latency = min(rng.uniform(2.0, 8.0), timeout_s * 0.9)

    async with semaphore:
        t0 = time.monotonic()
        _enter_inflight()
        try:
            await asyncio.sleep(simulated_latency)
            return TaskResult(
                task_id=task_id,
                success=True,
                latency_s=time.monotonic() - t0,
            )
        except asyncio.CancelledError:
            return TaskResult(
                task_id=task_id,
                success=False,
                latency_s=time.monotonic() - t0,
                error_kind="timeout",
            )
        finally:
            _exit_inflight()


async def _run_task_real(
    task_id: int,
    client: object,
    messages: list[dict],
    timeout_s: float,
) -> TaskResult:
    """Real run: call the generator core path (acquires GatedTransport semaphore internally)."""
    # Lazy import — keeps module-level --help fast even without PYTHONPATH
    from llm.generator import ErrorCategory, generate_quick_response  # type: ignore[import]

    t0 = time.monotonic()
    _enter_inflight()
    try:
        result = await asyncio.wait_for(
            generate_quick_response(client=client, messages=messages),
            timeout=timeout_s,
        )
        latency = time.monotonic() - t0

        if result.error_category is None:
            return TaskResult(task_id=task_id, success=True, latency_s=latency)

        # Map generator error category → load-test bucket
        if result.error_category == ErrorCategory.QUOTA_EXCEEDED:
            error_kind = "402"
        elif result.error_category == ErrorCategory.TRANSIENT:
            # Covers 429, 502, 503, connection errors — all mean "provider pushed back"
            error_kind = "429"
        else:
            error_kind = "other"
        return TaskResult(
            task_id=task_id,
            success=False,
            latency_s=latency,
            error_kind=error_kind,
        )
    except TimeoutError:
        return TaskResult(
            task_id=task_id,
            success=False,
            latency_s=time.monotonic() - t0,
            error_kind="timeout",
        )
    except Exception:  # noqa: BLE001
        return TaskResult(
            task_id=task_id,
            success=False,
            latency_s=time.monotonic() - t0,
            error_kind="other",
        )
    finally:
        _exit_inflight()


# ---------------------------------------------------------------------------
# Main async orchestrator
# ---------------------------------------------------------------------------


async def _async_main(args: argparse.Namespace) -> int:
    global _in_flight, _peak_in_flight
    _in_flight = 0
    _peak_in_flight = 0

    n: int = args.n
    arrival_sec: float = args.arrival_sec
    concurrency: int = args.concurrency
    timeout_s: float = args.timeout
    dry_run: bool = args.dry_run
    seed: int = args.seed

    print()
    print("=" * 64)
    print("  IfsBot Spike Load-Test")
    print("=" * 64)
    print(f"  Mode         : {'DRY RUN (no real API calls)' if dry_run else 'LIVE — calls OpenRouter!'}")
    print(f"  Tasks (N)    : {n}")
    print(f"  Arrival      : {n} tasks over {arrival_sec}s  (Poisson inter-arrivals)")
    print(f"  Concurrency  : {concurrency}  (gate/semaphore == LLM_MAX_CONCURRENCY)")
    print(f"  Per-task TO  : {timeout_s}s")
    print(f"  Random seed  : {seed}")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Build infrastructure
    # ------------------------------------------------------------------

    # Dry-run uses an explicit semaphore; real mode embeds it in GatedTransport.
    dry_semaphore: asyncio.Semaphore | None = None
    real_client: AsyncOpenAI | None = None
    all_messages: list[list[dict]] = []

    if dry_run:
        dry_semaphore = asyncio.Semaphore(concurrency)
        all_messages = [[] for _ in range(n)]
    else:
        # Lazy imports — .env required, real credentials required
        from app.config import Settings  # type: ignore[import]
        from data.models import RequestBranch  # type: ignore[import]
        from llm.gate import make_semaphore  # type: ignore[import]
        from llm.http_client import create_openrouter_client  # type: ignore[import]
        from llm.prompts.quick import QuickContext, build_initial_messages  # type: ignore[import]

        print("  Loading settings from .env ...")
        settings = Settings()
        llm_sem = make_semaphore(concurrency)
        real_client = create_openrouter_client(
            api_key=settings.openrouter_api_key,
            proxy=settings.llm_proxy,
            base_url=settings.openrouter_base_url,
            semaphore=llm_sem,
        )
        print("  Pre-building synthetic prompts ...")
        for i in range(n):
            d = _synthetic_dream(i + seed)
            ctx = QuickContext(
                user_display_name=d["name"],
                user_life_context=d["life_context"],
                per_session_context=d["per_session"],
                request_branch=RequestBranch.FEEL_STATE,
                dream_transcription=d["dream"],
            )
            all_messages.append(build_initial_messages(ctx))
        print(f"  Client and {n} message sets ready.\n")

    # ------------------------------------------------------------------
    # Generate Poisson inter-arrival times
    # ------------------------------------------------------------------

    rng = random.Random(seed)
    # Exponential(rate = N/arrival_sec) inter-arrivals
    rate = n / max(arrival_sec, 1e-9)
    raw_intervals = [rng.expovariate(rate) for _ in range(n)]
    # Normalize so the cumulative span equals exactly arrival_sec
    total_raw = sum(raw_intervals)
    if total_raw > 0:
        raw_intervals = [iv * arrival_sec / total_raw for iv in raw_intervals]
    arrival_times: list[float] = []
    acc = 0.0
    for iv in raw_intervals:
        acc += iv
        arrival_times.append(acc)

    # ------------------------------------------------------------------
    # Scheduler: launch all N coroutines; each sleeps until its arrival
    # ------------------------------------------------------------------

    results: list[TaskResult] = []
    results_lock = asyncio.Lock()
    wall_start = time.monotonic()
    progress_step = max(1, n // 20)

    async def _launch(arrival_time: float, task_id: int) -> None:
        delay = arrival_time - (time.monotonic() - wall_start)
        if delay > 0.0:
            await asyncio.sleep(delay)

        if dry_run:
            assert dry_semaphore is not None
            result = await _run_task_dry(task_id, dry_semaphore, timeout_s)
        else:
            result = await _run_task_real(
                task_id, real_client, all_messages[task_id], timeout_s
            )

        async with results_lock:
            results.append(result)
            done = len(results)
            if done % progress_step == 0 or done == n:
                elapsed = time.monotonic() - wall_start
                ok = sum(1 for r in results if r.success)
                print(
                    f"  [{elapsed:7.1f}s]  {done:5d}/{n} done"
                    f"  ok={ok}  in-flight={_in_flight:3d}  peak={_peak_in_flight:3d}"
                )

    print("  Launching tasks (Poisson arrival) ...")
    await asyncio.gather(*[_launch(arrival_times[i], i) for i in range(n)])

    if real_client is not None:
        try:
            await real_client.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Compute statistics
    # ------------------------------------------------------------------

    total_elapsed = time.monotonic() - wall_start

    successes       = [r for r in results if r.success]
    errors_429      = [r for r in results if r.error_kind == "429"]
    errors_402      = [r for r in results if r.error_kind == "402"]
    errors_timeout  = [r for r in results if r.error_kind == "timeout"]
    errors_other    = [r for r in results if r.error_kind == "other"]

    all_latencies     = sorted(r.latency_s for r in results)
    success_latencies = sorted(r.latency_s for r in successes)

    p50_all = _percentile(all_latencies, 50)
    p95_all = _percentile(all_latencies, 95)
    p50_ok  = _percentile(success_latencies, 50)
    p95_ok  = _percentile(success_latencies, 95)

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------

    print()
    print("=" * 64)
    print(f"  RESULTS — {'DRY RUN' if dry_run else 'LIVE RUN'}")
    print("=" * 64)
    print(f"  {'Metric':<35} {'Value':>10}")
    print(f"  {'-'*47}")
    print(f"  {'Tasks scheduled':<35} {n:>10}")
    print(f"  {'Wall time':<35} {total_elapsed:>9.1f}s")
    print(f"  {'-'*47}")
    pct = lambda k, d=n: f"{100*k/d:5.1f}%"  # noqa: E731
    print(f"  {'Successes':<35} {len(successes):>6}  {pct(len(successes))}")
    print(f"  {'Errors 429 (rate-limited)':<35} {len(errors_429):>6}  {pct(len(errors_429))}")
    print(f"  {'Errors 402 (quota / no credits)':<35} {len(errors_402):>6}  {pct(len(errors_402))}")
    print(f"  {'Timeouts (outer wall-clock)':<35} {len(errors_timeout):>6}  {pct(len(errors_timeout))}")
    print(f"  {'Other errors':<35} {len(errors_other):>6}  {pct(len(errors_other))}")
    print(f"  {'-'*47}")
    print(f"  {'Latency p50 (all tasks)':<35} {p50_all:>9.2f}s")
    print(f"  {'Latency p95 (all tasks)':<35} {p95_all:>9.2f}s")
    if success_latencies:
        print(f"  {'Latency p50 (successes only)':<35} {p50_ok:>9.2f}s")
        print(f"  {'Latency p95 (successes only)':<35} {p95_ok:>9.2f}s")
    print(f"  {'-'*47}")
    print(f"  {'Peak tasks launched (>= in-gate)':<35} {_peak_in_flight:>10}")
    print(f"  {'Gate concurrency limit':<35} {concurrency:>10}")
    if not dry_run:
        print("  note: in live mode this counts tasks past launch, incl. those queued")
        print("        at the gate; it can exceed the gate limit. The 429-rate is the")
        print("        real saturation signal.")
    print("=" * 64)
    print()

    # ------------------------------------------------------------------
    # Recommendations
    # ------------------------------------------------------------------

    if dry_run:
        print("  Dry-run complete — no real API calls were made.")
        print("  Run without --dry-run to test against the real provider (costs money).")
    else:
        if errors_402:
            print("  CRITICAL: 402 errors detected — OpenRouter balance is zero!")
            print("  Top up balance before re-running.")
        if errors_429:
            print("  WARNING: 429 rate-limits observed — concurrency is AT or ABOVE saturation.")
            print(f"  Step --concurrency down from {concurrency} and re-test.")
        elif _peak_in_flight < concurrency * 0.7:
            print("  Peak in-flight well below the gate limit.")
            print(f"  Try raising --concurrency above {concurrency} to probe higher throughput.")
        else:
            print(f"  No 429s at concurrency={concurrency}. This value may be safe for prod.")
            print("  Consider one more round with a slightly higher value to confirm the ceiling.")
    print()

    # Exit 1 if 402s (billing emergency) or >=5% 429s (real rate-limit signal)
    if errors_402:
        return 2
    if len(errors_429) / max(n, 1) >= 0.05:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    sys.exit(asyncio.run(_async_main(args)))


if __name__ == "__main__":
    main()
