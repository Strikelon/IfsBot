import os
import subprocess
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_alembic_upgrade_creates_tables(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "migrate.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221 — alembic CLI must run as a subprocess
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert (
        result.returncode == 0
    ), f"alembic failed:\nstdout={result.stdout}\nstderr={result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda sync_conn: inspect(sync_conn).get_table_names())
        assert "users" in tables
        assert "dreams" in tables
        assert "sessions" in tables
        assert "alembic_version" in tables
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_adds_profile_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0002.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221 — alembic CLI must run as a subprocess
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
            user_cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("users")}
            )
            dream_cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("dreams")}
            )
            usage_cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("llm_usage_events")}
            )
            indexes = await conn.run_sync(
                lambda c: {ix["name"] for ix in inspect(c).get_indexes("llm_usage_events")}
            )

        assert "llm_usage_events" in tables
        assert "profile_life_context" in user_cols
        assert "profile_experience_level" in user_cols
        assert "raw_audio_file_id" in dream_cols
        assert {
            "id",
            "user_id",
            "session_id",
            "service",
            "model",
            "usage_metric",
            "usage_amount",
            "cost_usd",
            "prompt_version",
            "created_at",
        }.issubset(usage_cols)
        assert "ix_llm_usage_events_user_id_created_at" in indexes
        assert "ix_llm_usage_events_session_id" in indexes
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_users_has_unique_telegram_user_id_constraint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "u.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    subprocess.run(  # noqa: ASYNC221 — alembic CLI must run as a subprocess
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ},
    )

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO users (telegram_user_id, locale, created_at, updated_at) "
                    "VALUES (42, 'ru', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                )
            )
        with pytest.raises(IntegrityError):  # UNIQUE constraint violation
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO users (telegram_user_id, locale, created_at, updated_at) "
                        "VALUES (42, 'ru', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)"
                    )
                )
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_adds_session_cols(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0003.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            session_cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("sessions")}
            )
        assert "per_session_context" in session_cols
        assert "request_branch" in session_cols
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_creates_dream_images_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0004_images.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("dream_images")}
            )
        assert {
            "id",
            "dream_id",
            "name",
            "ifs_category",
            "shape",
            "analyzed_in_session_id",
        }.issubset(cols)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_creates_session_state_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """session_state — 1:1 с sessions, JSON-колонка `data`."""
    db_path = tmp_path / "m0004_state.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("session_state")}
            )
        assert {"session_id", "data"}.issubset(cols)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_creates_session_messages_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0004_messages.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("session_messages")}
            )
        expected = {
            "id",
            "session_id",
            "direction",
            "content_type",
            "content",
            "audio_file_id",
            "llm_role_used",
            "llm_phase_label",
            "created_at",
        }
        assert expected.issubset(cols)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_extends_sessions_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0004_sessions.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("sessions")}
            )
        assert {
            "target_image_id",
            "current_phase",
            "p4_substep",
            "pivot_count",
        }.issubset(cols)
    finally:
        await engine.dispose()


@pytest.mark.asyncio
async def test_schema_adds_analytics_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path = tmp_path / "m0013.db"
    monkeypatch.setenv("BOT_TOKEN", "test")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")

    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(  # noqa: ASYNC221
        ["python", "-m", "uv", "run", "alembic", "upgrade", "head"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
        env={**os.environ},
    )
    assert result.returncode == 0, f"alembic failed:\n{result.stdout}\n{result.stderr}"

    engine = create_async_engine(f"sqlite+aiosqlite:///{db_path}")
    try:
        async with engine.connect() as conn:
            tables = await conn.run_sync(lambda c: inspect(c).get_table_names())
            cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("analytics_events")}
            )
            user_cols = await conn.run_sync(
                lambda c: {col["name"] for col in inspect(c).get_columns("users")}
            )
        assert "analytics_events" in tables
        assert {
            "id", "user_id", "telegram_user_id", "event_type",
            "occurred_at", "event_day", "event_metadata",
        } <= cols
        assert "acquisition_source" in user_cols
    finally:
        await engine.dispose()
