"""Unit-тесты shutdown-логики AppDependencies."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

import app.deps as sut
from app.deps import AppDependencies, dispose_dependencies


def _make_deps(
    openai_client: AsyncMock,
    openrouter_client: AsyncMock,
    engine: AsyncMock,
    openrouter_classifier_client: AsyncMock | None = None,
) -> AppDependencies:
    """Собрать AppDependencies с мок-ресурсами (frozen=True, slots=True)."""
    if openrouter_classifier_client is None:
        openrouter_classifier_client = AsyncMock()
    return AppDependencies(
        settings=MagicMock(name="Settings"),
        engine=engine,
        session_factory=MagicMock(name="session_factory"),
        openai_client=openai_client,
        openrouter_client=openrouter_client,
        openrouter_classifier_client=openrouter_classifier_client,
    )


@pytest.mark.asyncio
async def test_dispose_dependencies_runs_all_steps_on_success() -> None:
    openai_client = AsyncMock()
    openrouter_client = AsyncMock()
    openrouter_classifier_client = AsyncMock()
    engine = AsyncMock()
    deps = _make_deps(openai_client, openrouter_client, engine, openrouter_classifier_client)

    await dispose_dependencies(deps)

    openai_client.close.assert_awaited_once()
    openrouter_client.close.assert_awaited_once()
    openrouter_classifier_client.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()


@pytest.mark.asyncio
async def test_dispose_dependencies_isolates_failure_of_openai_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Если openai_client.close() падает — openrouter и engine всё равно закрываются."""
    openai_client = AsyncMock()
    openai_client.close.side_effect = RuntimeError("boom: openai")
    openrouter_client = AsyncMock()
    engine = AsyncMock()
    deps = _make_deps(openai_client, openrouter_client, engine)

    log_warning = MagicMock()
    monkeypatch.setattr(sut.log, "warning", log_warning)

    # dispose НЕ должен пробрасывать исключение наружу
    await dispose_dependencies(deps)

    openai_client.close.assert_awaited_once()
    openrouter_client.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()

    # log.warning должен быть вызван хотя бы для упавшего шага
    assert log_warning.called
    failed_events = [call.args[0] for call in log_warning.call_args_list]
    assert "deps.dispose.openai_client.failed" in failed_events


@pytest.mark.asyncio
async def test_dispose_dependencies_isolates_failure_of_openrouter_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_client = AsyncMock()
    openrouter_client = AsyncMock()
    openrouter_client.close.side_effect = RuntimeError("boom: openrouter")
    engine = AsyncMock()
    deps = _make_deps(openai_client, openrouter_client, engine)

    log_warning = MagicMock()
    monkeypatch.setattr(sut.log, "warning", log_warning)

    await dispose_dependencies(deps)

    openai_client.close.assert_awaited_once()
    openrouter_client.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()
    failed_events = [call.args[0] for call in log_warning.call_args_list]
    assert "deps.dispose.openrouter_client.failed" in failed_events


@pytest.mark.asyncio
async def test_dispose_dependencies_isolates_failure_of_openrouter_classifier_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_client = AsyncMock()
    openrouter_client = AsyncMock()
    openrouter_classifier_client = AsyncMock()
    openrouter_classifier_client.close.side_effect = RuntimeError("boom: classifier")
    engine = AsyncMock()
    deps = _make_deps(openai_client, openrouter_client, engine, openrouter_classifier_client)

    log_warning = MagicMock()
    monkeypatch.setattr(sut.log, "warning", log_warning)

    await dispose_dependencies(deps)

    openai_client.close.assert_awaited_once()
    openrouter_client.close.assert_awaited_once()
    openrouter_classifier_client.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()
    failed_events = [call.args[0] for call in log_warning.call_args_list]
    assert "deps.dispose.openrouter_classifier_client.failed" in failed_events


@pytest.mark.asyncio
async def test_dispose_dependencies_isolates_failure_of_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_client = AsyncMock()
    openrouter_client = AsyncMock()
    engine = AsyncMock()
    engine.dispose.side_effect = RuntimeError("boom: engine")
    deps = _make_deps(openai_client, openrouter_client, engine)

    log_warning = MagicMock()
    monkeypatch.setattr(sut.log, "warning", log_warning)

    await dispose_dependencies(deps)

    openai_client.close.assert_awaited_once()
    openrouter_client.close.assert_awaited_once()
    engine.dispose.assert_awaited_once()
    failed_events = [call.args[0] for call in log_warning.call_args_list]
    assert "deps.dispose.engine.failed" in failed_events


@pytest.mark.asyncio
async def test_app_dependencies_includes_openrouter_classifier_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from app.config import Settings
    from app.deps import create_dependencies, dispose_dependencies

    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.chdir(tmp_path)

    settings = Settings()
    deps = create_dependencies(settings)
    assert deps.openrouter_classifier_client is not None
    assert deps.openrouter_classifier_client is not deps.openrouter_client  # отдельный клиент

    await dispose_dependencies(deps)
