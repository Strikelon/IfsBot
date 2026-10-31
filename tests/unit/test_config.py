import pytest
from pydantic import ValidationError

from app.config import Settings


def test_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("LLM_PROXY", raising=False)
    monkeypatch.chdir(tmp_path)  # чтобы случайно не подхватился реальный .env

    s = Settings()

    assert s.bot_token.get_secret_value() == "123:ABC"
    assert s.database_url == "sqlite+aiosqlite:///./test.db"
    assert s.log_level == "INFO"
    assert s.webhook_enabled is False
    assert s.webhook_base_url is None
    assert s.openai_api_key.get_secret_value() == "sk-test"
    assert s.llm_proxy is None


def test_settings_missing_bot_token_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.delenv("BOT_TOKEN", raising=False)
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "bot_token" in str(exc_info.value).lower()


def test_settings_log_level_override(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.log_level == "DEBUG"


def test_settings_loads_openai_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.chdir(tmp_path)

    s = Settings()

    assert s.openai_api_key.get_secret_value() == "sk-test-123"


def test_settings_llm_proxy_optional_defaults_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("LLM_PROXY", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.llm_proxy is None


def test_settings_llm_proxy_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_PROXY", "http://user:pass@us-proxy.example:8080")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.llm_proxy is not None
    assert s.llm_proxy.get_secret_value() == "http://user:pass@us-proxy.example:8080"


def test_settings_llm_proxy_empty_string_treated_as_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """`.env` ships LLM_PROXY= (empty) — must resolve to None, not SecretStr('')."""
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("LLM_PROXY", "")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.llm_proxy is None


def test_settings_missing_openai_api_key_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "openai_api_key" in str(exc_info.value).lower()


def test_settings_loads_openrouter_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_api_key.get_secret_value() == "sk-or-v1-test"


def test_settings_openrouter_base_url_has_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_BASE_URL", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_base_url == "https://openrouter.ai/api/v1"


def test_settings_openrouter_generator_model_has_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_GENERATOR_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_generator_model == "google/gemini-3-pro"


def test_settings_openrouter_classifier_model_has_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_CLASSIFIER_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_classifier_model == "google/gemini-2.5-flash"


def test_settings_openrouter_classifier_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_CLASSIFIER_MODEL", "google/gemini-flash-1.5")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_classifier_model == "google/gemini-flash-1.5"


def test_settings_openrouter_intent_model_has_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_INTENT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_intent_model == "openai/gpt-4o-mini"


def test_settings_openrouter_intent_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_INTENT_MODEL", "google/gemini-2.5-flash-lite")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_intent_model == "google/gemini-2.5-flash-lite"


def test_settings_openrouter_context_model_has_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Память (memory.extract + компоновка) едет на gemini-3.1-flash-lite.

    Выбор по сухому прогону 2026-08-15: единственная модель, у которой живёт
    механизм обновления ядровых фактов (replaces_outdated).
    """
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("OPENROUTER_CONTEXT_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_context_model == "google/gemini-3.1-flash-lite"


def test_settings_openrouter_context_model_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("OPENROUTER_CONTEXT_MODEL", "openai/gpt-4o-mini")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.openrouter_context_model == "openai/gpt-4o-mini"


def test_settings_memory_compaction_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Уплотнение памяти включено по умолчанию, порог — 10 разрозненных ячеек."""
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("MEMORY_COMPACT_ENABLED", raising=False)
    monkeypatch.delenv("MEMORY_COMPACT_BUDGET_CHARS", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.memory_compact_enabled is True
    assert s.memory_compact_budget_chars == 2500


def test_settings_memory_compaction_kill_switch(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.setenv("MEMORY_COMPACT_ENABLED", "false")
    monkeypatch.setenv("MEMORY_COMPACT_BUDGET_CHARS", "2500")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.memory_compact_enabled is False
    assert s.memory_compact_budget_chars == 2500


def test_settings_missing_openrouter_api_key_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError) as exc_info:
        Settings(_env_file=None)

    assert "openrouter_api_key" in str(exc_info.value).lower()


def test_settings_sentry_dsn_defaults_none(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SENTRY_DSN not set → sentry_dsn is None (Sentry will not be initialized)."""
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.delenv("SENTRY_DSN", raising=False)
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.sentry_dsn is None


def test_settings_sentry_dsn_empty_string_treated_as_none(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """SENTRY_DSN= (empty in .env) must resolve to None, same as llm_proxy."""
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("SENTRY_DSN", "")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.sentry_dsn is None


def test_settings_sentry_dsn_when_set(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """SENTRY_DSN set → sentry_dsn holds the value as SecretStr."""
    monkeypatch.setenv("BOT_TOKEN", "123:ABC")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///./test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("SENTRY_DSN", "https://abc123@o000.ingest.sentry.io/999")
    monkeypatch.chdir(tmp_path)

    s = Settings()
    assert s.sentry_dsn is not None
    assert s.sentry_dsn.get_secret_value() == "https://abc123@o000.ingest.sentry.io/999"
