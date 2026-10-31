from admin.deps import AdminSettings


def test_admin_settings_reads_env(monkeypatch):
    monkeypatch.setenv("ADMIN_USERNAME", "boss")
    monkeypatch.setenv("ADMIN_PASSWORD_HASH", "$2b$12$abc")
    monkeypatch.setenv("ADMIN_SESSION_SECRET", "secret")
    monkeypatch.setenv("ADMIN_ID", "555")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("BOT_TOKEN", "123:abc")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = AdminSettings(_env_file=None)  # hermetic: ignore repo .env, use monkeypatched vars only
    assert s.admin_username == "boss"
    assert s.admin_id == 555
    assert s.cookie_secure is True
    assert s.session_ttl_days == 7


def test_effective_bot_token_prefers_dev(monkeypatch):
    for k in ("ADMIN_USERNAME", "ADMIN_PASSWORD_HASH", "ADMIN_SESSION_SECRET", "ADMIN_ID",
              "DATABASE_URL", "BOT_TOKEN", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.setenv(k, "x" if k != "ADMIN_ID" else "1")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("BOT_TOKEN", "prod:token")
    monkeypatch.setenv("BOT_DEV_TOKEN", "dev:token")
    s = AdminSettings(_env_file=None)  # hermetic: ignore repo .env, use monkeypatched vars only
    assert s.effective_bot_token.get_secret_value() == "dev:token"


def test_admin_settings_spend_fields_defaults(monkeypatch):
    monkeypatch.delenv("USER_SPEND_LIMIT_USD", raising=False)
    monkeypatch.delenv("NEW_USER_SPEND_LIMIT_USD", raising=False)
    monkeypatch.delenv("DAILY_SPEND_ALERT_USD", raising=False)
    monkeypatch.delenv("SPEND_LIMIT_EXEMPT_USERNAMES", raising=False)
    s = AdminSettings(
        database_url="sqlite+aiosqlite:///:memory:",
        bot_token="t",
        admin_username="b",
        admin_password_hash="h",
        admin_session_secret="s",
        admin_id=1,
        _env_file=None,
    )
    assert s.user_spend_limit_usd == 1.0
    assert s.new_user_spend_limit_usd == 1.0
    assert s.daily_spend_alert_usd == 100.0
    assert s.spend_limit_exempt_usernames == ""
