from app.config import Settings


def _base_env(monkeypatch):
    monkeypatch.setenv("BOT_TOKEN", "t")
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")


def test_spend_limit_defaults(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.delenv("USER_SPEND_LIMIT_USD", raising=False)
    monkeypatch.delenv("SPEND_LIMIT_EXEMPT_USERNAMES", raising=False)
    s = Settings(_env_file=None)
    assert s.user_spend_limit_usd == 1.0
    assert s.spend_limit_exempt_set == frozenset()


def test_spend_limit_parses_exempt(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("USER_SPEND_LIMIT_USD", "2.5")
    monkeypatch.setenv("SPEND_LIMIT_EXEMPT_USERNAMES", "@Owner_user, partner_user ,")
    s = Settings(_env_file=None)
    assert s.user_spend_limit_usd == 2.5
    # lowercased, @ stripped, blanks dropped
    assert s.spend_limit_exempt_set == frozenset({"owner_user", "partner_user"})


def test_new_user_spend_limit_default(monkeypatch):
    # Батч 2026-08-18: порог новых поднят 0.5 -> 1.0 (на $0.5 не помещался один разбор).
    _base_env(monkeypatch)
    monkeypatch.delenv("NEW_USER_SPEND_LIMIT_USD", raising=False)
    s = Settings(_env_file=None)
    assert s.new_user_spend_limit_usd == 1.0


def test_new_user_spend_limit_from_env(monkeypatch):
    _base_env(monkeypatch)
    monkeypatch.setenv("NEW_USER_SPEND_LIMIT_USD", "0.25")
    s = Settings(_env_file=None)
    assert s.new_user_spend_limit_usd == 0.25
