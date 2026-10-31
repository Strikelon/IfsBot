"""Прод-.env собирается deploy.py — тест ловит забытые переменные.

Исторический дефект: DAILY_SPEND_ALERT_USD не переносился на сервер вообще,
прод молча работал на дефолте из кода.
"""
import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def deploy_module():
    spec = importlib.util.spec_from_file_location(
        "deploy_script", REPO_ROOT / "deploy" / "deploy.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _env() -> dict[str, str]:
    return {
        "db_password": "pg",
        "bot_token": "t",
        "log_level": "INFO",
        "openai_api_key": "sk",
        "llm_proxy": "",
        "openrouter_api_key": "or",
        "openrouter_base_url": "https://x",
        "openrouter_generator_model": "g",
        "openrouter_classifier_model": "c",
        "openrouter_intent_model": "int",
        "openrouter_context_model": "ctx",
        "memory_write_enabled": "true",
        "memory_inject_enabled": "true",
        "memory_compact_enabled": "true",
        "memory_compact_budget_chars": "2500",
        "admin_username": "boss",
        "admin_password_hash": "h",
        "admin_session_secret": "s",
        "admin_id": "1",
        "cookie_secure": "false",
        "admin_tz": "Europe/Moscow",
        "user_spend_limit_usd": "1.0",
        "spend_limit_exempt_usernames": "owner_user",
        "new_user_spend_limit_usd": "0.5",
        "daily_spend_alert_usd": "100",
        "support_contact_url": "https://t.me/support",
    }


def test_prod_env_carries_every_required_setting(deploy_module):
    """Каждая обязательная настройка бота обязана доехать до серверного .env.

    Иначе бот на сервере не стартует: pydantic не найдёт поле без дефолта.
    """
    from app.config import Settings

    text = deploy_module.build_prod_env(_env())
    keys = {line.split("=", 1)[0] for line in text.splitlines() if "=" in line}
    required = {
        name.upper() for name, field in Settings.model_fields.items() if field.is_required()
    }
    assert required <= keys, f"не передаются на сервер: {sorted(required - keys)}"


def test_prod_env_carries_both_spend_limits(deploy_module):
    text = deploy_module.build_prod_env(_env())
    assert "USER_SPEND_LIMIT_USD=1.0" in text
    assert "NEW_USER_SPEND_LIMIT_USD=0.5" in text


def test_prod_env_carries_daily_alert(deploy_module):
    text = deploy_module.build_prod_env(_env())
    assert "DAILY_SPEND_ALERT_USD=100" in text


def test_prod_env_carries_memory_models(deploy_module):
    """Модели памяти и intent'ов не переносились на сервер вовсе (долг с 2026-08-15).

    Прод молча ехал на дефолтах из кода: сменить модель памяти на сервере было
    нельзя, только новым релизом.
    """
    text = deploy_module.build_prod_env(_env())
    assert "OPENROUTER_CONTEXT_MODEL=ctx" in text
    assert "OPENROUTER_INTENT_MODEL=int" in text


def test_prod_env_carries_memory_switches(deploy_module):
    """Рубильники памяти на сервере: без них выключить запись/подстановку/уплотнение
    можно было бы только новым релизом."""
    text = deploy_module.build_prod_env(_env())
    assert "MEMORY_WRITE_ENABLED=true" in text
    assert "MEMORY_INJECT_ENABLED=true" in text
    assert "MEMORY_COMPACT_ENABLED=true" in text
    assert "MEMORY_COMPACT_BUDGET_CHARS=2500" in text




def test_runtime_dirs_still_uploaded(deploy_module):
    """Обратная сторона: то, без чего бот не поедет, исключать нельзя."""
    for name in ("src", "alembic", "deploy"):
        assert name not in deploy_module.SKIP_DIRS, name
